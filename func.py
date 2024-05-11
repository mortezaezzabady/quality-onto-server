from typing import Dict, List, Tuple
import pandas as pd
import os, re
from openai import OpenAI
from tqdm import tqdm
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from . import models as M

df_entities: pd.DataFrame
covrel: pd.DataFrame
rel2id: Dict[str, int]
id2rel: Dict[int, str]
ent2id: Dict[str, int]
id2ent: Dict[int, str]
edg2id: Dict[Tuple[int, int], int]
id2edg: Dict[int, Tuple[int, int]]
E: List[str]
KG: List[Tuple[int, int, int]]
_client: OpenAI
embeddings: Tensor
adj_h2t: Dict[int, List[Tuple[int, int]]]
adj_t2h: Dict[int, List[Tuple[int, int]]]

def chat(messages, LLM='gpt'):
    if LLM == 'gpt':
        response = _client.chat.completions.create(model='gpt-3.5-turbo-0125', messages=messages)
        return response.choices[0].message.content.strip()
    if LLM == 'llama':
        url = 'http://localhost:11434/api/chat'
        headers = {'content-type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps({'model': 'llama3', 'messages': messages, 'stream': False}))
        if response.status_code == 200:
            response_text = response.text
            data = json.loads(response_text)
            return data['message']['content']
    return None

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def extract_entities(text, LLM='gpt'):
    messages = [{'role': 'user', 'content': 'extract entities from the text:\n' + text + '\n(numerize the entities)\nEntities:'}]
    entities = chat(messages)
    pattern = r'^\d+\.\s*(.*)'
    enumerated_items = []
    for line in entities.strip().split('\n'):
        match = re.match(pattern, line)
        if match:
            item = match.group(1)
            enumerated_items.append(item)
    return enumerated_items

def get_embeddings(input):
    tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-large')
    model = AutoModel.from_pretrained('thenlper/gte-large')
    batch_dict = tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def match_entities(entities):
    embs = get_embeddings(entities)
    embs = embs.to(torch.float64)
    labels, ids = [], []
    for i, entity in enumerate(entities):
        scores = (embs[i] @ embeddings.T) * 100
        scores = scores.tolist()
        threshold = 90
        top5 = np.argsort(scores)[::-1][:5]
        for j in top5:
            if scores[j] < threshold:
                break
            if ent2id[E[j]] not in ids:
                ids.append(ent2id[E[j]])
                labels.append(E[j])
    return labels, ids

def find_path(source, target, mode='h2t'):
    adj_list, adj_rev = adj_h2t, adj_t2h
    if mode == 't2t':
        source, target = target, source
        adj_list, adj_rev = adj_t2h, adj_h2t
    visited = set()
    queue = [(source, [])]
    while queue:
        node, path = queue.pop(0)
        if mode == 'h2t':
            if node == target:
                return path
        else:
            if node in adj_rev:
                for r, h in adj_rev[node]:
                    if h == target and h not in visited:
                        return path + [[r, h]]
        if node in visited:
            continue
        visited.add(node)
        if node in adj_list:
            for r, t in adj_list[node]:
                queue.append((t, path + [[r, t]]))
    return []

def get_paths(entities):
    paths = []
    for can1 in entities:
        for can2 in entities:
            if can1 != can2:
                path = find_path(can1, can2, mode='h2t')
                if len(path) > 0:
                    paths.append(id2ent[can1] + ' -> ' + ' -> '.join([id2ent[n[1]] for n in path]))
                path = find_path(can1, can2, mode='h2h')
                if len(path) > 1:
                    paths.append(id2ent[can1] + ' -> ' + ' -> '.join([id2ent[n[1]] for n in path[:-1]]) + ' <- ' + id2ent[can2])
                path = find_path(can1, can2, mode='t2t')
                path = list(reversed(path))
                if len(path) > 1:
                    paths.append(id2ent[can1] + ' <- ' + ' -> '.join([id2ent[n[1]] for n in path[1:]]) + ' -> ' + id2ent[can2])
    return paths

def get_edges(path):
    edges = []
    path = path.replace('->', '#>').replace('<-', '<#')
    positions = [(path[m.start() - 1] + path[m.start()] + path[m.start() + 1]).strip() for m in re.finditer('#', path)]
    nodes = [node.strip() for node in path.replace('#>', '#').replace('<#', '#').split('#')]
    for i in range(len(nodes) - 1):
        head, tail = nodes[i], nodes[i + 1]
        if positions[i] == '<#':
            head, tail = tail, head
        edges.append('edge' + str(edg2id[(ent2id[head], ent2id[tail])]))
    return edges

def get_top_paths(query, hypothesis, paths, K=5):
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
    model.eval()

    p = [[i, [x.strip() for x in path.replace('->', '##').replace('<-', '##').split('##')]] for i, path in enumerate(paths)]
    ps = [[], []]
    for i, path in p:
        s = '##'.join(path)
        r = '##'.join(list(reversed(path)))
        if s in ps[0] or r in ps[0]:
            continue
        ps[0].append(s)
        ps[1].append(i)
    paths = [path for i, path in enumerate(paths) if i in ps[1]]
    chunks = [query] + [sent.strip() for sent in hypothesis.split('.') if len(sent.strip()) > 0]
    pairs = []
    for chunk in chunks:
        for i, path in enumerate(paths):
            pairs.append([chunk, path])
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = scores.reshape(len(chunks), -1).T
    avg_scores = []
    for i, path in enumerate(paths):
        avg_scores.append(scores[i].mean().item())
    topK = np.argsort(avg_scores)[::-1][:K]
    paths = {'path' + str(i + 1): M.Path(text=paths[i], score=avg_scores[i], edges=get_edges(paths[i]), zIndex=0) for i in topK}
    return paths

def get_facts(paths):
    facts = []
    for path in paths:
        path = path.replace('->', '#>').replace('<-', '<#')
        positions = [(path[m.start() - 1] + path[m.start()] + path[m.start() + 1]).strip() for m in re.finditer('#', path)]
        nodes = [node.strip() for node in path.replace('#>', '#').replace('<#', '#').split('#')]
        for i in range(len(nodes) - 1):
            head, tail = nodes[i], nodes[i + 1]
            if positions[i] == '<#':
                head, tail = tail, head
            index = edg2id[(ent2id[head], ent2id[tail])]
            fact = covrel.iloc[index]['Text']
            if fact[-1] != '.':
                fact += '.'
            if fact not in facts:
                facts.append(fact)
                
    return facts

def get_subgraph(entities):
    nodes, edges = {}, {}
    for entity in entities:
        nodes['node' + str(entity)] = M.Node(name=id2ent[entity])
        if entity in adj_h2t:
            for r, t in adj_h2t[entity]:
                edges['edge' + str(edg2id[(entity, t)])] = M.Edge(label=id2rel[r], source='node' + str(entity), target='node' + str(t))
                if t not in entities:
                    nodes['node' + str(t)] = M.Node(name=id2ent[t]) 
        if entity in adj_t2h:
            for r, s in adj_t2h[entity]:
                edges['edge' + str(edg2id[(s, entity)])] = M.Edge(label=id2rel[r], target='node' + str(entity), source='node' + str(s))
                if s not in entities:
                    nodes['node' + str(s)] = M.Node(name=id2ent[s]) 
    return nodes, edges

def prune_graph(entities, nodes, edges, paths, top_paths):
    print(len(nodes), len(edges), len(paths), len(entities))
    new_nodes, new_edges, new_paths = {}, {}, {}
    e_cnt = {}
    for path in paths:
        for edge_id in get_edges(path):
            if edge_id in edges:
                new_edges[edge_id] = edges[edge_id]
                new_nodes[edges[edge_id].source] = nodes[edges[edge_id].source]
                new_nodes[edges[edge_id].target] = nodes[edges[edge_id].target]
    print(len(new_nodes), len(new_edges))
    max_path_size = max([len(path.edges) for path in top_paths.values()])

    for name, path in top_paths.items():
        e = []
        for edge_id in path.edges:
            if edge_id not in e_cnt:
                e_cnt[edge_id] = 1
                new_edges[edge_id + '_' + str(e_cnt[edge_id])] = new_edges[edge_id] 
                del new_edges[edge_id]
            else:
                e_cnt[edge_id] += 1
                new_edges[edge_id + '_' + str(e_cnt[edge_id])] = M.Edge(label='', target=new_edges[edge_id + '_1'].target, source=new_edges[edge_id + '_1'].source)
            e.append(edge_id + '_' + str(e_cnt[edge_id]))
        new_paths[name] = M.Path(text=path.text, score=path.score, edges=e, zIndex=max_path_size - len(e) + 1)
    return new_nodes, new_edges, new_paths

def setup():
    global df_entities, covrel, rel2id, id2rel, ent2id, id2ent, E, KG, _client, embeddings, adj_h2t, adj_t2h, id2edg, edg2id
    _client = OpenAI(api_key='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',) 
    df_entities = pd.read_csv('data/entities.csv') 
    covrel = pd.read_csv('data/covrel.csv')

    rel2id = {rel: i for i, rel in enumerate(covrel['Label'].unique())}
    id2rel = {v: k for k, v in rel2id.items()}
    E = []
    KG = []
    ent2id = {}
    edg2id = {}
    df_entities = df_entities.fillna('thisnanwillreplaceback').apply(lambda x: x.str.lower()).replace('thisnanwillreplaceback', np.nan)
    for index, row in df_entities.iterrows():
        head = row['Head']
        tail = row['Tail']
        rel = covrel.iloc[index]['Label']
        if not pd.isnull(head) and not pd.isnull(tail):
            head = head.strip().lower()
            if head not in ent2id:
                ent2id[head] = len(ent2id)
            tail = tail.strip().lower()
            if tail not in ent2id:
                ent2id[tail] = len(ent2id)
            edg2id[(ent2id[head], ent2id[tail])] = index
            E.append(head)
            E.append(tail)
            KG.append((ent2id[head], rel2id[rel], ent2id[tail]))
    id2ent = {v: k for k, v in ent2id.items()}
    id2edg = {v: k for k, v in edg2id.items()}
    E = list(set(E))
    E.sort()

    adj_h2t, adj_t2h = {}, {}
    for h, r, t in KG:
        if h not in adj_h2t:
            adj_h2t[h] = []
        adj_h2t[h].append((r, t))
        if t not in adj_t2h:
            adj_t2h[t] = []
        adj_t2h[t].append((r, h))


    df = pd.read_csv('data/embeddings.csv')
    embeddings = torch.tensor(df.iloc[:, :].values)
    embeddings = embeddings.to(torch.float64)
