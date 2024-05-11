from typing import Union
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from . import func as F
from . import models as M
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    F.setup()
    yield
    
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.post("/chat")
def chat(input: M.ChatInput) -> M.ChatOutput:
    messages = [{'role': 'user', 'content': input.query}]
    hypothesis = F.chat(messages, input.model)
    entities_hyp = F.extract_entities(hypothesis, input.model)
    entities_kg, ent_ids = F.match_entities(entities_hyp)
    nodes, edges = F.get_subgraph(ent_ids)
    paths = F.get_paths(ent_ids)
    top_paths = F.get_top_paths(input.query, hypothesis, paths)
    nodes, edges, top_paths = F.prune_graph(ent_ids, nodes, edges, paths, top_paths)
    facts = F.get_facts([path.text for name, path in top_paths.items()])
    new_query = input.query + '\nBackground Knowledge:\n' + '\n'.join(facts[:5])
    messages = [{'role': 'user', 'content': new_query}]
    response = F.chat(messages, input.model)
    return M.ChatOutput(hypothesis=hypothesis, entities_hyp=entities_hyp, entities_kg=entities_kg, new_query=new_query, response=response, graph=M.Graph(nodes=nodes, edges=edges, paths=top_paths))

@app.post("/graph")
def graph() -> M.Graph:
    nodes, edges = F.get_subgraph([7, 46])
    path = F.get_edges('vitamin d -> covid-19 <- innate immune response')
    return M.Graph(nodes=nodes, edges=edges)

@app.get("/test")
def test() -> str:
    return 'hi'

 