from pydantic import BaseModel
from typing import Dict, List, Tuple


class ChatInput(BaseModel):
    query: str
    model: str

class Node(BaseModel):
    name: str

class Edge(BaseModel):
    source: str
    target: str
    label: str

class Path(BaseModel):
    text: str
    score: float
    edges: List[str]
    zIndex: int

class Graph(BaseModel):
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    paths: Dict[str, Path]

class ChatOutput(BaseModel):
    hypothesis: str
    response: str
    entities_hyp: List[str]
    entities_kg: List[str]
    new_query: str
    graph: Graph