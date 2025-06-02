# Quality Ontology Server

A FastAPI-based server that provides an interface for querying and analyzing quality-related knowledge graphs. This server enables natural language interaction with a knowledge graph, extracting entities, finding relationships, and generating insights.

## Features

- Natural language query processing
- Entity extraction and matching
- Knowledge graph traversal and path finding
- Graph visualization support
- CORS-enabled API endpoints
- Asynchronous request handling

## API Endpoints

### POST /chat
Processes natural language queries and returns comprehensive results including:
- Generated hypothesis
- Extracted entities
- Knowledge graph matches
- Enhanced query with background knowledge
- Response with context
- Relevant subgraph with nodes, edges, and paths

### POST /graph
Retrieves a subgraph from the knowledge base with specified nodes and edges.

### GET /test
Simple health check endpoint that returns 'hi'.

## Data Models

The server uses the following main data models:

- `ChatInput`: Query and model specification
- `ChatOutput`: Comprehensive response including hypothesis, entities, and graph data
- `Graph`: Knowledge graph structure with nodes, edges, and paths
- `Node`: Graph node representation
- `Edge`: Graph edge representation with source, target, and label
- `Path`: Path representation with text, score, and edge information

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- Windows:
```bash
.venv\Scripts\activate
```
- Unix/MacOS:
```bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
uvicorn app:app --reload
```

## Project Structure

- `app.py`: Main FastAPI application and route definitions
- `models.py`: Pydantic models for data validation
- `func.py`: Core functionality and business logic
- `config.py`: Configuration settings
- `data/`: Directory containing knowledge graph data

## Dependencies

- FastAPI: Web framework
- Pydantic: Data validation
- Uvicorn: ASGI server
- Additional dependencies specified in requirements.txt

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable] 