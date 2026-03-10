import httpx
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import db
from auth_db import SessionLocal, ApiKey, verify_api_key_hash, generate_api_key, get_hash, init_db
from models import (
    QueryRequest, GraphQueryRequest, QueryResponse, QueryResponseData,
    GraphResponse, GraphResponseData, GraphModel, NodeModel
)
from pydantic import BaseModel

app = FastAPI(title="Babcock Knowledge Graph API", version="0.1.0")
security = HTTPBearer()

# Enable CORS for external SDK requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

def get_db():
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db_session = Depends(get_db)
):
    """Validate API Key against PostgreSQL Database."""
    plain_key = credentials.credentials
    if not plain_key.startswith("bu_kg_"):
        raise HTTPException(status_code=401, detail="Invalid API Key format.")
    
    # Use direct hash lookup for efficiency
    hashed_key = get_hash(plain_key)
    key_record = db_session.query(ApiKey).filter(ApiKey.key_hash == hashed_key).first()
    
    if key_record:
        return plain_key
            
    # Diagnostic: Logging key hunt (useful for Vercel logs)
    all_count = db_session.query(ApiKey).count()
    print(f"Auth failed for key prefix {plain_key[:10]}... Total keys in DB: {all_count}")
    
    raise HTTPException(
        status_code=401, 
        detail=f"Unauthorized. Invalid API Key. (Checked {all_count} keys)"
    )

@app.on_event("shutdown")
def shutdown_event():
    db.close()

# -------------------------------------------------------------------------
# Admin: Generate API Key
# -------------------------------------------------------------------------

class GenerateKeyRequest(BaseModel):
    owner_email: str

@app.post("/v1/admin/generate_key")
async def admin_generate_key(req: GenerateKeyRequest, db_session = Depends(get_db)):
    """Generate a new API key and store its hash in the DB."""
    new_key = generate_api_key()
    key_hash = get_hash(new_key)
    
    db_record = ApiKey(owner=req.owner_email, key_hash=key_hash)
    db_session.add(db_record)
    db_session.commit()
    
    return {"message": "Key generated successfully.", "api_key": new_key, "owner": req.owner_email}

# -------------------------------------------------------------------------
# Graph Query Endpoint
# -------------------------------------------------------------------------

@app.post("/v1/graph", response_model=GraphResponse)
async def graph_query(request: GraphQueryRequest, token: str = Depends(verify_token)):
    """Runs a raw Cypher query dynamically mapping to nodes and edges."""
    try:
        results = db.run_query_with_retry(request.query)
        
        nodes = []
        for index, record in enumerate(results):
            nodes.append(NodeModel(
                id=f"node_{index}", 
                label="ResultNode", 
                properties=record
            ))
            
        graph = GraphModel(nodes=nodes, edges=[])
        return GraphResponse(
            status="success", 
            data=GraphResponseData(graph=graph, context="Raw query executed successfully.")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------------------------------------------------------
# AI LLM Query Endpoint (Text-to-Cypher Pipeline)
# -------------------------------------------------------------------------

BABCOCK_SCHEMA_PROMPT = (
    "You are a Neo4j Cypher expert for the Babcock University Knowledge Graph.\n\n"
    "## Graph Schema (Node Labels & Key Properties)\n"
    "- (s:School) - Name\n"
    "- (p:Program) - Name\n"
    "- (s:Staff) - Staff_Name, Role, Department  (IMPORTANT: use Staff_Name NOT Name)\n"
    "- (c:Course) - Name\n"
    "- (h:Hall) - Name, Location\n"
    "- (c:Cafeteria) - Name, Location\n"
    "- (w:WorshipCenter) - Name, Location\n"
    "- (i:Infrastructure) - Name, Location\n"
    "- (v:Vendor) - Name, Location\n"
    "- (i:Insight) - Category, Summary, Date\n\n"
    "## Relationships\n"
    "- (Program)-[:BELONGS_TO]->(School)\n"
    "- (Staff)-[:WORKS_IN]->(School)\n"
    "- (Course)-[:MANAGED_BY]->(Department)\n"
    "- (Insight)-[:RELATES_TO]->(AnyNode)\n\n"
    "## Few-Shot Examples\n"
    "Q: Who are the lecturers in Computer Science?\n"
    "Cypher: MATCH (s:Staff) WHERE toLower(s.Department) CONTAINS 'computer science' OR toLower(s.Department) CONTAINS 'computing' RETURN s.Staff_Name AS Name, s.Role AS Role ORDER BY s.Staff_Name\n\n"
    "Q: What programs are in School of Computing?\n"
    "Cypher: MATCH (p:Program)-[:BELONGS_TO]->(s:School) WHERE toLower(s.Name) CONTAINS 'computing' RETURN p.Name AS Program\n\n"
    "Q: Where is the Medical Centre?\n"
    "Cypher: MATCH (i:Infrastructure) WHERE toLower(i.Name) CONTAINS 'medical centre' RETURN i.Name AS Name, i.Location AS Location\n\n"
    "Q: Latest announcements about Library?\n"
    "Cypher: MATCH (i:Insight) WHERE toLower(i.Summary) CONTAINS 'library' RETURN i.Category AS Category, i.Summary AS Details ORDER BY i.Date DESC LIMIT 5\n\n"
    "## Rules\n"
    "- Return ONLY the Cypher query, nothing else. No markdown, no explanation.\n"
    "- Use toLower() for case-insensitive matching.\n"
    "- Use Staff_Name (not Name) for Staff nodes.\n"
    "- If the question cannot be answered from this schema, return: MATCH (n) RETURN 'No relevant data found' AS result LIMIT 1\n"
)


@app.post("/v1/query", response_model=QueryResponse)
async def ai_query(request: QueryRequest, token: str = Depends(verify_token)):
    """Text-to-Cypher pipeline: NL -> Cypher -> Neo4j -> LLM Answer."""
    model_name = request.model or settings.llm_model_name

    # Step 1: Generate Cypher from natural language
    cypher_query = None
    try:
        async with httpx.AsyncClient() as http_client:
            cypher_resp = await http_client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.together_api_key}"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": BABCOCK_SCHEMA_PROMPT},
                        {"role": "user", "content": f"Generate a Cypher query for: {request.query}"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 300,
                },
                timeout=30.0,
            )
            cypher_resp.raise_for_status()
            cypher_query = cypher_resp.json()["choices"][0]["message"]["content"].strip()
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cypher generation failed (Model: {model_name}): {str(e)}")

    # Step 2: Execute Cypher against Neo4j
    graph_results = []
    nodes = []
    context_str = "No graph data found."
    try:
        graph_results = db.run_query_with_retry(cypher_query)
        if graph_results:
            context_str = f"Cypher Used: {cypher_query}\n\nResults from Babcock Database:\n"
            for idx, rec in enumerate(graph_results):
                context_str += f"- {rec}\n"
                nodes.append(NodeModel(
                    id=f"rag_node_{idx}",
                    label="Result",
                    properties={k: (str(v) if v is not None else None) for k, v in rec.items()},
                ))
    except Exception as e:
        context_str = f"Cypher query failed: {str(e)}. The generated query was: {cypher_query}"

    # Step 3: Generate final answer with graph context
    prompt = f"User Question: {request.query}\n\n{context_str}\n\nPlease answer the user's question using the database results above."
    sys_prompt = request.system_prompt or "You are a helpful Babcock University AI assistant. Answer questions using the provided database context. Be concise and accurate."

    try:
        async with httpx.AsyncClient() as http_client:
            together_payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            if request.temperature is not None:
                together_payload["temperature"] = request.temperature
            if request.max_tokens is not None:
                together_payload["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                together_payload["top_p"] = request.top_p

            response = await http_client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.together_api_key}"},
                json=together_payload,
                timeout=30.0,
            )
            response.raise_for_status()
            ai_text = response.json()["choices"][0]["message"]["content"]

            return QueryResponse(
                status="success",
                data=QueryResponseData(
                    text_response=ai_text,
                    graph=GraphModel(nodes=nodes, edges=[]),
                ),
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Together AI API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error while querying LLM: {str(e)}")
