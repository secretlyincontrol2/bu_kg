import jwt
from datetime import datetime, timedelta
import httpx
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import db
from auth_db import SessionLocal, ApiKey, get_hash, init_db
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
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Validate JWT Token statelessly using JWT_SECRET."""
    token = credentials.credentials
    try:
        # Decode and verify signature
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload["sub"]  # Returns the owner/email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please refresh your token.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.")

@app.on_event("shutdown")
def shutdown_event():
    db.close()

# -------------------------------------------------------------------------
# Auth: Token Exchange (Permanent Key -> JWT)
# -------------------------------------------------------------------------

class TokenRequest(BaseModel):
    api_key: str

@app.post("/v1/auth/token")
async def exchange_token(req: TokenRequest, db_session = Depends(get_db)):
    """Exchange a permanent API key for a short-lived session JWT."""
    hashed_key = get_hash(req.api_key)
    key_record = db_session.query(ApiKey).filter(ApiKey.key_hash == hashed_key).first()
    
    if not key_record:
        raise HTTPException(status_code=401, detail="Invalid API Key.")

    # Create JWT
    expiration = datetime.utcnow() + timedelta(hours=24) # 24h for balance of security/speed
    payload = {
        "sub": key_record.owner,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer", "expires_in": 86400}

# -------------------------------------------------------------------------
# Graph Query Endpoint
# -------------------------------------------------------------------------

@app.post("/v1/graph", response_model=GraphResponse)
async def graph_query(request: GraphQueryRequest, user: str = Depends(verify_token)):
    """Runs a raw Cypher query dynamically mapping to nodes and edges."""
    try:
        results = db.run_query_with_retry(request.query)
        
        nodes = []
        for index, record in enumerate(results):
            # Record is already sanitized by Neo4jHandler
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
    "## Graph Schema\n"
    "- (s:School) {Name: 'School Name'}\n"
    "- (p:Program) {Name: 'Department/Program Name'}\n"
    "- (st:Staff) {Staff_Name: 'Name', Role: 'Position', Department: 'Dept'}\n\n"
    "## Key Relationships\n"
    "- (p:Program)-[:BELONGS_TO]->(s:School)\n"
    "- (st:Staff)-[:WORKS_IN]->(s:School)\n\n"
    "## Critical Join Logic\n"
    "- If asked about a 'Department', it often maps to a 'Program' node.\n"
    "- To find which School a Department/Program belongs to, use: MATCH (p:Program)-[:BELONGS_TO]->(s:School)\n"
    "- To find which School a Staff member works in, use: MATCH (st:Staff)-[:WORKS_IN]->(s:School)\n\n"
    "## Examples\n"
    "Q: Which school does Computer Science belong to?\n"
    "Cypher: MATCH (p:Program)-[:BELONGS_TO]->(s:School) WHERE toLower(p.Name) CONTAINS 'computer science' RETURN s.Name AS SchoolName\n\n"
    "Q: Who leads the School of Computing?\n"
    "Cypher: MATCH (st:Staff)-[:WORKS_IN]->(s:School) WHERE toLower(s.Name) CONTAINS 'computing' AND (toLower(st.Role) CONTAINS 'dean' OR toLower(st.Role) CONTAINS 'head') RETURN st.Staff_Name AS Dean, st.Role AS Role\n\n"
    "## Rules\n"
    "- Return ONLY Cypher. No markdown. No comments.\n"
    "- Use toLower() for comparisons.\n"
    "- If no match is possible, return: MATCH (n) WHERE false RETURN 'None' LIMIT 1\n"
)


@app.post("/v1/query", response_model=QueryResponse)
async def ai_query(request: QueryRequest, user: str = Depends(verify_token)):
    """Text-to-Cypher pipeline: NL -> Cypher -> Neo4j -> LLM Answer."""
    model_name = request.model or settings.llm_model_name

    # Step 1: Generate Cypher
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
                        {"role": "user", "content": f"Query: {request.query}"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
                timeout=20.0,
            )
            cypher_resp.raise_for_status()
            cypher_query = cypher_resp.json()["choices"][0]["message"]["content"].strip()
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cypher generation failed (Model: {model_name}): {str(e)}")

    # Step 2: Execute Cypher
    nodes = []
    context_str = "No graph data found."
    try:
        graph_results = db.run_query_with_retry(cypher_query)
        if graph_results:
            context_str = f"Database Context:\n"
            for idx, rec in enumerate(graph_results):
                context_str += f"- {rec}\n"
                nodes.append(NodeModel(
                    id=f"node_{idx}",
                    label="Facts",
                    properties=rec,
                ))
    except Exception as e:
        context_str = f"Error fetching data: {str(e)}"

    # Step 3: Answer
    sys_prompt = request.system_prompt or "You are a helpful Babcock University assistant. Use the provided context to answer accurately."
    prompt = f"Question: {request.query}\n\nContext: {context_str}\n\nAnswer:"

    try:
        async with httpx.AsyncClient() as http_client:
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                "temperature": request.temperature if request.temperature is not None else 0.4,
                "max_tokens": request.max_tokens if request.max_tokens is not None else 500,
            }
            if request.top_p: payload["top_p"] = request.top_p

            response = await http_client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.together_api_key}"},
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            ai_text = response.json()["choices"][0]["message"]["content"]

            return QueryResponse(
                status="success",
                data=QueryResponseData(text_response=ai_text, graph=GraphModel(nodes=nodes, edges=[])),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
