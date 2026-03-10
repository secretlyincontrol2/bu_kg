from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

class GraphQueryRequest(BaseModel):
    query: str

# ---------------------------------------------------------
# Response Shared Models (matching SDK structure)
# ---------------------------------------------------------

class NodeModel(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class EdgeModel(BaseModel):
    from_node: str = Field(alias="from")
    to: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"populate_by_name": True}

class GraphModel(BaseModel):
    nodes: List[NodeModel] = Field(default_factory=list)
    edges: List[EdgeModel] = Field(default_factory=list)

# ---------------------------------------------------------
# Endpoint Response Models
# ---------------------------------------------------------

class QueryResponseData(BaseModel):
    text_response: str
    graph: GraphModel

class QueryResponse(BaseModel):
    status: str
    data: QueryResponseData

class GraphResponseData(BaseModel):
    graph: GraphModel
    context: Optional[str] = None

class GraphResponse(BaseModel):
    status: str
    data: GraphResponseData
