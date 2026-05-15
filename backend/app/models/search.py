from pydantic import BaseModel, Field, model_validator
from uuid import UUID


class SearchRequest(BaseModel):
    """
    What the client sends when searching.
    At least one of query or image_base64 must be provided.
    """
    query: str | None = Field(default=None, max_length=1000)
    image_base64: str | None = Field(default=None)
    top_k: int = Field(default=20, ge=1, le=100)
    filters: dict | None = Field(default=None)

    @model_validator(mode="after")
    def at_least_one_input(self):
        if self.query is None and self.image_base64 is None:
            raise ValueError(
                "Provide at least one of: query (text) or image_base64 (image)"
            )
        return self
    
    
class SearchResultItem(BaseModel):
    """Single product result with relevance score."""
    id: UUID
    sku: str
    name: str
    price: float
    currency: str
    image_url: str
    image_path: str | None
    category: str | None
    brand: str | None
    tags: list[str]
    score: float


class SearchResponse(BaseModel):
    """Full search response with results and metadata."""
    results: list[SearchResultItem]
    total: int
    query_time_ms: float
    query: str | None = None
    search_type: str