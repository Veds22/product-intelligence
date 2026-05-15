from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Literal
from datetime import datetime
from uuid import UUID


class ProductSchema(BaseModel):
    """
    INPUT model — what the client sends when creating a product.
    FastAPI automatically validates every field against these rules.
    If anything is wrong, a 422 error is returned with clear details.
    """

    sku: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique product identifier e.g. CHAIR-001"
    )

    name: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Product name/title"
    )

    description: str | None = Field(
        default=None,
        max_length=5000
    )

    price: float = Field(
        ...,
        gt=0,
        description="Product price — must be positive"
    )

    currency: str = Field(
        ...,
        pattern=r"^[A-Z]{3}$",
        description="ISO currency code e.g. USD, INR, EUR"
    )

    image_url: HttpUrl = Field(
        ...,
        description="Public URL of the product image"
    )

    category: str | None = Field(default=None, max_length=255)
    brand: str | None = Field(default=None, max_length=255)

    source: Literal["csv", "api", "scrape", "s3"] = Field(
        ...,
        description="Where this product came from"
    )

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        return v.upper()

    @field_validator("sku", "name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ProductDB(ProductSchema):
    """
    DB model — extends ProductSchema with fields WE generate,
    not the client. Added after validation, before saving to DB.
    """
    id: UUID
    image_path: str | None = None
    phash: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
    

class ProductResponse(BaseModel):
    """
    RESPONSE model — what the API returns to the client.
    Excludes internal fields. This is what the frontend receives.
    """
    id: UUID
    sku: str
    name: str
    description: str | None
    price: float
    currency: str
    image_url: str
    image_path: str | None
    category: str | None
    brand: str | None
    source: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}
    
