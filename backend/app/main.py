# app/main.py

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import connect, disconnect
from app.api import ingest

app = FastAPI(
    title="Product Intelligence API",
    version="1.0.0",
    description="Multi-modal product search — text + image hybrid search, "
                "duplicate detection, pricing suggestions, LLM enrichment."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    ingest.router,
    prefix="/products",
    tags=["Ingestion"]
)

@app.on_event("startup")
async def startup():
    from app.services.image_pipeline import get_minio_client, ensure_bucket_exists
    from app.services.embedder import load_models, setup_qdrant
    await connect()
    minio_client = get_minio_client()
    ensure_bucket_exists(minio_client)
    load_models()
    setup_qdrant() 

# Ensure the stdlib logging is configured so module loggers print to console
logging.basicConfig(level=logging.INFO)
    
@app.on_event("shutdown")
async def shutdown():
    await disconnect()

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "product-intelligence-api"}

@app.get("/")
async def root():
    return {
        "message": "Product Intelligence API",
        "docs": "/docs",
        "health": "/health"
    }