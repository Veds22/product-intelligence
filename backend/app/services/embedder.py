"""
    Handles all three embedding models and Qdrant operations.
    Models are loaded ONCE at startup and reused for every request.
    All three models run in PARALLEL using asyncio + ThreadPoolExecutor.
"""

import io
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import open_clip
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff, OptimizersConfigDiff
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from app.config import settings
from app.models.orm import ProductORM

logger = logging.getLogger(__name__)

# ================================================================
# MODULE-LEVEL STATE
# Models and clients stored here — loaded once, reused forever.
# None until load_models() and setup_qdrant() are called at startup.
# ================================================================

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_dino_processor = None
_dino_model = None
_bge_model = None
_qdrant_client = None

# ThreadPoolExecutor for running sync embedding functions in threads
_executor = ThreadPoolExecutor(max_workers=3)

# Qdrant collection names — defined once, used everywhere
CLIP_COLLECTION = "products_clip"
DINO_COLLECTION = "products_dino"
BGE_COLLECTION = "products_bge"


# ================================================================
# STARTUP — load models and set up Qdrant
# ================================================================

def load_models():
    """
    Load all three ML models into memory.
    Called ONCE from main.py startup event before server accepts requests.

    Module-level globals are used instead of a class because:
    Simpler, no instantiation needed, Python modules are singletons —
    the same module object is shared across all imports in the process.
    Every file that does `from app.services.embedder import get_bge_embedding`
    uses the same loaded model automatically.

    Loading order doesn't matter — models are independent.
    """
    
    global _clip_model, _clip_preprocess, _clip_tokenizer
    global _dino_processor, _dino_model, _bge_model

    logger.info("Loading CLIP ViT-L/14 model...")
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", 
        pretrained="openai"
    )
    _clip_model.eval()  # Set to eval mode for inference
    _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    logger.info("CLIP model loaded successfully.")
    
    logger.info("Loading DINOv2 ViT-L...")
    _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    _dino_model = AutoModel.from_pretrained("facebook/dinov2-large")
    _dino_model.eval()
    logger.info("DINOv2 model loaded successfully.")
    
    logger.info("Loading BGE-large-en-v1.5...")
    _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    _bge_model.eval()
    logger.info("BGE model loaded successfully.")
    
    
def setup_qdrant():
    """
    Connect to Qdrant and create the 3 collections if they don't exist.
    Called ONCE from main.py startup after load_models().

    recreate_on_collection_exists=True would wipe existing data — we never
    use that. Instead we check existence first and skip if already created.
    This makes startup idempotent — safe to restart the server at any time.
    """
    global _qdrant_client
    logger.info("Connecting to Qdrant...")
    _qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=30,
    )
    
    existing = {c.name for c in _qdrant_client.get_collections().collections}
    
    collections = [
        (CLIP_COLLECTION, 768),
        (DINO_COLLECTION, 1024),
        (BGE_COLLECTION, 1024)
    ]
    
    for name, dim in collections:
        if name not in existing:
            _qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2
                )
            )
            logger.info(f"Created Qdrant collection: {name} ({dim}-dim)")
        else:
            logger.info(f"Qdrant collection already exists: {name} - skipping creation")


# ================================================================
# SYNC EMBEDDING FUNCTIONS
# These are called inside thread pool — must be synchronous.
# PyTorch inference is CPU-bound so it runs in threads,
# not as native coroutines.
# ================================================================
def _clip_image_embed_sync(image_bytes: bytes) -> list[float]:
    """
    Synchronous CLIP image embedding.
    Runs in thread pool — never call directly from async code.
    Use get_clip_image_embedding() instead.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = _clip_preprocess(img).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        features = _clip_model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        return features[0].numpy().tolist()
    
def _clip_text_embed_sync(text: str) -> list[float]:
    """
    Synchronous CLIP text embedding.
    Runs in thread pool — never call directly from async code.
    Use get_clip_text_embedding() instead.
    """
    text_tensor = _clip_tokenizer([text])
    
    with torch.no_grad():
        features = _clip_model.encode_text(text_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        return features[0].numpy().tolist()
    
def _dino_embed_sync(image_bytes: bytes) -> list[float]:
    """
    Synchronous DINOv2 image embedding.
    Extracts CLS token — global visual summary of the image.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _dino_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = _dino_model(**inputs)
        # CLS token = first token = global image representation
        # shape: [batch=1, num_patches+1=257, hidden=1024]
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].numpy().tolist()


def _bge_embed_sync(text: str) -> list[float]:
    """
    Synchronous BGE text embedding.
    Instruction prefix is critical — tells model this is a retrieval query.
    Without it quality drops noticeably (see Phase 2 notes).
    """
    instruction = "Represent this sentence for searching relevant passages: "
    embedding = _bge_model.encode(
        instruction + text,
        normalize_embeddings=True
        # sentence-transformers handles normalisation automatically
    )
    return embedding.tolist()


# ================================================================
# ASYNC WRAPPERS
# Wrap sync functions in run_in_executor so they run in threads
# without blocking the FastAPI event loop.
# ================================================================

async def get_clip_image_embedding(image_bytes: bytes) -> list[float]:
    """
    Async wrapper for CLIP image embedding.
    Runs in thread pool — event loop is not blocked.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _clip_image_embed_sync,
        image_bytes
    )


async def get_clip_text_embedding(text: str) -> list[float]:
    """Async wrapper for CLIP text embedding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _clip_text_embed_sync, text)

async def get_dino_embedding(image_bytes: bytes) -> list[float]:
    """Async wrapper for DINOv2 image embedding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _dino_embed_sync, image_bytes)


async def get_bge_embedding(text: str) -> list[float]:
    """Async wrapper for BGE text embedding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _bge_embed_sync, text)


# ================================================================
# QDRANT OPERATIONS
# ================================================================

def _upsert_to_qdrant(
    product_id: str,
    sku: str,
    name: str,
    price: float,
    currency: str,
    category: str | None,
    brand: str | None,
    clip_vec: list[float],
    dino_vec: list[float],
    bge_vec: list[float]
):
    """
    Upsert all 3 vectors into their respective Qdrant collections.
    Synchronous — called inside thread pool.
    
    Upsert = insert if ID doesn't exist, update if it does.
    Safe to call multiple times — re-embedding a product just overwrites
    the old vectors. No duplicate points, no errors.

    The payload stored alongside each vector contains the fields
    most commonly needed in search results. This avoids a PostgreSQL
    round-trip for every search result. Price, name, category, brand
    are all you need to render a product card in the frontend.
    """
    payload = {
        "sku": sku,
        "name": name,
        "price": price,
        "currency": currency,
        "category": category,
        "brand": brand,
    }
    
    # Product UUID is the id — same across all collections
    # This lets us join results by id after multi-collection search

    point_id = str(product_id)
    
    _qdrant_client.upsert(
        collection_name=CLIP_COLLECTION,
        points=[PointStruct(id=point_id, vector=clip_vec, payload=payload)]
    )

    _qdrant_client.upsert(
        collection_name=DINO_COLLECTION,
        points=[PointStruct(id=point_id, vector=dino_vec, payload=payload)]
    )

    _qdrant_client.upsert(
        collection_name=BGE_COLLECTION,
        points=[PointStruct(id=point_id, vector=bge_vec, payload=payload)]
    )


# ================================================================
# MAIN PIPELINE — embed one product
# ================================================================

async def embed_and_index_product(
    product: ProductORM,
    db: AsyncSession
):
    """
    Full embedding pipeline for one product.
    Called from ingest.py after image pipeline completes.

    Flow:
    1. Get image bytes — from MinIO if available, placeholder if not
    2. Run all 3 models IN PARALLEL using asyncio.gather
    3. Upsert all 3 vectors into Qdrant
    4. Update PostgreSQL — mark product as embedded

    If any step fails, we log and return gracefully.
    Product data is already saved — embedding can be retried
    by the Phase 4 Temporal backfill job.
    """
    logger.info(f"Starting embedding pipeline for SKU: {product.sku}")

    # ── Step 1: Get image bytes ──────────────────────────────────
    image_bytes = await _get_image_bytes(product)

    # ── Step 2: Build text for BGE ───────────────────────────────
    text = _build_product_text(product.name, product.description, product.category)
    
    # ── Step 3: Run all 3 models in parallel ─────────────────────
    try:
        if image_bytes:
            clip_vec, dino_vec, bge_vec = await asyncio.gather(
                get_clip_image_embedding(image_bytes),
                get_dino_embedding(image_bytes),
                get_bge_embedding(text)
            )
        else:
            logger.warning(f"No image for {product.sku} — text-only embedding")
            bge_vec = await get_bge_embedding(text)
            clip_vec = [0.0] * 768  # Placeholder zero vector
            dino_vec = [0.0] * 1024  # Placeholder zero vector  
    except Exception as e:
        logger.error(f"Embedding failed for {product.sku}: {e}")
        return # Embedding failed, but product is saved — can retry later
    
    # ── Step 4: Upsert into Qdrant ───────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            _upsert_to_qdrant,
            str(product.id),
            product.sku,
            product.name,
            float(product.price),
            product.currency,
            product.category,
            product.brand,
            clip_vec,
            dino_vec,
            bge_vec
        )
    except Exception as e:
        logger.error(f"Qdrant upsert failed for {product.sku}: {e}")
        return # Embedding succeeded but indexing failed — can retry 
    
    # ── Step 5: Update PostgreSQL ───────────────────────────────
    try:
        from datetime import datetime, timezone
        await db.execute(
            update(ProductORM)
            .where(ProductORM.id == product.id)
            .values(
                clip_embedded=True,
                embedded_at=datetime.now(timezone.utc)
            )
            .execution_options(synchronize_session=False)
        )
        await db.flush()
    except Exception as e:
        logger.error(f"DB update failed after embedding {product.sku}: {e}")
        return
    logger.info(f"Completed embedding pipeline for SKU: {product.sku}")
    

# ================================================================
# HELPERS
# ================================================================

async def _get_image_bytes(product: ProductORM) -> bytes | None:
    """
    Get image bytes for a product.
    If image_path exists in MinIO → download from MinIO.
    Otherwise → download directly from image_url.
    If both fail → return None (text-only embedding will be used).
    """
    import httpx

    if product.image_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    str(product.image_url),
                    follow_redirects=True
                )
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.warning(f"Could not fetch image for {product.sku}: {e}")

    return None


def _build_product_text(
    name: str,
    description: str | None,
    category: str | None
) -> str:
    """
    Build the text string fed into BGE.
    More context = better embedding quality.
    Pipe separator helps BGE understand these are distinct fields.

    Example output:
    "Ergonomic Office Chair | Furniture | Mesh back with lumbar support"
    """
    parts = [name]
    if category:
        parts.append(category)
    if description:
        parts.append(description)
    return " | ".join(parts)


def get_qdrant_client() -> QdrantClient:
    """Returns the shared Qdrant client. Used by search service in Phase 3."""
    if _qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized. Call setup_qdrant() first.")
    return _qdrant_client