# app/api/ingest.py

import csv
import io
import time
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services.embedder import embed_and_index_product
from app.services.image_pipeline import process_product_image
from app.models.product import ProductSchema, ProductResponse
from app.models.orm import ProductORM
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ================================================================
# HELPER — fetch product by SKU
# ================================================================

async def get_product_by_sku(sku: str, db: AsyncSession):
    """
    Checks if a product with this SKU already exists.
    Used for idempotency — uploading the same product twice
    returns the existing one instead of creating a duplicate.

    select(ProductORM).where(...) → SELECT * FROM products WHERE sku = $1
    scalars().first()             → return first ORM object or None
    """
    result = await db.execute(
        select(ProductORM).where(ProductORM.sku == sku)
    )
    return result.scalars().first()


# ================================================================
# HELPER — convert ORM object to ProductResponse
# ================================================================

def orm_to_response(product: ProductORM) -> ProductResponse:
    """
    Converts a ProductORM (SQLAlchemy) object to ProductResponse (Pydantic).
    These are different classes — SQLAlchemy for DB, Pydantic for API.
    float(product.price) needed because SQLAlchemy returns Numeric as Decimal.
    Pydantic expects float — Decimal is not automatically converted.
    """
    return ProductResponse(
        id=product.id,
        sku=product.sku,
        name=product.name,
        description=product.description,
        price=float(product.price),
        currency=product.currency,
        image_url=product.image_url,
        image_path=product.image_path,
        category=product.category,
        brand=product.brand,
        source=product.source,
        tags=product.tags or [],
        created_at=product.created_at,
        updated_at=product.updated_at
    )


# ================================================================
# ROUTE 1 — POST /products/ingest
# Single product ingestion
# ================================================================

@router.post(
    "/ingest",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single product",
    description="Validates and stores a single product. Idempotent — re-ingesting same SKU returns existing product."
)
async def ingest_product(
    product: ProductSchema,
    db: AsyncSession = Depends(get_db),
):
    # ── Idempotency check ─────────────────────────────────────────
    existing = await get_product_by_sku(product.sku, db)
    if existing:
        logger.info(f"SKU {product.sku} already exists — returning existing")
        return orm_to_response(existing)

    # ── Insert product ────────────────────────────────────────────
    new_product = ProductORM(
        sku=product.sku,
        name=product.name,
        description=product.description,
        price=product.price,
        currency=product.currency,
        image_url=str(product.image_url),
        category=product.category,
        brand=product.brand,
        source=product.source,
        tags=[]
    )

    db.add(new_product)
    await db.flush()
    # flush() sends INSERT to PostgreSQL within current transaction
    # does NOT commit yet — we need the generated id before returning

    await db.refresh(new_product)
    # refresh() re-reads the row so we have server-generated values:
    # id, created_at, updated_at

    # ── Image pipeline ────────────────────────────────────────────
    # Download image → convert to WebP → upload to MinIO → compute pHash
    # Runs synchronously so image_path is populated before embedding
    await process_product_image(
        product_id=str(new_product.id),
        sku=new_product.sku,
        image_url=str(product.image_url),
        db=db
    )

    await db.refresh(new_product)
    # Refresh again — image_path and phash now populated by pipeline

    # ── Embedding pipeline ────────────────────────────────────────
    # CLIP + DINOv2 + BGE run in parallel → upsert into Qdrant
    # Updates clip_embedded=True and embedded_at in PostgreSQL
    await embed_and_index_product(product=new_product, db=db)

    await db.refresh(new_product)
    # Final refresh — clip_embedded and embedded_at now populated

    logger.info(f"Ingested product: {product.sku}")
    return orm_to_response(new_product)


# ================================================================
# ROUTE 2 — POST /products/bulk
# CSV file upload — many products at once
# ================================================================

@router.post(
    "/bulk",
    status_code=status.HTTP_200_OK,
    summary="Bulk ingest products from CSV",
    description="Upload a CSV file. Returns summary of inserted, skipped, and failed rows."
)
async def bulk_ingest(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    skip_images: bool = False,
    # skip_images=true → skip image pipeline for all rows (fast seeding)
    # skip_images=false (default) → run image pipeline for each row
    # Usage: POST /products/bulk?skip_images=true
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted. Please upload a .csv file."
        )

    start_time = time.time()
    contents = await file.read()

    text = contents.decode("utf-8-sig")
    # utf-8-sig strips the invisible BOM character Excel adds to CSV files.
    # Without this, the first column header becomes "sku" with hidden prefix
    # which breaks field matching completely.

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    logger.debug(f"Bulk ingest received {len(rows)} rows")

    inserted = []
    skipped = []
    failed = []

    for i, row in enumerate(rows, start=2):
        # start=2 because row 1 is the header
        row_num = i
        try:
            # Strip whitespace from all keys and values
            # CSV files often have accidental spaces: " 299.0 " → "299.0"
            cleaned = {}
            for k, v in (row or {}).items():
                if not k:
                    continue
                val = v if v is not None else ""
                cleaned[k.strip()] = val.strip()

            # Convert price string → float (CSV has no types, everything is string)
            if "price" in cleaned and cleaned["price"] != "":
                cleaned["price"] = float(cleaned["price"])

            # Default source if not provided
            if "source" not in cleaned or not cleaned.get("source"):
                cleaned["source"] = "csv"

            # Validate through Pydantic — same rules as single ingest
            # Raises ValidationError if any field is invalid
            product = ProductSchema.model_validate(cleaned)

            # Idempotency check
            existing = await get_product_by_sku(product.sku, db)
            if existing:
                skipped.append(product.sku)
                continue

            # Insert product
            new_product = ProductORM(
                sku=product.sku,
                name=product.name,
                description=product.description,
                price=product.price,
                currency=product.currency,
                image_url=str(product.image_url),
                category=product.category,
                brand=product.brand,
                source=product.source,
                tags=[]
            )
            db.add(new_product)
            await db.flush()
            # Single flush — removed duplicate db.add/flush that was here before
            inserted.append(product.sku)

            # Run image pipeline unless explicitly skipped
            if not skip_images:
                await process_product_image(
                    product_id=str(new_product.id),
                    sku=new_product.sku,
                    image_url=str(product.image_url),
                    db=db
                )

        except Exception as e:
            # Log error but continue to next row
            # One bad row never stops the whole batch
            logger.error(f"Row {row_num} failed for SKU {row.get('sku', 'unknown')}: {e}")
            failed.append({
                "row": row_num,
                "sku": row.get("sku", "unknown"),
                "error": str(e)
            })

    elapsed = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"Bulk ingest complete — "
        f"inserted: {len(inserted)}, "
        f"skipped: {len(skipped)}, "
        f"failed: {len(failed)}"
    )

    return {
        "summary": {
            "total_rows": len(inserted) + len(skipped) + len(failed),
            "inserted": len(inserted),
            "skipped": len(skipped),
            "failed": len(failed),
            "time_ms": elapsed
        },
        "inserted_skus": inserted,
        "skipped_skus": skipped,
        "failed_rows": failed
    }


# ================================================================
# ROUTE 3 — GET /products/unembedded
# Returns all products not yet embedded into Qdrant
# ================================================================

@router.get(
    "/unembedded",
    summary="Get all products not yet embedded into Qdrant"
)
async def get_unembedded_products(
    db: AsyncSession = Depends(get_db)
):
    """
    Returns products where clip_embedded=False.
    Used by bulk_embed.py script to find products needing embedding.
    Also useful for monitoring — shows what's pending in the pipeline.
    """
    result = await db.execute(
        select(ProductORM)
        .where(ProductORM.clip_embedded == False)
        .order_by(ProductORM.created_at.asc())
    )
    products = result.scalars().all()

    return [
        {
            "id": str(p.id),
            "sku": p.sku,
            "name": p.name,
            "image_url": p.image_url,
            "category": p.category,
        }
        for p in products
    ]


# ================================================================
# ROUTE 4 — POST /products/{product_id}/embed
# Embed a single product into Qdrant
# ================================================================

@router.post(
    "/{product_id}/embed",
    summary="Embed a single product into Qdrant"
)
async def embed_single_product(
    product_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetches product from DB and runs the full embedding pipeline.
    Called by bulk_embed.py for each un-embedded product.
    Also useful for manually re-embedding a specific product
    after its image has been updated.
    """
    result = await db.execute(
        select(ProductORM).where(ProductORM.id == product_id)
    )
    product = result.scalars().first()

    if not product:
        raise HTTPException(
            status_code=404,
            detail=f"Product {product_id} not found"
        )

    await embed_and_index_product(product=product, db=db)

    return {
        "status": "ok",
        "sku": product.sku,
        "embedded": True
    }