import csv
import io
import time
import logging
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.services.image_pipeline import process_product_image
from app.models.product import ProductSchema, ProductResponse
from app.models.orm import ProductORM
from app.database import get_db

logger = logging.getLogger(__name__)

# APIRouter is a mini-application that holds a group of related routes.
# We create one per feature (ingest, search, dedup etc.) and register
# them all in main.py. This keeps each file focused on one responsibility
# instead of dumping every route into main.py.
router = APIRouter()

# ================================================================
# HELPER — fetch product by SKU
# ================================================================

async def get_product_by_sku(sku: str, db: AsyncSession):
    """
    SQLAlchemy select() builds a SELECT query in Python.
    No raw SQL strings — the ORM generates the SQL for you.
    
    select(ProductORM)           → SELECT * FROM products
    .where(ProductORM.sku == sku) → WHERE sku = 'value'
    scalars().first()             → return first result or None
    """
    print(f"Checking for existing SKU: {sku}")  # Debug log for SKU check
    result = await db.execute(
        select(ProductORM).where(ProductORM.sku == sku)
    )
    print(f"SKU check result: {result}")  # Debug log for raw result
    return result.scalars().first()


# ================================================================
# HELPER — convert ORM object to ProductResponse
# ================================================================

def orm_to_response(product: ProductORM) -> ProductResponse:
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
# ================================================================

@router.post(
    "/ingest",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single product",
)
async def ingest_product(
    product: ProductSchema,
    db: AsyncSession = Depends(get_db),
):
    # Check if SKU already exists — idempotency
    existing = await get_product_by_sku(product.sku, db)
    if existing:
        logger.info(f"SKU {product.sku} already exists — returning existing")
        return orm_to_response(existing)

    # Create new ORM object and add to session
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
    # db.add() stages the object — not yet saved to DB.
    # The actual INSERT happens when session.commit() is called,
    # which happens automatically in get_db() after the route finishes.

    await db.flush()
    # flush() sends the INSERT to PostgreSQL within the current transaction
    # but does NOT commit yet. We need this to get the generated id,
    # created_at etc. back from PostgreSQL before returning the response.

    await db.refresh(new_product)
    # refresh() re-reads the row from PostgreSQL so we have
    # server-generated values like id, created_at, updated_at.
    # In ingest_product route, after db.refresh(new_product) add:

    # Inside ingest_product, after refresh:
    await process_product_image(
        product_id=str(new_product.id),
        sku=new_product.sku,
        image_url=str(product.image_url),
        db=db
    )
    
    logger.info(f"Inserted new product: {product.sku}")
    
    await db.refresh(new_product)
    
    return orm_to_response(new_product)


# ================================================================
# ROUTE 2 — POST /products/bulk
# ================================================================

@router.post(
    "/bulk",
    status_code=status.HTTP_200_OK,
    summary="Bulk ingest products from CSV",
)
async def bulk_ingest(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted."
        )

    start_time = time.time()
    contents = await file.read()
    text = contents.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    inserted = []
    skipped = []
    failed = []

    # Convert reader to list so we can log / inspect rows reliably
    rows = list(reader)
    logger.debug(f"Bulk ingest received {len(rows)} rows from CSV")

    for i, row in enumerate(rows, start=2):
        row_num = i
        try:
            # Defensive: handle None values and strip safely
            cleaned = {}
            for k, v in (row or {}).items():
                if not k:
                    continue
                val = v if v is not None else ""
                cleaned[k.strip()] = val.strip()

            if "price" in cleaned and cleaned["price"] != "":
                cleaned["price"] = float(cleaned["price"])
            if "source" not in cleaned or not cleaned.get("source"):
                cleaned["source"] = "csv"
            print(f"Row {row_num}: Cleaned data: {cleaned}")  # Debug log for cleaned data
            product = ProductSchema.model_validate(cleaned)
            print(f"Row {row_num}: Validated product: {product}")  # Debug log for validated product

            existing = await get_product_by_sku(product.sku, db)
            print(f"Row {row_num}: Existing product check: {'found' if existing else 'not found'}")  # Debug log for existing check
            if existing:
                skipped.append(product.sku)
                continue

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
            print(f"Row {row_num}: Inserting SKU {product.sku}")  # Debug log for each insert
            db.add(new_product)
            await db.flush()
            inserted.append(product.sku)

        except Exception as e:
            print(f"ROW {row_num} ERROR: {type(e).__name__}: {e}")  # add this
            failed.append({
                "row": row_num,
                "sku": row.get("sku", "unknown"),
                "error": str(e)
            })

    elapsed = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Bulk ingest — inserted: {len(inserted)}, skipped: {len(skipped)}, failed: {len(failed)}")

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