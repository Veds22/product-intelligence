import io
import logging
import imagehash
import httpx
import boto3
import asyncio

from PIL import Image
from botocore.client import Config
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from app.models.orm import ProductORM
from app.config import settings

logger = logging.getLogger(__name__)

# ================================================================
# MinIO client — created once, reused across all requests
# ================================================================

def get_minio_client():
    """
    Creates a boto3 S3 client pointing at MinIO.
    boto3 is Amazon's official AWS SDK for Python.
    Since MinIO is S3-compatible, the exact same boto3 code
    that talks to MinIO locally will talk to real AWS S3 in
    production — just change the endpoint_url in .env.

    signature_version="s3v4" is the auth method MinIO requires.
    Without it, requests get rejected with signature errors.
    """
    return boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
        # region_name is required by boto3 even for MinIO.
        # MinIO ignores it but boto3 throws an error without it.
    )
    
def ensure_bucket_exists(client):
    """
    Creates the MinIO bucket if it doesn't already exist.
    Called once at startup — safe to call multiple times
    because of the try/except on BucketAlreadyOwnedByYou.
    
    A bucket is like a top-level folder in S3/MinIO.
    All product images go into one bucket: "product-images"
    Inside the bucket, each image is stored at: products/{sku}.webp
    """
    try:
        client.create_bucket(Bucket=settings.minio_bucket)
        logger.info(f"Created MinIO bucket: {settings.minio_bucket}")
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
        # Bucket already exists — this is fine, not an error
    except Exception as e:
        logger.exception("Bucket check failed")

# ================================================================
# STEP 1 — Download image from URL
# ================================================================

async def download_image(image_url: str) -> bytes | None:
    """
    Downloads an image from a URL and returns raw bytes.

    Why httpx and not requests?
    requests is synchronous — it blocks the entire server while
    waiting for the download. httpx is async — other requests
    can be handled while this download is in progress.

    timeout=15.0 means if the server doesn't respond in 15 seconds,
    we give up and return None instead of hanging forever.

    follow_redirects=True handles URLs that redirect to the actual
    image — very common with CDNs and product image hosting.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                image_url,
                follow_redirects=True
            )
            response.raise_for_status()
            # raise_for_status() throws an exception if status is 4xx or 5xx
            # e.g. 404 Not Found, 403 Forbidden
            # Without this, a 404 response would silently return empty bytes

            return response.content
            # .content = raw bytes of the response body
            # For an image, this is the binary image data

    except httpx.TimeoutException:
        logger.warning(f"Timeout downloading image: {image_url}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error downloading {image_url}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Failed to download image {image_url}: {e}")
        return None


# ================================================================
# STEP 2 — Convert image to WebP
# ================================================================

def convert_to_webp(image_bytes: bytes) -> bytes | None:
    """
    Converts any image format (JPEG, PNG, GIF, BMP etc.)
    to WebP format and returns the result as bytes.

    Why WebP?
    WebP is a modern image format developed by Google.
    At the same visual quality, WebP is:
      - 25-34% smaller than JPEG
      - 26% smaller than PNG
    For 1M product images, this is a massive storage saving.
    
    img.convert("RGB"):
    Some images have an alpha channel (transparency) — PNG files
    for example. WebP supports transparency but JPEG doesn't.
    Converting to RGB removes the alpha channel safely.
    Without this, saving a transparent PNG as WebP sometimes fails.

    quality=85:
    85 is the sweet spot — visually indistinguishable from 100
    but 40% smaller file size. Going below 75 starts showing
    visible compression artifacts on product images.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Image.open() reads the bytes and auto-detects the format
        # io.BytesIO() wraps bytes in a file-like object so Pillow can read it

        img = img.convert("RGB")

        output = io.BytesIO()
        img.save(output, format="WEBP", quality=85)
        # Save into an in-memory buffer (not to disk)
        # This keeps everything in memory — no temp files to clean up

        return output.getvalue()
        # .getvalue() returns all bytes written to the buffer

    except Exception as e:
        logger.warning(f"Failed to convert image to WebP: {e}")
        return None

# ================================================================
# STEP 3 — Compute pHash fingerprint
# ================================================================

def compute_phash(image_bytes: bytes) -> str | None:
    """
    Computes a perceptual hash (pHash) of the image.

    What is pHash?
    A regular hash (MD5, SHA256) changes completely if even one
    pixel changes. A perceptual hash captures the VISUAL STRUCTURE
    of an image — two photos of the same product from slightly
    different angles will have very similar pHashes.

    How it works:
    1. Resize image to 32x32 pixels (loses fine detail, keeps structure)
    2. Convert to grayscale
    3. Apply DCT (Discrete Cosine Transform) — like JPEG compression
    4. Compare each value to the average → 0 or 1
    5. Result: 64 bits represented as a 16-character hex string

    Example output: "f8e0c0a0b0d0e0f0"

    How we use it in Phase 4 (dedup):
    - Two identical images: distance = 0
    - Same product, different photo: distance = 2-4
    - Completely different products: distance = 20+
    - Our threshold: distance ≤ 4 = probable duplicate
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        hash_value = imagehash.phash(img)
        return str(hash_value)
        # str() converts the hash object to hex string
        # e.g. "f8e0c0a0b0d0e0f0"

    except Exception as e:
        logger.warning(f"Failed to compute pHash: {e}")
        return None
    
    
# ================================================================
# STEP 4 — Upload image to MinIO
# ================================================================

def upload_to_minio(
    image_bytes: bytes,
    sku: str,
    client
) -> str | None:
    """
    Uploads the WebP image to MinIO and returns the storage key.

    Storage key = the path inside the bucket.
    e.g. "products/CHAIR-001.webp"

    This key is stored in the products.image_path column in PostgreSQL.
    When the frontend needs to display an image, it uses this key
    to fetch the image directly from MinIO — not through our API.

    io.BytesIO(image_bytes) wraps the bytes as a file-like object
    because boto3's put_object expects something it can read from,
    not raw bytes.

    ContentType="image/webp" tells MinIO what kind of file this is.
    Without this, MinIO stores it as "application/octet-stream"
    which means browsers won't display it correctly.
    """
    try:
        key = f"products/{sku}.webp"

        client.put_object(
            Bucket=settings.minio_bucket,
            Key=key,
            Body=io.BytesIO(image_bytes),
            ContentType="image/webp"
        )

        logger.info(f"Uploaded image to MinIO: {key}")
        return key

    except Exception as e:
        logger.exception(f"Failed to upload image to MinIO for SKU {sku}")
        return None


# ================================================================
# STEP 5 — Update product in DB with image_path and phash
# ================================================================

async def update_product_image_data(
    product_id: str,
    image_path: str,
    phash: str,
    db: AsyncSession
):
    """
    Updates the product row in PostgreSQL with the MinIO path
    and pHash after the image pipeline completes.

    SQLAlchemy update() builds an UPDATE statement:
    UPDATE products
    SET image_path = ..., phash = ...
    WHERE id = ...

    synchronize_session=False tells SQLAlchemy not to try to
    update any in-memory ORM objects — we're just firing a raw
    UPDATE and don't need the ORM to track the change.
    """
    await db.execute(
        update(ProductORM)
        .where(ProductORM.id == product_id)
        .values(image_path=image_path, phash=phash)
        .execution_options(synchronize_session=False)
    )
    await db.flush()
    logger.info(f"Updated image data for product {product_id}")



# ================================================================
# MAIN — orchestrates all 5 steps
# ================================================================

async def process_product_image(
    product_id: str,
    sku: str,
    image_url: str,
    db: AsyncSession
):
    """
    Orchestrates the full image pipeline for one product.
    Called from ingest.py after a product is inserted.

    Steps run sequentially — each step depends on the previous.
    If any step fails, we log the warning and return gracefully.
    A failed image pipeline does NOT fail the ingest request —
    the product is saved regardless, just without image_path/phash.
    These can be backfilled later via a background job (Phase 4).

    This "fail gracefully" design is intentional:
    - A broken image URL is the seller's problem, not ours
    - The product data itself is valid and should be saved
    - Image processing can be retried independently
    """
    logger.info(f"Starting image pipeline for SKU: {sku}")
    
    # Step 1 — Download
    image_bytes = await download_image(image_url)
    if image_bytes is None:
        logger.warning(f"Skipping image pipeline for {sku} — download failed")
        return

    # Step 2 — Convert to WebP
    webp_bytes = convert_to_webp(image_bytes)
    if webp_bytes is None:
        logger.warning(f"Skipping image pipeline for {sku} — WebP conversion failed")
        return

    # Step 3 — Compute pHash (use original bytes, not WebP)
    # We hash the original image not the WebP because WebP compression
    # can slightly alter pixel values which would change the hash.
    phash = compute_phash(image_bytes)

    # Step 4 — Upload to MinIO
    minio_client = get_minio_client()
    ensure_bucket_exists(minio_client)
    image_path = await asyncio.to_thread(
        upload_to_minio, webp_bytes, sku, minio_client
    )
    
    if image_path is None:
        logger.warning(f"Skipping DB update for {sku} — MinIO upload failed")
        return

    # Step 5 — Update DB
    await update_product_image_data(
        product_id=product_id,
        image_path=image_path,
        phash=phash or "",
        db=db
    )
    logger.info(f"Image pipeline complete for SKU: {sku}")