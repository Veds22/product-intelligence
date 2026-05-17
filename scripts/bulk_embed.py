"""
  Embeds all products in PostgreSQL that haven't been embedded yet.
  Run AFTER server is running — models must be loaded.
  Run with: uv run python scripts/bulk_embed.py
"""
import asyncio
import logging
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def bulk_embed():
    """
    Fetches all un-embedded products and triggers embedding
    by calling POST /products/embed/{id} on each one.
    We add a new endpoint for this — see Step 8.
    """
    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=120.0  # embedding takes longer than normal requests
    ) as client:

        # Get all un-embedded products
        response = await client.get("/products/unembedded")
        if response.status_code != 200:
            logger.error(f"Failed to fetch unembedded products: {response.text}")
            return

        products = response.json()
        total = len(products)
        logger.info(f"Found {total} products to embed")

        success = 0
        failed = 0

        for i, product in enumerate(products, start=1):
            try:
                r = await client.post(f"/products/{product['id']}/embed")
                if r.status_code == 200:
                    success += 1
                    logger.info(f"[{i}/{total}] Embedded: {product['sku']}")
                else:
                    failed += 1
                    logger.error(f"[{i}/{total}] Failed: {product['sku']} — {r.text}")
            except Exception as e:
                failed += 1
                logger.error(f"[{i}/{total}] Error: {product['sku']} — {e}")

        print(f"\n{'='*50}")
        print(f"Bulk embed complete!")
        print(f"  Success : {success}")
        print(f"  Failed  : {failed}")
        print(f"  Total   : {total}")
        print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(bulk_embed())