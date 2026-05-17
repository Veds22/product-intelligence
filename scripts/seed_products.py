# Generates 500 fake products and inserts them via the API.
# Run this AFTER the server is running:
#   uv run uvicorn app.main:app --reload
# Then in a separate terminal:
#   uv run python scripts/seed_products.py

import httpx
import random
import asyncio
import sys

# ================================================================
# PRODUCT DATA — realistic Indian e-commerce catalog
# ================================================================

CATEGORIES = {
    "Furniture": {
        "brands": ["HomeCraft", "WoodWorks", "UrbanTree", "NestMart", "CasaLiving"],
        "products": [
            ("Wooden Dining Chair", "Solid sheesham wood dining chair with cushioned seat"),
            ("Office Chair", "Ergonomic mesh back office chair with lumbar support"),
            ("Coffee Table", "Round glass top coffee table with wooden legs"),
            ("Bookshelf", "5-tier engineered wood bookshelf with metal frame"),
            ("Sofa Set", "3+1+1 fabric sofa set in grey with wooden armrests"),
            ("Bed Frame", "Queen size solid wood bed frame with storage drawers"),
            ("Wardrobe", "3-door sliding wardrobe with mirror and shelves"),
            ("Study Table", "L-shaped study table with keyboard tray and drawer"),
            ("Recliner Chair", "Single seater fabric recliner with footrest"),
            ("TV Unit", "Wall mounted TV unit with LED backlight and storage"),
        ]
    },
    "Lighting": {
        "brands": ["BrightLife", "LumaHome", "GlowUp", "RadiantCo", "LightHouse"],
        "products": [
            ("Floor Lamp", "Minimalist LED floor lamp with adjustable brightness"),
            ("Desk Lamp", "USB-C rechargeable desk lamp with touch control"),
            ("Ceiling Light", "Round LED ceiling light with remote control"),
            ("Wall Sconce", "Decorative wall sconce with warm white LED"),
            ("Pendant Light", "Industrial style pendant light with Edison bulb"),
            ("Night Light", "Motion sensor LED night light for bedroom"),
            ("String Lights", "10m fairy string lights with 100 warm white LEDs"),
            ("Table Lamp", "Ceramic table lamp with linen shade"),
            ("Track Light", "4-head adjustable track light for living room"),
            ("Solar Garden Light", "Waterproof solar powered garden stake light"),
        ]
    },
    "Electronics": {
        "brands": ["TechPro", "DigiHome", "SmartGear", "ByteBox", "NanoTech"],
        "products": [
            ("Wireless Earbuds", "True wireless earbuds with 30hr battery and ANC"),
            ("Smart Speaker", "WiFi Bluetooth smart speaker with voice assistant"),
            ("Power Bank", "20000mAh fast charging power bank with USB-C PD"),
            ("Webcam", "1080p HD webcam with built-in microphone and ring light"),
            ("Mechanical Keyboard", "TKL mechanical keyboard with blue switches and RGB"),
            ("Mouse Pad", "XXL extended mouse pad with anti-slip base"),
            ("USB Hub", "7-port USB 3.0 hub with individual power switches"),
            ("Monitor Stand", "Adjustable monitor stand with USB hub and cable management"),
            ("Laptop Stand", "Foldable aluminum laptop stand with heat dissipation"),
            ("LED Strip", "5m RGB LED strip with app control and music sync"),
        ]
    },
    "Kitchen": {
        "brands": ["CookMaster", "KitchenPro", "HomeChef", "SpiceBox", "CulinaryArt"],
        "products": [
            ("Air Fryer", "5.5L digital air fryer with 8 preset cooking modes"),
            ("Coffee Maker", "Drip coffee maker with thermal carafe and timer"),
            ("Knife Set", "7-piece German steel knife set with wooden block"),
            ("Cutting Board", "Large bamboo cutting board with juice groove"),
            ("Mixing Bowls", "Set of 5 stainless steel mixing bowls with lids"),
            ("Spice Rack", "Wall mounted spice rack with 12 glass jars"),
            ("Dish Rack", "Stainless steel 2-tier dish drying rack with drip tray"),
            ("Food Processor", "600W food processor with 5 attachments"),
            ("Electric Kettle", "1.7L double wall electric kettle with temperature control"),
            ("Pressure Cooker", "5L stainless steel pressure cooker with safety valve"),
        ]
    },
    "Fitness": {
        "brands": ["FitForce", "ActiveGear", "IronBody", "PulseFit", "ZenSport"],
        "products": [
            ("Yoga Mat", "6mm thick non-slip TPE yoga mat with carry strap"),
            ("Resistance Bands", "Set of 5 latex resistance bands with carry bag"),
            ("Dumbbells", "Pair of adjustable dumbbells 2-20kg with stand"),
            ("Jump Rope", "Speed jump rope with ball bearings and foam handles"),
            ("Pull Up Bar", "Doorframe pull up bar with foam grip — no screws needed"),
            ("Foam Roller", "High density EVA foam roller for muscle recovery"),
            ("Ab Wheel", "Double wheel ab roller with knee pad"),
            ("Gym Gloves", "Leather palm gym gloves with wrist support"),
            ("Water Bottle", "1L BPA free tritan water bottle with time markings"),
            ("Gym Bag", "40L gym duffel bag with shoe compartment and wet pocket"),
        ]
    },
}

# Image seeds — picsum.photos returns consistent images per seed
IMAGE_SEEDS = [
    "chair", "table", "lamp", "desk", "sofa", "shelf", "bed", "wardrobe",
    "light", "phone", "laptop", "keyboard", "speaker", "camera", "watch",
    "kitchen", "coffee", "knife", "bowl", "kettle", "yoga", "gym", "fitness",
    "dumbbell", "bottle", "bag", "plant", "decor", "frame", "mirror"
]

CURRENCIES = ["INR", "INR", "INR", "INR", "USD"]  # mostly INR

SOURCES = ["csv", "api", "scrape"]


def generate_products(count: int = 500) -> list[dict]:
    """
    Generates `count` fake products with realistic data.
    Uses combinations of categories, brands, and product names
    with slight variations to simulate a real catalog.
    """
    products = []
    sku_counter = 1

    categories = list(CATEGORIES.keys())

    while len(products) < count:
        category = random.choice(categories)
        cat_data = CATEGORIES[category]
        brand = random.choice(cat_data["brands"])
        product_name, base_description = random.choice(cat_data["products"])

        # Add slight variation to name so duplicates are realistic but not identical
        variants = ["", " - Premium", " - Standard", " Pro", " Lite", " Plus"]
        variant = random.choice(variants)
        name = product_name + variant

        # Generate realistic price based on category
        price_ranges = {
            "Furniture": (1999, 49999),
            "Lighting": (499, 7999),
            "Electronics": (999, 29999),
            "Kitchen": (599, 15999),
            "Fitness": (299, 9999),
        }
        min_price, max_price = price_ranges[category]
        price = round(random.uniform(min_price, max_price), 2)

        currency = random.choice(CURRENCIES)
        image_seed = random.choice(IMAGE_SEEDS)
        image_url = f"https://picsum.photos/seed/{image_seed}{sku_counter}/400/400"

        sku = f"{category[:3].upper()}-{sku_counter:05d}"

        products.append({
            "sku": sku,
            "name": name,
            "description": base_description,
            "price": price,
            "currency": currency,
            "image_url": image_url,
            "category": category,
            "brand": brand,
            "source": random.choice(SOURCES),
        })

        sku_counter += 1

    return products[:count]


async def seed(count: int = 500, batch_size: int = 50):
    """
    Inserts products via the API in batches.
    Uses the bulk endpoint for speed — 50 products per CSV batch.
    """
    print(f"Generating {count} products...")
    products = generate_products(count)

    # Convert products to CSV format
    import csv
    import io

    headers = ["sku", "name", "description", "price", "currency",
               "image_url", "category", "brand", "source"]

    total_inserted = 0
    total_skipped = 0
    total_failed = 0
    batch_num = 0

    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=60.0   # longer timeout — image pipeline runs per product
    ) as client:

        # Process in batches
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            batch_num += 1

            # Build CSV string for this batch
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            writer.writerows(batch)
            csv_content = output.getvalue()

            # Upload batch
            response = await client.post(
                "/products/bulk",
                params={"skip_images": True},
                files={
                    "file": (
                        f"batch_{batch_num}.csv",
                        io.BytesIO(csv_content.encode("utf-8")),
                        "text/csv"
                    )
                }
            )

            if response.status_code == 200:
                summary = response.json()["summary"]
                total_inserted += summary["inserted"]
                total_skipped += summary["skipped"]
                total_failed += summary["failed"]

                print(
                    f"Batch {batch_num:02d}/{(count // batch_size) + 1} — "
                    f"inserted: {summary['inserted']}, "
                    f"skipped: {summary['skipped']}, "
                    f"failed: {summary['failed']}, "
                    f"time: {summary['time_ms']}ms"
                )
            else:
                print(f"Batch {batch_num} failed: {response.status_code} {response.text}")

    print("\n" + "="*50)
    print(f"Seed complete!")
    print(f"  Total inserted : {total_inserted}")
    print(f"  Total skipped  : {total_skipped}")
    print(f"  Total failed   : {total_failed}")
    print("="*50)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    # Run with custom count: uv run python scripts/seed_products.py 100
    asyncio.run(seed(count=count))