# Run this to understand embeddings before building the full pipeline.
# This is a learning experiment — not production code.
# Run with: uv run python scripts/embedding_experiment.py

import torch
import open_clip
import numpy as np
from PIL import Image
import httpx
import asyncio
import io


# ================================================================
# STEP 1 — Load CLIP model
# ================================================================

print("Loading CLIP model... (takes 10-30 seconds first time)")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")
print("Model loaded!\n")


# ================================================================
# STEP 2 — Download two product images
# ================================================================

async def download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, follow_redirects=True)
        return r.content


async def get_images():
    print("Downloading images...")
    img1_bytes = await download("https://picsum.photos/seed/chair1/400/400")
    img2_bytes = await download("https://picsum.photos/seed/chair2/400/400")
    img3_bytes = await download("https://picsum.photos/seed/phone1/400/400")
    print("Images downloaded!\n")
    return img1_bytes, img2_bytes, img3_bytes

img1_bytes, img2_bytes, img3_bytes = asyncio.run(get_images())


# ================================================================
# STEP 3 — Embed the images
# ================================================================

def embed_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy()


def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy()


print("Embedding images and text...")
vec_img1 = embed_image(img1_bytes)   # image 1
vec_img2 = embed_image(img2_bytes)   # image 2 (similar seed)
vec_img3 = embed_image(img3_bytes)   # image 3 (different seed)

vec_text_chair = embed_text("wooden dining chair furniture")
vec_text_phone = embed_text("smartphone mobile phone electronics")
print("Done!\n")


# ================================================================
# STEP 4 — Compute similarities
# ================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
    # dot product of normalised vectors = cosine similarity


print("=" * 55)
print("EMBEDDING EXPERIMENT RESULTS")
print("=" * 55)

print(f"\n📐 Vector shape: {vec_img1.shape}")
print(f"   Each image/text → {vec_img1.shape[0]} numbers")
print(f"   First 5 values of image1 vector: {vec_img1[:5].round(4)}")

print("\n📊 Cosine Similarity Scores:")
print(f"   Image1 vs Image2 (similar seeds) : {cosine_similarity(vec_img1, vec_img2):.4f}")
print(f"   Image1 vs Image3 (different seed): {cosine_similarity(vec_img1, vec_img3):.4f}")
print(f"   Image1 vs 'chair' text           : {cosine_similarity(vec_img1, vec_text_chair):.4f}")
print(f"   Image1 vs 'phone' text           : {cosine_similarity(vec_img1, vec_text_phone):.4f}")
print(f"   'chair' text vs 'phone' text     : {cosine_similarity(vec_text_chair, vec_text_phone):.4f}")

print("\n🔍 Interpretation:")
print("   Score > 0.85 = very similar")
print("   Score 0.70-0.85 = similar")
print("   Score < 0.50 = different")

print("\n✅ Key insight:")
print("   Text and images share the SAME vector space in CLIP.")
print("   'chair' text vector is close to chair image vectors.")
print("   This is what makes text-to-image search work.")
print("=" * 55)