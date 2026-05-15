# tests/test_ingest.py

import pytest
import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from app.main import app

pytestmark = pytest.mark.asyncio


# ================================================================
# HELPERS
# ================================================================

def make_mock_orm(sku: str, name: str = "Test Wooden Chair"):
    """
    Creates a fake ORM object that looks like a ProductORM instance.
    Used to simulate what SQLAlchemy returns from the DB.
    MagicMock lets us set any attribute on it freely.
    """
    mock = MagicMock()
    mock.id = uuid.uuid4()
    mock.sku = sku
    mock.name = name
    mock.description = "A test product"
    mock.price = 1999.00
    mock.currency = "INR"
    mock.image_url = "https://picsum.photos/seed/pytest1/400/400"
    mock.image_path = None
    mock.category = "Furniture"
    mock.brand = "TestBrand"
    mock.source = "api"
    mock.tags = []
    mock.created_at = datetime.now(timezone.utc)
    mock.updated_at = datetime.now(timezone.utc)
    return mock


def make_mock_session(existing_product=None):
    """
    Creates a mock SQLAlchemy AsyncSession.

    existing_product controls what get_product_by_sku returns:
    - None    → product does not exist (fresh insert)
    - mock    → product exists (duplicate SKU scenario)

    refresh is mocked to populate id/created_at/updated_at
    on the ORM object passed to it — simulates what PostgreSQL
    does when it generates these server-side values.
    """
    mock_session = AsyncMock()

    # Mock execute() — used by get_product_by_sku SELECT
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = existing_product
    mock_session.execute = AsyncMock(return_value=mock_result)

    # Mock flush() — sends INSERT to DB within transaction
    mock_session.flush = AsyncMock()

    # Mock add() — stages ORM object (sync, not async)
    mock_session.add = MagicMock()

    # Mock commit() and rollback()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    # Mock refresh() — simulates PostgreSQL populating server-generated fields
    # Sets id, created_at, updated_at on whatever object is passed to refresh()
    async def mock_refresh(obj):
        obj.id = uuid.uuid4()
        obj.created_at = datetime.now(timezone.utc)
        obj.updated_at = datetime.now(timezone.utc)
    mock_session.refresh = mock_refresh

    return mock_session


def apply_mock_db(existing_product=None):
    """
    Applies mock DB session as FastAPI dependency override.
    Call app.dependency_overrides.clear() after the test request.
    """
    from app import database

    mock_session = make_mock_session(existing_product)

    async def mock_get_db():
        yield mock_session

    app.dependency_overrides[database.get_db] = mock_get_db
    return mock_session


# ================================================================
# FIXTURE — base valid product payload
# ================================================================

@pytest.fixture(scope="session")
def valid_product():
    """
    Base valid product payload reused across tests.
    Uses a random SKU suffix so it never conflicts with real DB data.
    scope="session" — created once for all tests.
    """
    return {
        "sku": f"TEST-{uuid.uuid4().hex[:8]}",
        "name": "Test Wooden Chair",
        "description": "A test product for pytest",
        "price": 1999.00,
        "currency": "INR",
        "image_url": "https://picsum.photos/seed/pytest1/400/400",
        "category": "Furniture",
        "brand": "TestBrand",
        "source": "api"
    }


# ================================================================
# TEST 1 — valid product insert
# ================================================================

async def test_ingest_valid_product(client, valid_product):
    """
    Happy path — valid product returns 201 with all fields.
    DB is mocked — no real PostgreSQL call.
    Image pipeline is mocked — no real image download.
    """
    apply_mock_db(existing_product=None)  # product does not exist yet

    with patch("app.api.ingest.process_product_image", new_callable=AsyncMock):
        response = await client.post("/products/ingest", json=valid_product)

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert data["sku"] == valid_product["sku"]
    assert data["name"] == valid_product["name"]
    assert data["price"] == valid_product["price"]
    assert data["currency"] == "INR"
    assert "id" in data
    assert "created_at" in data


# ================================================================
# TEST 2 — duplicate SKU (idempotency)
# ================================================================

async def test_ingest_duplicate_sku(client, valid_product):
    """
    Inserting same SKU twice returns existing product — no duplicate.
    Mock returns an existing product on the SKU check.
    """
    existing = make_mock_orm(valid_product["sku"])
    apply_mock_db(existing_product=existing)  # product already exists

    r1 = await client.post("/products/ingest", json=valid_product)

    app.dependency_overrides.clear()

    # Apply mock again for second request
    apply_mock_db(existing_product=existing)

    r2 = await client.post("/products/ingest", json=valid_product)

    app.dependency_overrides.clear()

    assert r1.status_code == 201
    assert r2.status_code == 201
    # Both return same id — idempotency confirmed
    assert r1.json()["id"] == r2.json()["id"]


# ================================================================
# TEST 3 — invalid price
# ================================================================

async def test_ingest_invalid_price_zero(client, valid_product):
    """
    Price = 0 rejected by Pydantic before DB is touched.
    No mock needed — Pydantic validation happens before get_db() runs.
    """
    payload = {**valid_product, "price": 0}
    response = await client.post("/products/ingest", json=payload)
    assert response.status_code == 422


async def test_ingest_invalid_price_negative(client, valid_product):
    payload = {**valid_product, "price": -500}
    response = await client.post("/products/ingest", json=payload)
    assert response.status_code == 422


# ================================================================
# TEST 4 — invalid currency
# ================================================================

async def test_ingest_invalid_currency(client, valid_product):
    """
    Currency must be exactly 3 uppercase letters.
    Pydantic rejects anything else before DB is touched.
    """
    for bad_currency in ["us", "USDD", "123", ""]:
        payload = {**valid_product, "currency": bad_currency}
        response = await client.post("/products/ingest", json=payload)
        assert response.status_code == 422, f"Expected 422 for currency: {bad_currency}"


# ================================================================
# TEST 5 — missing required fields
# ================================================================

async def test_ingest_missing_sku(client, valid_product):
    payload = {k: v for k, v in valid_product.items() if k != "sku"}
    response = await client.post("/products/ingest", json=payload)
    assert response.status_code == 422


async def test_ingest_missing_name(client, valid_product):
    payload = {k: v for k, v in valid_product.items() if k != "name"}
    response = await client.post("/products/ingest", json=payload)
    assert response.status_code == 422


async def test_ingest_invalid_source(client, valid_product):
    payload = {**valid_product, "source": "manual"}
    response = await client.post("/products/ingest", json=payload)
    assert response.status_code == 422


# ================================================================
# TEST 6 — CSV bulk upload
# ================================================================

async def test_bulk_ingest_valid_csv(client):
    """
    3 valid rows — all should be inserted.
    DB mocked — no real PostgreSQL.
    """
    unique = uuid.uuid4().hex[:8]
    csv_content = "\n".join([
        "sku,name,description,price,currency,image_url,category,brand,source",
        f"BULK-{unique}-1,Chair One,First,1999.00,INR,https://picsum.photos/seed/b1/400/400,Furniture,Brand,csv",
        f"BULK-{unique}-2,Chair Two,Second,2999.00,INR,https://picsum.photos/seed/b2/400/400,Furniture,Brand,csv",
        f"BULK-{unique}-3,Chair Three,Third,3999.00,INR,https://picsum.photos/seed/b3/400/400,Furniture,Brand,csv",
    ])

    apply_mock_db(existing_product=None)

    response = await client.post(
        "/products/bulk",
        files={"file": ("products.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["inserted"] == 3
    assert data["summary"]["failed"] == 0


async def test_bulk_ingest_partial_failure(client):
    """
    1 valid row, 1 invalid (negative price).
    Valid → inserted, invalid → failed.
    Entire batch does NOT fail because of one bad row.
    """
    unique = uuid.uuid4().hex[:8]
    csv_content = "\n".join([
        "sku,name,description,price,currency,image_url,category,brand,source",
        f"GOOD-{unique},Good Product,Valid,1999.00,INR,https://picsum.photos/seed/g1/400/400,Furniture,Brand,csv",
        f"BAD-{unique},Bad Product,Invalid,-500,INR,https://picsum.photos/seed/bad1/400/400,Furniture,Brand,csv",
    ])

    apply_mock_db(existing_product=None)

    response = await client.post(
        "/products/bulk",
        files={"file": ("products.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["inserted"] == 1
    assert data["summary"]["failed"] == 1


async def test_bulk_ingest_wrong_file_type(client):
    """Non-CSV file should return 400 — no DB needed."""
    response = await client.post(
        "/products/bulk",
        files={"file": ("data.json", io.BytesIO(b'{"sku": "X"}'), "application/json")}
    )
    assert response.status_code == 400