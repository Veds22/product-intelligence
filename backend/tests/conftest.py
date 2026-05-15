# tests/conftest.py
# Mock the DB entirely in tests that need it.
# Real DB integration is verified manually through /docs.
# Tests verify logic — validation, response format, error handling.
#
# The only real external calls that happen in tests:
# - MinIO bucket check at startup (sync, no loop issues)
# - httpx calls to FastAPI (handled by ASGITransport, no real network)

import sys
import asyncio

# MUST be set before any asyncio/sqlalchemy/asyncpg imports
# WindowsSelectorEventLoopPolicy is the only loop compatible with
# asyncpg on Windows — ProactorEventLoop (default in Python 3.10+)
# causes "attached to a different loop" errors with asyncpg

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.image_pipeline import get_minio_client, ensure_bucket_exists


@pytest.fixture(scope="session")
def event_loop():
    """
    Single event loop for the entire test session.
    scope="session" means one loop is created and shared across
    ALL tests — no loop switching between tests which is what
    causes the asyncpg "different loop" error.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_app():
    """
    Runs once before any test in the session.
    autouse=True means every test gets this automatically.
    Only sets up MinIO bucket — DB is mocked per test.
    """
    minio_client = get_minio_client()
    ensure_bucket_exists(minio_client)
    yield
    # no DB disconnect needed — we never connect in tests


@pytest_asyncio.fixture(scope="session")
async def client():
    """
    Single shared HTTP test client for all tests.
    scope="session" — created once, reused across all tests.
    ASGITransport connects directly to the FastAPI app
    without starting a real server.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac