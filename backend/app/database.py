# app/database.py

import logging
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from app.config import settings
from app.models.orm import Base

logger = logging.getLogger(__name__)

# --- Engine ---
# The engine is the core SQLAlchemy object that manages the connection
# to PostgreSQL. create_async_engine creates an async version that
# works with FastAPI's async routes without blocking.
#
# asyncpg is the actual driver doing the low-level PostgreSQL communication.
# SQLAlchemy sits on top of asyncpg and gives you the ORM layer.
# The URL scheme "postgresql+asyncpg://" tells SQLAlchemy to use asyncpg.
engine = create_async_engine(
    settings.database_url.replace(
        "postgresql://",
        "postgresql+asyncpg://"
    ),
    # SQLAlchemy needs the +asyncpg dialect in the URL.
    # We replace it dynamically so your .env stays clean
    # with a standard postgresql:// URL.

    echo=settings.debug,
    # echo=True prints every SQL query to the terminal when debug=True.
    # Extremely useful during development — you see exactly what
    # SQLAlchemy is sending to PostgreSQL.
    # Set DEBUG=False in .env to silence it in production.

    pool_size=5,
    # Number of connections kept open permanently in the pool.

    max_overflow=10,
    # Extra connections allowed beyond pool_size when under heavy load.
    # Total max connections = pool_size + max_overflow = 15
)

# --- Session factory ---
# AsyncSessionLocal is a factory that creates new database sessions.
# A session is a unit of work — you open one per request, do your
# DB operations, then close it. The session tracks all changes and
# sends them to PostgreSQL in one transaction.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # expire_on_commit=False means objects stay usable after commit.
    # Without this, accessing product.id after saving would trigger
    # another SELECT query — wasteful and confusing.
)


# --- Lifecycle functions ---

async def connect():
    """Called at FastAPI startup — creates all tables."""
    logger.info("Initializing database...")
    async with engine.begin() as conn:
        # engine.begin() opens a connection and starts a transaction.
        # create_all() creates every table defined in ORM models
        # IF NOT EXISTS — safe to call multiple times.
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified successfully.")


async def disconnect():
    """Called at FastAPI shutdown — closes all connections."""
    await engine.dispose()
    logger.info("Database engine disposed.")


# --- Dependency ---

async def get_db():
    """
    FastAPI dependency — provides a database session per request.
    
    Used in routes like:
        async def my_route(db: AsyncSession = Depends(get_db)):
    
    The session is automatically closed after the request completes,
    even if an exception is raised — the try/finally guarantees this.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
            # commit() saves all changes made during this request.
            # If no changes were made (GET request), commit does nothing.
        except Exception:
            await session.rollback()
            # rollback() undoes all changes if anything went wrong.
            # This ensures you never have partial/corrupted data in the DB.
            raise