# app/models/orm.py

from sqlalchemy import (
    Column, String, Text, Numeric, CHAR,
    ARRAY, DateTime, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
import uuid


class Base(DeclarativeBase):
    """
    Base class that all ORM models inherit from.
    SQLAlchemy uses this to track all your tables.
    When you call Base.metadata.create_all(), it creates
    every table that inherits from Base automatically.
    """
    pass


class ProductORM(Base):
    """
    This is the SQLAlchemy ORM model — it maps directly to the
    'products' table in PostgreSQL. Each class attribute = one column.
    
    The difference from Pydantic models:
    - Pydantic (product.py) = validates API input/output (request/response)
    - SQLAlchemy (orm.py)   = defines DB structure and runs queries
    Both exist together — they serve different purposes.
    """

    __tablename__ = "products"
    # __tablename__ tells SQLAlchemy which PostgreSQL table this maps to.
    # Must match exactly what you want in the DB.

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        # uuid.uuid4 generates a new UUID automatically for each row.
        # as_uuid=True means SQLAlchemy returns a Python UUID object,
        # not a raw string — cleaner to work with.
    )

    sku = Column(
        String(255),
        unique=True,       # no two products can have same SKU
        nullable=False,    # required — cannot be NULL in DB
        index=True,        # creates an index automatically — fast SKU lookups
    )

    name = Column(
        Text,
        nullable=False
    )

    description = Column(
        Text,
        nullable=True      # optional field
    )

    price = Column(
        Numeric(10, 2),
        # Numeric(10, 2) = up to 10 digits total, 2 after decimal
        # Always use Numeric for money — Float has rounding errors
        # e.g. Float: 299.1 + 0.2 = 299.29999999 (wrong)
        #      Numeric: 299.1 + 0.2 = 299.30 (correct)
        nullable=False
    )

    currency = Column(
        CHAR(3),           # exactly 3 characters — USD, INR, EUR
        nullable=False
    )

    image_url = Column(
        Text,
        nullable=False
    )

    image_path = Column(
        Text,
        nullable=True      # filled after image pipeline runs (Phase 2)
    )

    phash = Column(
        String(64),
        nullable=True      # filled after image pipeline runs (Phase 2)
    )

    category = Column(
        String(255),
        nullable=True,
        index=True         # index for fast category filtering
    )

    brand = Column(
        String(255),
        nullable=True
    )

    source = Column(
        String(50),        # "csv", "api", "scrape", "s3"
        nullable=False
    )

    tags = Column(
        ARRAY(Text),
        default=list,      # default empty list []
        nullable=True
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        # server_default=func.now() means PostgreSQL sets this automatically
        # on INSERT using its own clock — more reliable than Python time
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        # onupdate=func.now() means PostgreSQL updates this automatically
        # every time the row is modified — you never set it manually
        nullable=False
    )