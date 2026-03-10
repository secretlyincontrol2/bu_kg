from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings
from datetime import datetime
import secrets
import hashlib

import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

engine = create_engine(
    settings.database_url,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String, index=True, nullable=False)
    key_hash = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_hash(api_key: str) -> str:
    """SHA256 hash of the API key for secure storage."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

def verify_api_key_hash(plain_api_key: str, stored_hash: str) -> bool:
    """Compare the SHA256 hash of a plaintext key against the stored hash."""
    return get_hash(plain_api_key) == stored_hash

def generate_api_key() -> str:
    """Generate a secure runtime API key for the SDK."""
    return "bu_kg_" + secrets.token_urlsafe(32)
