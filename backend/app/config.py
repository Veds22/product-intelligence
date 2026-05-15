from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra = "ignore" # Ignore extra fields in .env file
    )
    
    # --- Database ---
    database_url: str = "postgresql://product_user:product_pass@127.0.0.1:5432/product_intelligence"
    
    # --- MinIO ---
    minio_endpoint: str
    minio_access_key: str   
    minio_secret_key: str
    minio_bucket: str = "product-images" # Default bucket name
    
    # --- Redis ---
    redis_url: str = "redis://localhost:6379/" # Default Redis URL
    
    # --- Qdrant ---
    qdrant_host: str = "localhost" # Default Qdrant host
    qdrant_port: int = 6333 # Default Qdrant port
    
    # --- Gemini ---
    gemini_api_key: str = "" # To prevent from app cashing the value as empty string, set default to empty string
    
    # --- App ---
    app_env: str = "development" # Default app environment
    debug: bool = True # Default debug mode
    app_port: int = 8000 # Default app port
    
settings = Settings()