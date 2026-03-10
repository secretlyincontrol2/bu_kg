import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "f6c60fc9"
    
    together_api_key: str
    llm_model_name: str = "Qwen2.5 7B Instruct Turbo"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
