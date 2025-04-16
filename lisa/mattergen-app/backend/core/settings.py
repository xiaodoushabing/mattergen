from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    mongo_host: str = Field(..., env="MONGO_HOST")
    mongo_port: int = Field(27017, env="MONGO_PORT")
    db_name: str = Field("mattergen", env="DB_NAME")
    collection_name: str = Field("lattices", env="COLLECTION_NAME")
    # log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()
