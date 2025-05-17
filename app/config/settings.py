from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Eureka Settings
    EUREKA_IP: str
    EUREKA_APP_NAME: str = "eum-chatbot"
    EUREKA_INSTANCE_HOST: str
    EUREKA_INSTANCE_PORT: int = 8000
    
    # LLM Provider Settings
    LIGHTWEIGHT_LLM_PROVIDER: str
    HIGH_PERFORMANCE_LLM_PROVIDER: str
    WEB_SEARCH_PROVIDER: str
    
    # Ollama Settings
    LIGHTWEIGHT_OLLAMA_URL: str
    LIGHTWEIGHT_OLLAMA_MODEL: str
    LIGHTWEIGHT_OLLAMA_TIMEOUT: int
    HIGH_PERFORMANCE_OLLAMA_URL: str
    HIGH_PERFORMANCE_OLLAMA_MODEL: str
    HIGH_PERFORMANCE_OLLAMA_TIMEOUT: int
    
    # OpenAI Settings
    LIGHTWEIGHT_OPENAI_API_KEY: str
    LIGHTWEIGHT_OPENAI_MODEL: str
    LIGHTWEIGHT_OPENAI_TIMEOUT: int
    HIGH_PERFORMANCE_OPENAI_API_KEY: str
    HIGH_PERFORMANCE_OPENAI_MODEL: str
    HIGH_PERFORMANCE_OPENAI_TIMEOUT: int
    
    # Groq Settings
    GROQ_API_KEY: str
    GROQ_LIGHTWEIGHT_MODEL: str
    GROQ_HIGHPERFORMANCE_MODEL: str
    
    # Search API Settings
    DUCKDUCKGO_API_KEY: str
    GOOGLE_API_KEY: str
    GOOGLE_CSE_ID: str
    
    # Logging Settings
    LOG_LEVEL: str
    LOG_FILE: str
    
    # Server Settings
    HOST: str
    PORT: int
    DEBUG: bool
    
    # Embedding Settings
    EMBEDDING_MODEL: str
    SEARCH_K: int
    SEARCH_THRESHOLD: float
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings() 