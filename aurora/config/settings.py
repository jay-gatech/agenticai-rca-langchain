# Pydantic-based configuration management with fallback
try:
    from pydantic import BaseSettings
except Exception:  # pragma: no cover
    class BaseSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class Settings(BaseSettings):
    debug: bool = False
    log_level: str = "INFO"

def get_settings() -> 'Settings':
    return Settings()
