# NOTE: Full implementation referenced in agent-implementations.md (Section 7.1).
# Minimal FastAPI app scaffold to make the repo usable out-of-the-box.
from fastapi import FastAPI
from .routes import router as api_router

app = FastAPI(title="AURORA RCA API")
app.include_router(api_router)
