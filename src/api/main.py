from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from contextlib import asynccontextmanager # Import asynccontextmanager
from config.settings import settings
from src.common.db import Base, engine # Import Base and engine

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables if they don't exist
    # Note: engine.begin() is synchronous, run_sync is for async functions
    # For synchronous table creation, use the engine directly
    # This assumes engine is configured for synchronous operations (default for SQLite)
    with engine.begin() as conn:
         Base.metadata.create_all(conn)
    yield
    # Shutdown: (Optional cleanup)
    # await engine.dispose() # dispose might be needed if using async engine

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan # Add lifespan context manager
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"], # Allow requests from the frontend origin
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)


@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": f"Welcome to the {settings.PROJECT_NAME}!"}

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

# --- API Routers ---
# Import and include routers from the routers module
from src.api.routers import ingestion as ingestion_router
from src.api.routers import scoring as scoring_router # Placeholder for scoring router
from src.api.routers import mvp as mvp_router

app.include_router(ingestion_router.router, prefix=settings.API_V1_STR, tags=["Ingestion"])
app.include_router(scoring_router.router, prefix=settings.API_V1_STR, tags=["Scoring"]) # Placeholder
app.include_router(mvp_router.router, prefix=f"{settings.API_V1_STR}/mvp", tags=["MVP"])

if __name__ == "__main__":
    # This is for local development testing only.
    # Use Uvicorn directly for production: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
