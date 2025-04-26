from fastapi import FastAPI
from config.settings import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
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

app.include_router(ingestion_router.router, prefix=settings.API_V1_STR, tags=["Ingestion"])
app.include_router(scoring_router.router, prefix=settings.API_V1_STR, tags=["Scoring"]) # Placeholder

if __name__ == "__main__":
    # This is for local development testing only.
    # Use Uvicorn directly for production: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
