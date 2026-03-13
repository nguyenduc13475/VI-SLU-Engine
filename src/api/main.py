import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .router import router

# Configure standard logging for production
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with Swagger UI metadata
app = FastAPI(
    title="Vi-SLU API (Vietnamese Spoken Language Understanding)",
    description="Stateless NLP Engine parsing Vietnamese natural language commands into IoT execution plans.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS (Cross-Origin Resource Sharing) to allow Gateway/Mobile App connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific IPs/domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the NLP router
app.include_router(router)

@app.get("/", tags=["Health Check"])
async def root() -> dict:
    """
    Health check endpoint to verify if the server is up and running.
    """
    return {
        "status": "online",
        "message": "Welcome to Vi-SLU API. Visit /docs to test the API endpoints."
    }

if __name__ == "__main__":
    # Start the ASGI server
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)