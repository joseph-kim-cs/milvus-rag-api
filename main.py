from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api-service")

# Environment variable fallback
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:4050")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# FastAPI instance
app = FastAPI(
    title="RAG Service API",
    description="watsonx.ai - Hybrid search RAG with Milvus",
    version="0.0.1-fastapi",
    servers=[{"url": SERVER_URL}]
)

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register root route
from app.route.root import routes as root_api
app.include_router(root_api.root_api_route)

from app.route.ingest import routes as ingest_api
app.include_router(ingest_api.ingest_api_route)

from app.route.query import routes as query_api
app.include_router(query_api.query_api_route)

'''
# Middleware for trusted hosts
# TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",")
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
'''

# Start logging
logger.info("Starting API service...")

# Application entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4050, log_level="info")
