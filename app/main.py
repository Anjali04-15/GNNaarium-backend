from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.api import router as api_router
from app.auth_routes import router as auth_router
from app.database import connect_to_mongo, close_mongo_connection
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="GNNaarium Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("JWT_SECRET_KEY", "your-secret-key")
)

@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# Mount routers
app.include_router(api_router, prefix="/api")
app.include_router(auth_router)
