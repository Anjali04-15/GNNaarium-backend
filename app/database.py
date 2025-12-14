from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import jwt
logger = logging.getLogger(__name__)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
JWT_ALGORITHM = "HS256"

class MongoDB:
    def __init__(self):
        self.client = None
        self.database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb_url = os.getenv("MONGODB_URL")
        if not mongodb_url:
            logger.warning("No MONGODB_URL found, skipping database connection")
            return
            
        mongodb.client = AsyncIOMotorClient(mongodb_url)
        mongodb.database = mongodb.client[os.getenv("DATABASE_NAME", "gcn_robustness")]
        
        # Test connection
        await mongodb.client.admin.command('ping')
        logger.info("Connected to MongoDB")
        
    except Exception as e:
        logger.warning(f"Failed to connect to MongoDB: {e}. App will run without database.")
        mongodb.client = None
        mongodb.database = None

async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        logger.info("Disconnected from MongoDB")

async def save_experiment(experiment_data: Dict[str, Any]) -> str:
    """Save experiment to MongoDB"""
    if mongodb.database is None:
        logger.warning("No database connection, skipping experiment save")
        return "no-db"
        
    try:
        collection = mongodb.database.experiments
        
        # Add timestamp if not present
        if 'timestamp' not in experiment_data:
            experiment_data['timestamp'] = datetime.utcnow()
        
        result = await collection.insert_one(experiment_data)
        logger.info(f"Experiment saved with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Error saving experiment: {e}")
        return "error"

async def get_experiments(limit: int = 100, skip: int = 0, user_email: str = None) -> List[Dict[str, Any]]:
    """Retrieve experiments from MongoDB for specific user"""
    if mongodb.database is None:
        return []
        
    try:
        collection = mongodb.database.experiments
        query = {"user_email": user_email} if user_email else {}
        cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        experiments = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string for JSON serialization
        for exp in experiments:
            exp['_id'] = str(exp['_id'])
            
        return experiments
        
    except Exception as e:
        logger.error(f"Error retrieving experiments: {e}")
        return []

async def delete_experiment(experiment_id: str, user_email: str = None) -> bool:
    """Delete experiment from MongoDB for specific user"""
    if mongodb.database is None:
        return False
        
    try:
        from bson import ObjectId
        collection = mongodb.database.experiments
        query = {"_id": ObjectId(experiment_id)}
        if user_email:
            query["user_email"] = user_email
        result = await collection.delete_one(query)
        return result.deleted_count > 0
        
    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        return False

async def get_experiment_stats(user_email: str = None) -> Dict[str, Any]:
    """Get experiment statistics for specific user"""
    if mongodb.database is None:
        return {"total_experiments": 0, "accuracy_stats": {}}
        
    try:
        collection = mongodb.database.experiments
        
        query = {"user_email": user_email} if user_email else {}
        total_count = await collection.count_documents(query)
        
        # Get accuracy distribution
        pipeline = [
            {"$match": {**query, "results.final_val_acc": {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": None,
                "avg_accuracy": {"$avg": "$results.final_val_acc"},
                "max_accuracy": {"$max": "$results.final_val_acc"},
                "min_accuracy": {"$min": "$results.final_val_acc"}
            }}
        ]
        
        accuracy_stats = await collection.aggregate(pipeline).to_list(1)
        
        return {
            "total_experiments": total_count,
            "accuracy_stats": accuracy_stats[0] if accuracy_stats else {}
        }
        
    except Exception as e:
        logger.error(f"Error getting experiment stats: {e}")
        return {"total_experiments": 0, "accuracy_stats": {}}

# User management functions
async def create_user(user_data: Dict[str, Any]) -> Optional[str]:
    """Create a new user"""
    if mongodb.database is None:
        return None
        
    try:
        collection = mongodb.database.users
        result = await collection.insert_one(user_data)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return None

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    if mongodb.database is None:
        return None
        
    try:
        collection = mongodb.database.users
        user = await collection.find_one({"email": email})
        if user:
            user['_id'] = str(user['_id'])
        return user
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None

def create_jwt_token(user_data: Dict[str, Any]) -> str:
    """Create JWT token"""
    payload = {
        "email": user_data["email"],
        "name": user_data["name"],
        "exp": datetime.utcnow().timestamp() + 86400  # 24 hours
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None