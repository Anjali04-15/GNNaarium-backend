from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from app.datasets import load_builtin_dataset, load_csv_dataset, dataset_stats
from app.run_manager import RunManager

from app.auth_middleware import get_current_user
import asyncio
import io
import pandas as pd
import time
from app.database import save_experiment, get_experiments, delete_experiment, get_experiment_stats
from datetime import datetime

class DatasetConfig(BaseModel):
    name: str = Field(default="Cora", description="Dataset name: Cora, Citeseer, or PubMed")

class ModelConfig(BaseModel):
    name: str = Field(default="GCN", description="Model architecture: GCN, GIN, GAT, GraphSage, GraphTransformer, KA-GNN")
    hidden_dim: int = Field(default=64, description="Hidden layer dimension")
    dropout: float = Field(default=0.5, description="Dropout rate")
    lr: float = Field(default=0.01, description="Learning rate")
    epochs: int = Field(default=50, description="Number of training epochs")
    seed: Optional[int] = Field(default=42, description="Random seed")
    weight_decay: float = Field(default=5e-4, description="Weight decay")

class AttackConfig(BaseModel):
    name: str = Field(description="Attack name: random_edge_flip")
    budget_pct: float = Field(default=0.05, description="Attack budget percentage")

class DefenseConfig(BaseModel):
    name: str = Field(description="Defense name: jaccard")
    threshold: float = Field(default=0.01, description="Defense threshold")

class ExplainerConfig(BaseModel):
    name: str = Field(description="Explainer name: GNNExplainer, PGExplainer, SubgraphX, ProtGNN, GraphMask, NeuronAnalysis")
    node_idx: Optional[str] = Field(default="0", description="Target node index(es) to explain - single number or comma-separated")
    explainer_epochs: Optional[int] = Field(default=100, description="Epochs for optimization-based explainers")
    num_samples: Optional[int] = Field(default=50, description="Number of samples for SubgraphX")
    num_prototypes: Optional[int] = Field(default=5, description="Number of prototypes for ProtGNN")
    sparsity_weight: Optional[float] = Field(default=0.01, description="Sparsity regularization weight")
    lr: Optional[float] = Field(default=0.01, description="Learning rate for explainer optimization")

class RunConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    attack: Optional[AttackConfig] = Field(default=None, description="Attack configuration")
    defense: Optional[DefenseConfig] = Field(default=None, description="Defense configuration")
    explainer: Optional[ExplainerConfig] = Field(default=None, description="Explainer configuration")

router = APIRouter()

BUILTIN_DATASET_NAMES = ["Cora", "Citeseer", "PubMed", "Reddit", "MUTAG", "PROTEINS", "ZINC", "OGBN-Arxiv"]

# All available attacks, defenses, and explainers
ALL_ATTACKS = ["FGSM", "PGD", "Nettack", "Metattack", "CLGA", "Model Inversion"]
ALL_DEFENSES = ["jaccard", "feature_denoising", "adversarial_training", "Gradient Regularization", "GNNGuard", "differential_privacy"]
ALL_EXPLAINERS = ["GNNExplainer", "PGExplainer", "SubgraphX", "ProtGNN", "GraphMask", "NeuronAnalysis"]
runs = {}
custom_datasets = {}  # Store uploaded datasets

@router.get("/datasets")
async def list_datasets():
    return {"builtins": BUILTIN_DATASET_NAMES}

@router.get("/attacks")
async def list_attacks():
    return {"attacks": ALL_ATTACKS}

@router.get("/defenses")
async def list_defenses():
    return {"defenses": ALL_DEFENSES}

@router.get("/explainers")
async def list_explainers():
    return {"explainers": ALL_EXPLAINERS}

@router.get("/models")
async def list_models():
    return {"models": ["GCN", "GIN", "GAT", "GraphSage", "GraphTransformer", "KA-GNN"]}

@router.post("/datasets/upload")
async def upload_dataset(nodes: UploadFile = File(...), edges: UploadFile = File(None)):
    nodes_bytes = await nodes.read()
    nodes_df = pd.read_csv(io.BytesIO(nodes_bytes))
    
    import tempfile
    import os
    temp_dir = tempfile.gettempdir()
    nodes_path = os.path.join(temp_dir, nodes.filename or "nodes.csv")
    nodes_df.to_csv(nodes_path, index=False)
    
    edges_path = None
    if edges:
        edges_bytes = await edges.read()
        edges_df = pd.read_csv(io.BytesIO(edges_bytes))
        edges_path = os.path.join(temp_dir, edges.filename or "edges.csv")
        edges_df.to_csv(edges_path, index=False)
    
    try:
        data = load_csv_dataset(nodes_path, edges_path, edges_optional=True)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
        raise HTTPException(status_code=400, detail=f"Dataset loading failed: {str(e)}")
    
    # Store the dataset for later use
    dataset_id = f"custom_{int(time.time() * 1000)}"
    custom_datasets[dataset_id] = data
    
    stats = dataset_stats(data)
    stats['dataset_id'] = dataset_id
    return stats

@router.post("/runs")
async def start_run(config: RunConfig, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    dataset_name = config.dataset.name
    
    if dataset_name in BUILTIN_DATASET_NAMES:
        data = load_builtin_dataset(dataset_name)
    elif dataset_name.startswith('custom_') and dataset_name in custom_datasets:
        data = custom_datasets[dataset_name]
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset. Please upload a custom dataset first or select a builtin dataset.")
    
    run_id = int(time.time() * 1000) % 1000000
    run_manager = RunManager(config.dict(), data)
    runs[run_id] = run_manager
    
    # Don't start training here - let WebSocket handle it
    return {"run_id": run_id, "status": "started"}

@router.get("/runs/{run_id}")
async def get_run(run_id: int):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run_manager = runs[run_id]
    return {
        "run_id": run_id,
        "status": run_manager.status,
        "config": run_manager.config,
        "metrics": run_manager.get_metrics_summary()
    }

@router.websocket("/ws/runs/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int, token: str = None):
    await websocket.accept()
    
    # Authenticate WebSocket connection
    if not token:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    from app.database import verify_jwt_token
    user_payload = verify_jwt_token(token)
    if not user_payload:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return
    
    if run_id not in runs:
        await websocket.close(code=1008, reason="Run not found")
        return
    run_manager = runs[run_id]

    async def send_event(event):
        try:
            await websocket.send_json(event)
        except Exception:
            pass

    try:
        await run_manager.run_training(send_event)
        
        # Save experiment to MongoDB after completion
        if run_manager.status == "completed":
            experiment_data = {
                "run_id": run_id,
                "timestamp": datetime.utcnow(),
                "config": run_manager.config,
                "results": run_manager.get_metrics_summary(),
                "metrics_history": run_manager.metrics,
                "user_email": user_payload.get("email") if user_payload else None
            }
            try:
                experiment_id = await save_experiment(experiment_data)
                pass
            except Exception:
                pass
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send_event({"type": "error", "message": str(e)})
        except:
            pass

@router.get("/experiments")
async def list_experiments(limit: int = 100, skip: int = 0):
    """Get experiments from MongoDB"""
    try:
        experiments = await get_experiments(limit, skip, None)
        return {"experiments": experiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/experiments/{experiment_id}")
async def remove_experiment(experiment_id: str, current_user: dict = Depends(get_current_user)):
    """Delete experiment from MongoDB for current user"""
    try:
        success = await delete_experiment(experiment_id, current_user.get("email"))
        if success:
            return {"message": "Experiment deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Experiment not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/stats")
async def experiment_statistics():
    """Get experiment statistics"""
    try:
        stats = await get_experiment_stats(None)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))