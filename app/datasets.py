from torch_geometric.datasets import Planetoid, Reddit, TUDataset
from torch_geometric.data import Data
import torch
import pandas as pd
import os
import tempfile
from sklearn.model_selection import train_test_split

try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    PygNodePropPredDataset = None

BUILTIN_DATASETS = ["Cora", "Citeseer", "PubMed", "Reddit", "MUTAG", "PROTEINS", "ZINC", "OGBN-Arxiv"]

def load_builtin_dataset(name):
    if name not in BUILTIN_DATASETS:
        raise ValueError(f"Dataset {name} not supported")
    
    temp_dir = tempfile.gettempdir()
    
    # Citation networks
    if name in ["Cora", "Citeseer", "PubMed"]:
        dataset = Planetoid(root=os.path.join(temp_dir, name), name=name)
        data = dataset[0]
    
    # Reddit dataset
    elif name == "Reddit":
        dataset = Reddit(root=os.path.join(temp_dir, "Reddit"))
        data = dataset[0]
    
    # TU datasets (molecular/protein)
    elif name in ["MUTAG", "PROTEINS"]:
        try:
            dataset = TUDataset(root=os.path.join(temp_dir, name), name=name)
            if len(dataset) == 0:
                raise ValueError(f"Dataset {name} is empty")
            
            # Use multiple graphs for proper training
            from torch_geometric.loader import DataLoader
            train_size = min(32, len(dataset))
            train_dataset = dataset[:train_size]
            
            # Create a batch of graphs
            train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=False)
            data = next(iter(train_loader))
            
        except (ImportError, RuntimeError, ValueError) as e:
            data = _create_synthetic_molecular_data(name)
    
    # ZINC dataset
    elif name == "ZINC":
        try:
            dataset = TUDataset(root=os.path.join(temp_dir, "ZINC"), name="ZINC")
            if len(dataset) == 0:
                raise ValueError("ZINC dataset is empty")
            data = dataset[0]
            if not hasattr(data, 'y') or data.y is None:
                data.y = torch.zeros(data.num_nodes, dtype=torch.long)

        except (ImportError, RuntimeError, ValueError) as e:
            data = _create_synthetic_molecular_data("ZINC")
    
    # OGB ArXiv dataset
    elif name == "OGBN-Arxiv":
        if PygNodePropPredDataset is None:
            # Fallback: create synthetic data with similar properties
            data = _create_synthetic_arxiv_data()
        else:
            try:
                dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(temp_dir, "ogbn-arxiv"))
                data = dataset[0]

            except (ImportError, RuntimeError, ValueError) as e:
                data = _create_synthetic_arxiv_data()
    
    else:
        raise ValueError(f"Dataset {name} not supported")
    
    # Ensure masks exist for node classification
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        data.train_mask, data.val_mask, data.test_mask = create_stratified_masks(data.y, data.num_nodes)
    
    return data

def _create_synthetic_molecular_data(name):
    """Create synthetic molecular-like data"""
    if name == "MUTAG":
        num_nodes = 17  # Average MUTAG molecule size
        num_features = 7   # MUTAG has 7 node features
        num_classes = 2    # Binary classification
    else:  # PROTEINS or ZINC
        num_nodes = 23
        num_features = 28
        num_classes = 2
    
    # Random molecular features
    x = torch.randn(num_nodes, num_features)
    
    # Create molecular-like graph structure (more connected)
    edge_prob = 0.3
    edge_list = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if torch.rand(1) < edge_prob:
                edge_list.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edge_list).t() if edge_list else torch.zeros(2, 0, dtype=torch.long)
    
    # Node classification labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    return Data(x=x, edge_index=edge_index, y=y)

def create_stratified_masks(labels, num_nodes, train_ratio=0.6, val_ratio=0.2):
    """Create stratified train/val/test masks ensuring balanced class distribution"""
    try:
        # Convert to numpy for sklearn
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        indices = torch.arange(num_nodes)
        
        # First split: train vs (val+test)
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, stratify=labels_np, random_state=42
        )
        
        # Second split: val vs test from remaining
        val_size = val_ratio / (1 - train_ratio)  # Adjust for remaining data
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, stratify=labels_np[temp_idx], random_state=42
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        return train_mask, val_mask, test_mask
        
    except (ValueError, ImportError):
        # Fallback to random split if stratification fails
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        indices = torch.randperm(num_nodes)
        train_end = int(train_ratio * num_nodes)
        val_end = int((train_ratio + val_ratio) * num_nodes)
        
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
        
        return train_mask, val_mask, test_mask

def _create_synthetic_arxiv_data():
    """Create realistic synthetic ArXiv-like data without data leaks"""
    num_nodes = 1000
    num_features = 128
    num_classes = 40
    
    # Create realistic random features (no class information encoded)
    x = torch.randn(num_nodes, num_features)
    
    # Generate labels with realistic class distribution
    # Some classes more common than others (like real ArXiv)
    class_probs = torch.softmax(torch.randn(num_classes), dim=0)
    y = torch.multinomial(class_probs, num_nodes, replacement=True)
    
    # Add subtle, realistic class-dependent patterns (much weaker signal)
    # This simulates how real papers in similar fields might have similar features
    class_centroids = torch.randn(num_classes, num_features) * 0.3
    for i in range(num_nodes):
        class_id = y[i].item()
        # Add weak class signal with noise - no direct encoding!
        noise_scale = 0.8 + torch.rand(1) * 0.4  # Random noise scale
        x[i] += class_centroids[class_id] * 0.2 + torch.randn(num_features) * noise_scale
    
    # Create realistic graph structure with moderate homophily
    edge_list = []
    for i in range(num_nodes):
        # Moderate homophily: 60% same class, 40% random
        num_edges = torch.randint(2, 8, (1,)).item()  # Variable degree
        
        for _ in range(num_edges):
            if torch.rand(1) < 0.6:  # 60% homophily
                # Connect to same class with some probability
                same_class_candidates = (y == y[i]).nonzero(as_tuple=False).squeeze()
                if len(same_class_candidates.shape) > 0 and len(same_class_candidates) > 1:
                    target = same_class_candidates[torch.randint(0, len(same_class_candidates), (1,))].item()
                    if target != i:
                        edge_list.extend([[i, target], [target, i]])
            else:  # 40% random connections
                target = torch.randint(0, num_nodes, (1,)).item()
                if target != i:
                    edge_list.extend([[i, target], [target, i]])
    
    # Remove duplicate edges
    if edge_list:
        edge_index = torch.tensor(edge_list).t()
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

def detect_task_type(data):
    """Detect the task type based on dataset characteristics"""
    # For now, we assume node classification since we have node labels
    # This could be extended to detect link prediction, graph classification, etc.
    if hasattr(data, 'y') and data.y is not None:
        if len(data.y.shape) == 1 and data.y.shape[0] == data.num_nodes:
            return "node_classification"
        elif len(data.y.shape) == 0 or (len(data.y.shape) == 1 and data.y.shape[0] == 1):
            return "graph_classification"
    return "node_classification"  # Default assumption

def load_csv_dataset(nodes_path, edges_path, edges_optional=False):
    nodes_df = pd.read_csv(nodes_path)
    # Handle optional edges file
    if edges_path and os.path.exists(edges_path):
        edges_df = pd.read_csv(edges_path)
        if "source" not in edges_df.columns or "target" not in edges_df.columns:
            raise ValueError("Edges CSV must contain 'source' and 'target' columns")
    elif not edges_optional:
        raise ValueError("Edges CSV file is required")
    else:
        edges_df = None
    
    # Validate required columns
    if "id" not in nodes_df.columns or "label" not in nodes_df.columns:
        raise ValueError("Nodes CSV must contain 'id' and 'label' columns")
    
    # Features: all columns except id and label
    feature_cols = [c for c in nodes_df.columns if c not in ("id", "label")]
    if len(feature_cols) == 0:
        raise ValueError("Nodes CSV must contain at least one feature column besides 'id' and 'label'")

    x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(nodes_df["label"].values, dtype=torch.long)
    
    # Handle edges
    if edges_df is not None:
        edge_index = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)
        # Add reverse edges for undirectedness if not present
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        # Create empty edge index for node-only datasets
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    
    # Create stratified train/val/test masks
    train_mask, val_mask, test_mask = create_stratified_masks(y, len(nodes_df))
    
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

def dataset_stats(data):
    task_type = detect_task_type(data)
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges) if hasattr(data, "num_edges") else int(data.edge_index.shape[1] // 2),
        "num_features": int(data.num_node_features),
        "num_classes": int(len(torch.unique(data.y).tolist())),
        "task_type": task_type,
    }
