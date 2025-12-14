import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.5, graph_level=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.graph_level = graph_level
        self.dropout = dropout
        
        if graph_level:
            self.classifier = torch.nn.Linear(hidden_dim, out_channels)
        else:
            self.conv3 = GCNConv(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        if self.graph_level:
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.5, graph_level=False):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)
        self.graph_level = graph_level
        self.dropout = dropout
        
        if graph_level:
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
            self.conv2 = GINConv(nn2)
            self.classifier = torch.nn.Linear(hidden_dim, out_channels)
        else:
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, out_channels)
            )
            self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        if self.graph_level:
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.5, heads=8, graph_level=False):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        self.graph_level = graph_level
        self.dropout = dropout
        
        if graph_level:
            self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
            self.classifier = torch.nn.Linear(hidden_dim, out_channels)
        else:
            self.conv2 = GATConv(hidden_dim * heads, out_channels, heads=1, dropout=dropout)

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        if self.graph_level:
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
        return x

class GraphSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.5, graph_level=False):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.graph_level = graph_level
        self.dropout = dropout
        
        if graph_level:
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, out_channels)
        else:
            self.conv2 = SAGEConv(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        if self.graph_level:
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
        return x

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.5, heads=8, graph_level=False):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        self.graph_level = graph_level
        self.dropout = dropout
        
        if graph_level:
            self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
            self.classifier = torch.nn.Linear(hidden_dim, out_channels)
        else:
            self.conv2 = TransformerConv(hidden_dim * heads, out_channels, heads=1, dropout=dropout)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        if self.graph_level:
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
        return x

class KAGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.5, k=3, graph_level=False):
        super().__init__()
        self.k = k
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.knowledge_attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.knowledge_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.graph_level = graph_level
        self.dropout = dropout
        
        if graph_level:
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, out_channels)
        else:
            self.conv2 = GCNConv(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch=None):
        # First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Knowledge-aware attention mechanism
        x_reshaped = x.unsqueeze(1)
        attn_output, _ = self.knowledge_attention(x_reshaped, x_reshaped, x_reshaped)
        x_knowledge = self.knowledge_linear(attn_output.squeeze(1))
        x = x + x_knowledge
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        if self.graph_level:
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
        return x

# Model factory function for easy model selection
def create_model(model_name, in_channels, hidden_dim, out_channels, dropout=0.5, graph_level=False, **kwargs):
    """Factory function to create different GNN models"""
    models = {
        'GCN': GCN,
        'GIN': GIN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GraphTransformer': GraphTransformer,
        'KA-GNN': KAGNN
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(models.keys())}")
    
    # All models now support graph_level parameter
    return models[model_name](in_channels, hidden_dim, out_channels, dropout, graph_level=graph_level, **kwargs)