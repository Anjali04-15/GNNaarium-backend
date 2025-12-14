import torch
import random
import numpy as np
from torch_geometric.utils import to_undirected

class BaseAttack:
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def run(self, model, data, **kwargs):
        raise NotImplementedError
    
    def calculate_attack_success_rate(self, model, original_data, attacked_data):
        """Calculate attack success rate"""
        model.eval()
        with torch.no_grad():
            # Original predictions
            orig_out = model(original_data.x, original_data.edge_index)
            orig_pred = orig_out.argmax(dim=1)
            
            # Attacked predictions
            att_out = model(attacked_data.x, attacked_data.edge_index)
            att_pred = att_out.argmax(dim=1)
            
            # Calculate success rate (percentage of changed predictions)
            mask = original_data.test_mask if hasattr(original_data, 'test_mask') else torch.ones(original_data.num_nodes, dtype=torch.bool)
            changed = (orig_pred[mask] != att_pred[mask]).sum().item()
            total = mask.sum().item()
            
            return changed / total if total > 0 else 0.0

class FGSM(BaseAttack):
    def run(self, model, data, budget_pct=0.05, **kwargs):
        """FGSM: Fast Gradient Sign Method"""
        data = data.clone()
        model.train()
        
        x = data.x.clone().requires_grad_(True)
        out = model(x, data.edge_index)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            out[data.train_mask], data.y[data.train_mask]
        )
        loss.backward()
        
        # FGSM perturbation
        epsilon = budget_pct
        with torch.no_grad():
            perturbation = epsilon * x.grad.sign()
            x_adv = x + perturbation
            x_adv = torch.clamp(x_adv, 0, 1)
        
        data.x = x_adv
        return data

class PGD(BaseAttack):
    def run(self, model, data, budget_pct=0.05, **kwargs):
        """PGD: Projected Gradient Descent attack on node features"""
        data = data.clone()
        model.train()
        
        x = data.x.clone().requires_grad_(True)
        epsilon = budget_pct
        alpha = epsilon / 10
        num_steps = 10
        
        for _ in range(num_steps):
            if x.grad is not None:
                x.grad.zero_()
            
            out = model(x, data.edge_index)
            # Maximize loss (attack)
            loss = -torch.nn.functional.cross_entropy(
                out[data.train_mask], data.y[data.train_mask]
            )
            loss.backward()
            
            # PGD step
            with torch.no_grad():
                x += alpha * x.grad.sign()
                # Project back to epsilon ball
                perturbation = torch.clamp(x - data.x, -epsilon, epsilon)
                x = data.x + perturbation
                x = torch.clamp(x, 0, 1)  # Ensure valid range
                x.requires_grad_(True)
        
        data.x = x.detach()
        return data

class Nettack(BaseAttack):
    def run(self, model, data, budget_pct=0.05, **kwargs):
        """Nettack: Targeted adversarial attack on graph structure and features"""
        data = data.clone()
        model.eval()
        
        # Simple implementation: perturb both structure and features
        edge_budget = int(budget_pct * data.edge_index.size(1))
        feature_budget = int(budget_pct * data.x.numel())
        
        # Structure perturbation
        edge_index = data.edge_index.clone()
        for _ in range(edge_budget):
            if random.random() < 0.5 and edge_index.size(1) > 0:
                # Remove edge
                idx = random.randint(0, edge_index.size(1) - 1)
                edge_index = torch.cat([edge_index[:, :idx], edge_index[:, idx+1:]], dim=1)
            else:
                # Add edge
                i, j = random.sample(range(data.num_nodes), 2)
                new_edge = torch.tensor([[i], [j]], device=edge_index.device)
                edge_index = torch.cat([edge_index, new_edge], dim=1)
        
        # Feature perturbation
        x = data.x.clone()
        for _ in range(feature_budget):
            node_idx = random.randint(0, x.size(0) - 1)
            feat_idx = random.randint(0, x.size(1) - 1)
            x[node_idx, feat_idx] += torch.randn(1).item() * 0.1
        
        data.edge_index = to_undirected(edge_index)
        data.x = x
        return data

class Metattack(BaseAttack):
    def run(self, model, data, budget_pct=0.05, **kwargs):
        """Metattack: Meta-learning based poisoning attack"""
        data = data.clone()
        
        # Simplified meta-attack: modify graph structure during training
        edge_index = data.edge_index.clone()
        num_modifications = int(budget_pct * edge_index.size(1))
        
        for _ in range(num_modifications):
            if random.random() < 0.7:
                # Add edge between different classes (more harmful)
                nodes_class_0 = (data.y == 0).nonzero().flatten()
                nodes_class_1 = (data.y == 1).nonzero().flatten()
                
                if len(nodes_class_0) > 0 and len(nodes_class_1) > 0:
                    i = nodes_class_0[random.randint(0, len(nodes_class_0) - 1)].item()
                    j = nodes_class_1[random.randint(0, len(nodes_class_1) - 1)].item()
                    new_edge = torch.tensor([[i], [j]], device=edge_index.device)
                    edge_index = torch.cat([edge_index, new_edge], dim=1)
            else:
                # Remove edge within same class
                if edge_index.size(1) > 0:
                    idx = random.randint(0, edge_index.size(1) - 1)
                    edge_index = torch.cat([edge_index[:, :idx], edge_index[:, idx+1:]], dim=1)
        
        data.edge_index = to_undirected(edge_index)
        return data

class CLGA(BaseAttack):
    def run(self, model, data, budget_pct=0.05, **kwargs):
        """CLGA: Contrastive Learning Graph Attack"""
        data = data.clone()
        model.eval()
        
        # Modify graph structure to disrupt contrastive learning
        edge_index = data.edge_index.clone()
        num_modifications = int(budget_pct * edge_index.size(1))
        
        for _ in range(num_modifications):
            if random.random() < 0.6:
                # Add edges between dissimilar nodes (different classes)
                different_class_pairs = []
                for i in range(data.num_nodes):
                    for j in range(i+1, data.num_nodes):
                        if data.y[i] != data.y[j]:
                            different_class_pairs.append((i, j))
                
                if different_class_pairs:
                    i, j = random.choice(different_class_pairs)
                    new_edge = torch.tensor([[i], [j]], device=edge_index.device)
                    edge_index = torch.cat([edge_index, new_edge], dim=1)
            else:
                # Remove edges between similar nodes (same class)
                if edge_index.size(1) > 0:
                    for attempt in range(10):  # Try up to 10 times
                        idx = random.randint(0, edge_index.size(1) - 1)
                        src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
                        if src < data.y.size(0) and dst < data.y.size(0) and data.y[src] == data.y[dst]:
                            edge_index = torch.cat([edge_index[:, :idx], edge_index[:, idx+1:]], dim=1)
                            break
        
        data.edge_index = to_undirected(edge_index)
        return data

class ModelInversion(BaseAttack):
    def run(self, model, data, budget_pct=0.05, **kwargs):
        """Model Inversion: Reconstruct private node attributes"""
        # Simplified implementation - attempt to reconstruct features
        model.eval()
        
        # Try to invert model predictions to recover features
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1)
            
            # This would normally try to reconstruct input features
            # from model outputs - simplified for demo
            pass
        
        return data

AVAILABLE_ATTACKS = {
    "FGSM": FGSM,
    "PGD": PGD,
    "Nettack": Nettack,
    "Metattack": Metattack,
    "CLGA": CLGA,
    "Model Inversion": ModelInversion,
}