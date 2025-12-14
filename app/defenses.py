import torch
import torch.nn.functional as F

import numpy as np

class BaseDefense:
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def apply(self, data, **kwargs):
        raise NotImplementedError
    
    def calculate_defense_effectiveness(self, original_data, defended_data, model):
        """Calculate defense effectiveness metrics"""
        model.eval()
        with torch.no_grad():
            orig_out = model(original_data.x, original_data.edge_index)
            def_out = model(defended_data.x, defended_data.edge_index)
            
            # Calculate prediction consistency
            orig_pred = orig_out.argmax(dim=1)
            def_pred = def_out.argmax(dim=1)
            
            consistency = (orig_pred == def_pred).float().mean().item()
            
            # Calculate confidence preservation
            orig_conf = torch.softmax(orig_out, dim=1).max(dim=1)[0]
            def_conf = torch.softmax(def_out, dim=1).max(dim=1)[0]
            
            conf_preservation = (def_conf / (orig_conf + 1e-8)).mean().item()
            
            return {
                "consistency": consistency,
                "confidence_preservation": conf_preservation
            }

class JaccardDefense(BaseDefense):
    def apply(self, data, threshold=0.01, **kwargs):
        """Jaccard similarity-based edge filtering"""
        x, edge_index = data.x, data.edge_index
        row, col = edge_index
        mask = []
        
        for i, j in zip(row.tolist(), col.tolist()):
            if i == j:  # Keep self-loops
                mask.append(True)
                continue
                
            # Calculate Jaccard similarity
            intersect = (x[i] > 0) & (x[j] > 0)
            union = (x[i] > 0) | (x[j] > 0)
            union_sum = union.sum().item()
            
            if union_sum > 0:
                score = intersect.sum().item() / union_sum
            else:
                score = 0.0
            
            mask.append(score > threshold)
        
        mask = torch.tensor(mask, device=edge_index.device)
        data.edge_index = edge_index[:, mask]
        return data

class FeatureDenoising(BaseDefense):
    def apply(self, data, noise_threshold=0.1, smoothing_factor=0.8, **kwargs):
        """Feature denoising using neighbor averaging"""
        x = data.x.clone()
        edge_index = data.edge_index
        
        # Calculate node degrees
        row, col = edge_index
        deg = torch.zeros(data.num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        # Smooth features using neighbor averaging
        smoothed_x = torch.zeros_like(x)
        
        for node in range(data.num_nodes):
            # Get neighbors
            neighbors = col[row == node]
            
            if len(neighbors) > 0:
                # Average neighbor features
                neighbor_features = x[neighbors].mean(dim=0)
                # Smooth with original features
                smoothed_x[node] = smoothing_factor * x[node] + (1 - smoothing_factor) * neighbor_features
            else:
                smoothed_x[node] = x[node]
        
        data.x = smoothed_x
        return data

class AdversarialTraining(BaseDefense):
    def apply(self, data, model=None, attack_config=None, **kwargs):
        """Adversarial training defense - modifies training process"""
        # This defense is applied during training, not to data preprocessing
        # Return original data, but set flag for training process
        data.adversarial_training = True
        data.attack_config = attack_config
        return data

class GradientRegularization(BaseDefense):
    def apply(self, data, **kwargs):
        """Gradient Regularization - applied during training"""
        # Mark data for gradient regularization during training
        data.use_grad_reg = True
        data.grad_reg_lambda = kwargs.get('lambda', 0.01)
        return data

class GNNGuard(BaseDefense):
    def apply(self, data, attention_threshold=0.1, **kwargs):
        """GNNGuard - attention-based defense against adversarial attacks"""
        edge_index = data.edge_index
        x = data.x
        
        # Calculate attention weights based on feature similarity
        row, col = edge_index
        attention_weights = []
        
        for i, j in zip(row.tolist(), col.tolist()):
            # Cosine similarity as attention weight
            sim = F.cosine_similarity(x[i].unsqueeze(0), x[j].unsqueeze(0))
            attention_weights.append(sim.item())
        
        attention_weights = torch.tensor(attention_weights, device=edge_index.device)
        
        # Filter edges based on attention threshold
        mask = attention_weights > attention_threshold
        data.edge_index = edge_index[:, mask]
        
        # Store attention weights for training
        data.attention_weights = attention_weights[mask]
        
        return data

class DifferentialPrivacy(BaseDefense):
    def apply(self, data, epsilon=1.0, delta=1e-5, **kwargs):
        """Differential Privacy protection via noise addition"""
        # Add calibrated noise for differential privacy
        sensitivity = 1.0  # Assume L2 sensitivity of 1
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        # Add Gaussian noise to features
        noise = torch.randn_like(data.x) * sigma
        data.x = data.x + noise
        
        # Clip values to maintain reasonable range
        data.x = torch.clamp(data.x, 0, 1)
        
        return data

AVAILABLE_DEFENSES = {
    "jaccard": JaccardDefense,
    "feature_denoising": FeatureDenoising,
    "adversarial_training": AdversarialTraining,
    "Gradient Regularization": GradientRegularization,
    "GNNGuard": GNNGuard,
    "differential_privacy": DifferentialPrivacy,
}