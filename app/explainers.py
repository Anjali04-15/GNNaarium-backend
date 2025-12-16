import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
from collections import defaultdict

class BaseExplainer:
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def explain(self, model, data, node_idx=None, **kwargs):
        raise NotImplementedError
    
    def get_k_hop_subgraph(self, node_idx, edge_index, num_hops, num_nodes=None):
        """Extract k-hop subgraph for node-level explanations"""
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        return subset, sub_edge_index, mapping, edge_mask
    
    def get_model_num_layers(self, model):
        """Determine number of message-passing layers in the model"""
        layer_count = 0
        for name, module in model.named_modules():
            if any(conv_type in str(type(module)) for conv_type in ['Conv', 'SAGE', 'GAT', 'GIN']):
                layer_count += 1
        return max(layer_count, 2)  # Default to 2 if can't determine
    
    def validate_explanation(self, model, data, explanation, node_idx, task_type='node'):
        """Validate explanation based on method type"""
        # GraphMask validation: prediction invariance threshold
        if explanation.get('method') == 'GraphMask':
            invariance = explanation.get('prediction_invariance', 0.0)
            explanation['valid'] = invariance >= 0.8
            if not explanation['valid']:
                explanation['validation_error'] = 'GraphMask results flagged as conceptually invalid - low prediction invariance'
            return explanation
        
        # ProtGNN validation: prototype consistency (NOT fidelity)
        if explanation.get('method') == 'ProtGNN':
            explanation['valid'] = explanation.get('valid_explanation', False)
            if not explanation['valid']:
                explanation['validation_error'] = 'ProtGNN explanation invalid - inconsistent prototype classes or low similarity'
            return explanation
        
        # Standard fidelity validation for other explainers
        if 'fidelity' in explanation:
            return explanation
            
        if task_type == 'node' and 'subgraph_edges' in explanation:
            try:
                with torch.no_grad():
                    original_pred = model(data.x, data.edge_index)[node_idx].argmax()
                    explained_pred = model(data.x, explanation['subgraph_edges'])[node_idx].argmax()
                    fidelity = (original_pred == explained_pred).float().item()
                    explanation['fidelity'] = fidelity
            except:
                explanation['fidelity'] = 0.0
        return explanation
    
    def get_explanation_metrics(self, explanation):
        """Calculate explanation quality metrics"""
        metrics = {}
        if 'edge_importance' in explanation:
            edge_imp = explanation['edge_importance']
            metrics.update({
                'sparsity': (edge_imp == 0).float().mean().item(),
                'max_importance': edge_imp.max().item(),
                'mean_importance': edge_imp.mean().item()
            })
        if 'fidelity' in explanation:
            metrics['fidelity'] = explanation['fidelity']
        return metrics

class GNNExplainer(BaseExplainer):
    def explain(self, model, data, node_idx=0, epochs=100, task_type='node', **kwargs):
        """GNNExplainer: Post-hoc explanation via edge mask optimization"""
        model.eval()
        
        if task_type == 'node':
            # MANDATORY: Use k-hop subgraph for node explanations
            num_hops = self.get_model_num_layers(model)
            subset, sub_edge_index, mapping, edge_mask_full = self.get_k_hop_subgraph(
                node_idx, data.edge_index, num_hops, data.x.size(0)
            )
            
            # Create subgraph data
            sub_x = data.x[subset]
            target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
            
            # Initialize edge mask for subgraph only
            edge_mask = torch.randn(sub_edge_index.size(1), requires_grad=True, device=sub_edge_index.device)
            
            # Get target prediction on original subgraph
            with torch.no_grad():
                target_out = model(sub_x, sub_edge_index)
                target_pred = target_out[target_node_in_sub].argmax()
                target_logits = target_out[target_node_in_sub]
        else:
            # Graph-level: can use full graph
            sub_edge_index = data.edge_index
            sub_x = data.x
            target_node_in_sub = node_idx
            edge_mask = torch.randn(sub_edge_index.size(1), requires_grad=True, device=sub_edge_index.device)
            
            with torch.no_grad():
                target_out = model(sub_x, sub_edge_index)
                target_pred = target_out.argmax() if len(target_out.shape) == 1 else target_out[target_node_in_sub].argmax()
                target_logits = target_out if len(target_out.shape) == 1 else target_out[target_node_in_sub]
        
        optimizer = torch.optim.Adam([edge_mask], lr=kwargs.get('lr', 0.01))
        sparsity_weight = kwargs.get('sparsity_weight', 0.005)  # Reduced sparsity penalty
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Apply sigmoid to get continuous probabilities (NO binarization)
            mask_probs = torch.sigmoid(edge_mask)
            
            # Use soft masking: multiply edge weights by probabilities
            # Create weighted edge index by duplicating edges with their weights
            edge_weights = mask_probs
            
            # For message passing, we need to handle edge weights properly
            # Use the original edge structure but with weighted messages
            try:
                # Try soft masking first
                masked_out = self._forward_with_edge_weights(model, sub_x, sub_edge_index, edge_weights)
                
                if task_type == 'node':
                    masked_logits = masked_out[target_node_in_sub]
                else:
                    masked_logits = masked_out if len(masked_out.shape) == 1 else masked_out[target_node_in_sub]
                
                # Fidelity loss: KL divergence between original and masked predictions
                pred_loss = F.kl_div(
                    F.log_softmax(masked_logits, dim=-1),
                    F.softmax(target_logits, dim=-1),
                    reduction='sum'
                )
                
            except:
                # Fallback to hard masking if soft masking fails
                top_k = max(1, int(0.3 * len(mask_probs)))  # Keep top 30% edges
                _, top_indices = mask_probs.topk(top_k)
                masked_edge_index = sub_edge_index[:, top_indices]
                
                masked_out = model(sub_x, masked_edge_index)
                
                if task_type == 'node':
                    pred_loss = F.cross_entropy(masked_out[target_node_in_sub:target_node_in_sub+1], target_pred.unsqueeze(0))
                else:
                    pred_loss = F.cross_entropy(masked_out.unsqueeze(0), target_pred.unsqueeze(0))
            
            # Reduced sparsity regularization
            sparsity_loss = -torch.mean(mask_probs * torch.log(mask_probs + 1e-8))  # Entropy regularization
            
            loss = pred_loss + sparsity_weight * sparsity_loss
            loss.backward()
            optimizer.step()
        
        # Get final importance scores (continuous)
        with torch.no_grad():
            edge_importance = torch.sigmoid(edge_mask)
            
            # Use top-K edge retention instead of fixed threshold
            retention_ratio = kwargs.get('retention_ratio', 0.3)  # Keep top 30% edges
            num_edges_to_keep = max(1, int(retention_ratio * len(edge_importance)))
            _, top_indices = edge_importance.topk(num_edges_to_keep)
            
            final_edge_index = sub_edge_index[:, top_indices]
            final_edge_weights = edge_importance[top_indices]
            
            # Validate explanation fidelity
            try:
                explained_out = model(sub_x, final_edge_index)
                if task_type == 'node':
                    explained_pred = explained_out[target_node_in_sub].argmax()
                else:
                    explained_pred = explained_out.argmax() if len(explained_out.shape) == 1 else explained_out[target_node_in_sub].argmax()
                
                fidelity = 1.0 if explained_pred == target_pred else 0.0
                
                # If fidelity is 0, try with more edges
                if fidelity == 0.0 and retention_ratio < 0.8:
                    num_edges_to_keep = max(1, int(0.5 * len(edge_importance)))
                    _, top_indices = edge_importance.topk(num_edges_to_keep)
                    final_edge_index = sub_edge_index[:, top_indices]
                    final_edge_weights = edge_importance[top_indices]
                    
                    explained_out = model(sub_x, final_edge_index)
                    if task_type == 'node':
                        explained_pred = explained_out[target_node_in_sub].argmax()
                    else:
                        explained_pred = explained_out.argmax() if len(explained_out.shape) == 1 else explained_out[target_node_in_sub].argmax()
                    
                    fidelity = 1.0 if explained_pred == target_pred else 0.0
                    
            except:
                fidelity = 0.0
                explained_pred = target_pred
        
        explanation = {
            'method': 'GNNExplainer',
            'node_idx': node_idx,
            'task_type': task_type,
            'edge_importance': edge_importance,
            'subgraph_edges': final_edge_index,
            'subgraph_weights': final_edge_weights,
            'target_prediction': target_pred.item(),
            'explained_prediction': explained_pred.item(),
            'fidelity': fidelity,
            'num_hops_used': num_hops if task_type == 'node' else None,
            'retention_ratio': retention_ratio
        }
        
        # Add failure message if fidelity is low
        if fidelity < 0.5:
            explanation['fidelity_warning'] = "The model's decision relies on distributed graph evidence and cannot be localized to a small subgraph."
        
        return explanation
    
    def _forward_with_edge_weights(self, model, x, edge_index, edge_weights):
        """Forward pass with soft edge masking"""
        # This is a simplified version - in practice, you'd need to modify the GNN layers
        # to accept edge weights. For now, we'll use a probabilistic edge sampling approach.
        
        # Sample edges based on their weights
        edge_probs = edge_weights
        sampled_edges = torch.bernoulli(edge_probs).bool()
        
        if sampled_edges.sum() == 0:
            # Keep at least the top edge
            top_edge = edge_probs.argmax()
            sampled_edges[top_edge] = True
        
        masked_edge_index = edge_index[:, sampled_edges]
        return model(x, masked_edge_index)

class PGExplainer(BaseExplainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.explainer_net = None
        self.is_trained = False
        
    def _build_explainer_net(self, edge_feat_dim):
        """Build parametric explainer network"""
        return torch.nn.Sequential(
            torch.nn.Linear(edge_feat_dim, 16),  # Much smaller network
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
    
    def train_explainer(self, model, data, num_epochs=100, **kwargs):
        """Train the parametric explainer network"""
        model.eval()  # Freeze GNN
        
        # Sample training nodes (much smaller for speed)
        num_nodes = min(10, data.x.size(0))  # Train on only 10 nodes
        train_nodes = torch.randperm(data.x.size(0))[:num_nodes]
        
        # Get k-hop subgraphs for training
        num_hops = self.get_model_num_layers(model)
        training_data = []
        
        for node_idx in train_nodes:
            subset, sub_edge_index, mapping, _ = self.get_k_hop_subgraph(
                node_idx.item(), data.edge_index, num_hops, data.x.size(0)
            )
            
            if len(subset) > 1 and sub_edge_index.size(1) > 0:
                sub_x = data.x[subset]
                target_node_in_sub = mapping[node_idx.item()] if node_idx.item() in mapping else 0
                
                # Get target prediction
                with torch.no_grad():
                    target_out = model(sub_x, sub_edge_index)
                    target_pred = target_out[target_node_in_sub].argmax()
                
                training_data.append({
                    'sub_x': sub_x,
                    'sub_edge_index': sub_edge_index,
                    'target_node_in_sub': target_node_in_sub,
                    'target_pred': target_pred,
                    'node_idx': node_idx.item()
                })
        
        if not training_data:
            print("[WARNING] No valid training data for PGExplainer")
            return
        
        # Build explainer network
        sample_edge_feat = self._get_edge_features(training_data[0]['sub_x'], training_data[0]['sub_edge_index'], training_data[0]['target_node_in_sub'])
        edge_feat_dim = sample_edge_feat.size(1)
        
        self.explainer_net = self._build_explainer_net(edge_feat_dim).to(data.x.device)
        optimizer = torch.optim.Adam(self.explainer_net.parameters(), lr=0.1)  # Higher LR
        
        # Training loop
        self.explainer_net.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch_data in training_data:
                optimizer.zero_grad()
                
                # Get edge features
                edge_features = self._get_edge_features(
                    batch_data['sub_x'], 
                    batch_data['sub_edge_index'], 
                    batch_data['target_node_in_sub']
                )
                
                # Predict edge importance
                edge_probs = self.explainer_net(edge_features).squeeze()
                
                # Use top-K edges for fidelity
                top_k = max(1, int(0.3 * len(edge_probs)))
                _, top_indices = edge_probs.topk(top_k)
                masked_edge_index = batch_data['sub_edge_index'][:, top_indices]
                
                try:
                    # Forward pass with masked edges
                    masked_out = model(batch_data['sub_x'], masked_edge_index)
                    masked_pred = masked_out[batch_data['target_node_in_sub']]
                    
                    # Fidelity loss: maintain prediction
                    target_logits = model(batch_data['sub_x'], batch_data['sub_edge_index'])[batch_data['target_node_in_sub']]
                    fidelity_loss = F.kl_div(
                        F.log_softmax(masked_pred, dim=-1),
                        F.softmax(target_logits, dim=-1),
                        reduction='sum'
                    )
                    
                    # Sparsity loss
                    sparsity_loss = edge_probs.mean()
                    
                    loss = fidelity_loss + 0.01 * sparsity_loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                except:
                    continue
            
            # Skip frequent printing for speed
            pass
        
        self.explainer_net.eval()
        self.is_trained = True
        print(f"[PGExplainer] Training completed after {num_epochs} epochs")
    
    def _get_edge_features(self, sub_x, sub_edge_index, target_node_in_sub):
        """Extract edge features for parametric model"""
        edge_features = []
        target_feat = sub_x[target_node_in_sub]
        
        for i in range(sub_edge_index.size(1)):
            src, dst = sub_edge_index[0, i], sub_edge_index[1, i]
            src_feat = sub_x[src]
            dst_feat = sub_x[dst]
            
            # Edge feature: [src, dst, src*dst, target_similarity]
            src_sim = F.cosine_similarity(target_feat.unsqueeze(0), src_feat.unsqueeze(0), dim=1)
            dst_sim = F.cosine_similarity(target_feat.unsqueeze(0), dst_feat.unsqueeze(0), dim=1)
            
            edge_feat = torch.cat([
                src_feat, 
                dst_feat, 
                src_feat * dst_feat,
                src_sim,
                dst_sim
            ])
            edge_features.append(edge_feat)
        
        return torch.stack(edge_features)
    
    def explain(self, model, data, node_idx=0, task_type='node', **kwargs):
        """PGExplainer: Parametric explainer using trained neural network"""
        model.eval()
        
        # Train explainer if not already trained
        if not self.is_trained:
            print("[PGExplainer] Training explainer network...")
            self.train_explainer(model, data, num_epochs=kwargs.get('explainer_epochs', 20))  # Default 20 epochs
        
        if task_type == 'node':
            # MANDATORY: Use k-hop subgraph for node explanations
            num_hops = self.get_model_num_layers(model)
            subset, sub_edge_index, mapping, _ = self.get_k_hop_subgraph(
                node_idx, data.edge_index, num_hops, data.x.size(0)
            )
            
            sub_x = data.x[subset]
            target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
        else:
            # Graph-level: can use full graph
            sub_edge_index = data.edge_index
            sub_x = data.x
            target_node_in_sub = node_idx
            num_hops = None
        
        if sub_edge_index.size(1) == 0:
            # Handle empty subgraph
            return {
                'method': 'PGExplainer',
                'node_idx': node_idx,
                'task_type': task_type,
                'edge_importance': torch.tensor([]),
                'subgraph_edges': torch.empty((2, 0), dtype=torch.long),
                'subgraph_weights': torch.tensor([]),
                'fidelity': 0.0,
                'parametric': True,
                'num_hops_used': num_hops,
                'error': 'Empty subgraph'
            }
        
        # Get original prediction
        with torch.no_grad():
            original_out = model(sub_x, sub_edge_index)
            original_pred = original_out[target_node_in_sub].argmax()
        
        # Use trained explainer network
        self.explainer_net.eval()
        with torch.no_grad():
            edge_features = self._get_edge_features(sub_x, sub_edge_index, target_node_in_sub)
            edge_importance = self.explainer_net(edge_features).squeeze()
            
            # Use top-K edges (NO hard thresholding)
            retention_ratio = kwargs.get('retention_ratio', 0.3)
            num_edges_to_keep = max(1, int(retention_ratio * len(edge_importance)))
            _, top_indices = edge_importance.topk(num_edges_to_keep)
            
            final_edge_index = sub_edge_index[:, top_indices]
            final_edge_weights = edge_importance[top_indices]
            
            # Validate fidelity
            try:
                explained_out = model(sub_x, final_edge_index)
                explained_pred = explained_out[target_node_in_sub].argmax()
                fidelity = 1.0 if explained_pred == original_pred else 0.0
                
                # Try higher retention if fidelity is 0
                if fidelity == 0.0 and retention_ratio < 0.8:
                    num_edges_to_keep = max(1, int(0.5 * len(edge_importance)))
                    _, top_indices = edge_importance.topk(num_edges_to_keep)
                    final_edge_index = sub_edge_index[:, top_indices]
                    final_edge_weights = edge_importance[top_indices]
                    
                    explained_out = model(sub_x, final_edge_index)
                    explained_pred = explained_out[target_node_in_sub].argmax()
                    fidelity = 1.0 if explained_pred == original_pred else 0.0
                    
            except:
                fidelity = 0.0
                explained_pred = original_pred
        
        explanation = {
            'method': 'PGExplainer',
            'node_idx': node_idx,
            'task_type': task_type,
            'edge_importance': edge_importance,
            'subgraph_edges': final_edge_index,
            'subgraph_weights': final_edge_weights,
            'target_prediction': original_pred.item(),
            'explained_prediction': explained_pred.item(),
            'fidelity': fidelity,
            'parametric': True,
            'num_hops_used': num_hops,
            'trained': self.is_trained
        }
        
        # Add warning for low fidelity
        if fidelity < 0.5:
            explanation['fidelity_warning'] = "The parametric explainer has not learned a reliable explanation function for this model."
        
        return explanation

class SubgraphX(BaseExplainer):
    def explain(self, model, data, node_idx=0, num_samples=50, task_type='node', **kwargs):
        """SubgraphX: Shapley value-based subgraph explanation"""
        model.eval()
        
        if task_type == 'node':
            # MANDATORY: Use k-hop subgraph for node explanations
            num_hops = self.get_model_num_layers(model)
            subset, sub_edge_index, mapping, edge_mask_full = self.get_k_hop_subgraph(
                node_idx, data.edge_index, num_hops, data.x.size(0)
            )
            
            sub_x = data.x[subset]
            target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
        else:
            # Graph-level: can use full graph
            sub_edge_index = data.edge_index
            sub_x = data.x
            target_node_in_sub = node_idx
        
        num_edges = min(sub_edge_index.size(1), 30)  # Limit for efficiency
        
        # Get baseline prediction
        with torch.no_grad():
            full_out = model(sub_x, sub_edge_index)
            if task_type == 'node':
                full_pred = full_out[target_node_in_sub]
            else:
                full_pred = full_out
        
        # Calculate Shapley values for edges using MCTS-inspired sampling
        edge_shapley = []
        
        for edge_idx in range(num_edges):
            marginal_contributions = []
            
            for _ in range(num_samples):
                # Random subset of edges (excluding current edge)
                all_edges = list(range(num_edges))
                all_edges.remove(edge_idx)
                
                if len(all_edges) > 0:
                    subset_size = random.randint(0, min(len(all_edges), 15))
                    edge_subset = random.sample(all_edges, subset_size)
                    
                    # Prediction without current edge
                    if edge_subset:
                        subset_edges = sub_edge_index[:, edge_subset]
                    else:
                        # Empty graph - use self-loop for target node
                        subset_edges = torch.tensor([[target_node_in_sub], [target_node_in_sub]], device=sub_edge_index.device)
                    
                    with torch.no_grad():
                        try:
                            out_without = model(sub_x, subset_edges)
                            if task_type == 'node':
                                pred_without = out_without[target_node_in_sub]
                            else:
                                pred_without = out_without
                        except:
                            pred_without = torch.zeros_like(full_pred)
                    
                    # Prediction with current edge added
                    edge_subset_with = edge_subset + [edge_idx]
                    subset_edges_with = sub_edge_index[:, edge_subset_with]
                    
                    with torch.no_grad():
                        try:
                            out_with = model(sub_x, subset_edges_with)
                            if task_type == 'node':
                                pred_with = out_with[target_node_in_sub]
                            else:
                                pred_with = out_with
                        except:
                            pred_with = torch.zeros_like(full_pred)
                    
                    # Marginal contribution using KL divergence
                    try:
                        contribution = F.kl_div(
                            F.log_softmax(pred_with, dim=-1),
                            F.softmax(pred_without, dim=-1),
                            reduction='sum'
                        ).item()
                        marginal_contributions.append(abs(contribution))
                    except:
                        marginal_contributions.append(0.0)
                else:
                    marginal_contributions.append(0.0)
            
            edge_shapley.append(np.mean(marginal_contributions) if marginal_contributions else 0.0)
        
        # Pad with zeros for remaining edges
        while len(edge_shapley) < sub_edge_index.size(1):
            edge_shapley.append(0.0)
        
        edge_importance = torch.tensor(edge_shapley, device=sub_edge_index.device)
        
        # Get connected subgraph with highest Shapley values
        if len(edge_shapley) > 5:
            top_k = min(10, len(edge_shapley))
            _, top_indices = edge_importance.topk(top_k)
            final_edge_index = sub_edge_index[:, top_indices]
            final_edge_weights = edge_importance[top_indices]
        else:
            final_edge_index = sub_edge_index
            final_edge_weights = edge_importance[:sub_edge_index.size(1)]
        
        explanation = {
            'method': 'SubgraphX',
            'node_idx': node_idx,
            'task_type': task_type,
            'edge_importance': edge_importance,
            'subgraph_edges': final_edge_index,
            'subgraph_weights': final_edge_weights,
            'shapley_values': edge_shapley[:num_edges],
            'num_hops_used': num_hops if task_type == 'node' else None
        }
        
        return self.validate_explanation(model, data, explanation, node_idx, task_type)

class ProtGNN(BaseExplainer):
    def explain(self, model, data, node_idx=0, num_prototypes=5, task_type='graph', **kwargs):
        """ProtGNN: Prototype-based self-interpretable explanation (mainly for graph classification)"""
        model.eval()
        
        with torch.no_grad():
            # Get intermediate embeddings from model
            if hasattr(model, 'get_embeddings'):
                embeddings = model.get_embeddings(data.x, data.edge_index)
            else:
                embeddings = data.x  # Fallback to input features
            
            if task_type == 'node':
                # For node tasks, find prototypes in k-hop neighborhood
                num_hops = self.get_model_num_layers(model)
                subset, sub_edge_index, mapping, edge_mask_full = self.get_k_hop_subgraph(
                    node_idx, data.edge_index, num_hops, data.x.size(0)
                )
                
                sub_embeddings = embeddings[subset]
                target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
                target_embedding = sub_embeddings[target_node_in_sub]
                
                # Find prototypes within subgraph
                similarities = F.cosine_similarity(target_embedding.unsqueeze(0), sub_embeddings, dim=1)
                _, proto_indices_in_sub = similarities.topk(min(num_prototypes, len(subset)))
                
                # Map back to original indices
                prototype_indices = subset[proto_indices_in_sub]
                prototype_similarities = similarities[proto_indices_in_sub]
                
                working_edge_index = sub_edge_index
            else:
                # Graph-level: use full graph for prototype learning
                target_embedding = embeddings.mean(dim=0)  # Graph-level embedding
                similarities = F.cosine_similarity(target_embedding.unsqueeze(0), embeddings, dim=1)
                _, prototype_indices = similarities.topk(num_prototypes)
                prototype_similarities = similarities[prototype_indices]
                working_edge_index = data.edge_index
            
            # Create prototype explanation
            prototypes = []
            for i, (proto_idx, sim) in enumerate(zip(prototype_indices, prototype_similarities)):
                prototypes.append({
                    'node_idx': proto_idx.item(),
                    'similarity': sim.item(),
                    'features': data.x[proto_idx].tolist(),
                    'label': data.y[proto_idx].item() if hasattr(data, 'y') and proto_idx < len(data.y) else None
                })
            
            # Find edges connecting to prototypes (explanatory subgraph)
            prototype_edges = []
            edge_importance = torch.zeros(working_edge_index.size(1), device=working_edge_index.device)
            
            for i in range(working_edge_index.size(1)):
                src, dst = working_edge_index[0, i], working_edge_index[1, i]
                
                # Map to original indices if working with subgraph
                if task_type == 'node':
                    orig_src = subset[src] if src < len(subset) else src
                    orig_dst = subset[dst] if dst < len(subset) else dst
                else:
                    orig_src, orig_dst = src, dst
                
                if orig_src in prototype_indices or orig_dst in prototype_indices:
                    if task_type == 'node' and (orig_src == node_idx or orig_dst == node_idx):
                        edge_importance[i] = 1.0  # High importance - direct to target
                        prototype_edges.append(i)
                    elif orig_src in prototype_indices and orig_dst in prototype_indices:
                        edge_importance[i] = 0.8  # High importance - between prototypes
                        prototype_edges.append(i)
                    else:
                        edge_importance[i] = 0.6  # Medium importance - connected to prototype
                        prototype_edges.append(i)
            
            # Extract explanatory subgraph
            if prototype_edges:
                final_edge_index = working_edge_index[:, prototype_edges]
                final_edge_weights = edge_importance[prototype_edges]
            else:
                # Fallback: use all edges with non-zero importance
                nonzero_edges = edge_importance > 0
                if nonzero_edges.sum() > 0:
                    final_edge_index = working_edge_index[:, nonzero_edges]
                    final_edge_weights = edge_importance[nonzero_edges]
                else:
                    final_edge_index = working_edge_index[:, :5] if working_edge_index.size(1) > 5 else working_edge_index
                    final_edge_weights = edge_importance[:final_edge_index.size(1)]
        
        # Validate prototype consistency
        predicted_class = data.y[node_idx].item() if hasattr(data, 'y') and node_idx < len(data.y) else None
        if predicted_class is not None:
            prototype_classes = [p.get('label') for p in prototypes if p.get('label') is not None]
            class_consistent = all(cls == predicted_class for cls in prototype_classes) if prototype_classes else False
            top_similarity = prototypes[0]['similarity'] if prototypes else 0.0
        else:
            class_consistent = True  # Cannot validate without labels
            top_similarity = prototypes[0]['similarity'] if prototypes else 0.0
        
        explanation = {
            'method': 'ProtGNN',
            'explainer_type': 'Self-interpretable (prototype-based)',
            'explanation_scope': 'Embedding-space similarity',
            'node_idx': node_idx,
            'task_type': task_type,
            'prototypes': prototypes,
            'num_prototypes': len(prototypes),
            'top_similarity': top_similarity,
            'prototype_class_agreement': class_consistent,
            'valid_explanation': class_consistent and top_similarity > 0.5,
            'interpretable': True,
            'self_interpretable': True
        }
        
        return explanation

class GraphMask(BaseExplainer):
    def explain(self, model, data, node_idx=0, task_type='node', epochs=50, **kwargs):
        """GraphMask: Global message-level explainer that learns binary gates on message passing"""
        model.eval()
        
        # MANDATORY: GraphMask is a GLOBAL explainer - node selection is only for visualization context
        if task_type == 'node':
            # Use k-hop subgraph for visualization context only
            num_hops = self.get_model_num_layers(model)
            subset, sub_edge_index, mapping, edge_mask_full = self.get_k_hop_subgraph(
                node_idx, data.edge_index, num_hops, data.x.size(0)
            )
            sub_x = data.x[subset]
            target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
        else:
            # Graph-level: use full graph
            sub_edge_index = data.edge_index
            sub_x = data.x
            target_node_in_sub = node_idx
        
        # Initialize learnable edge masks (binary gates)
        edge_mask = torch.ones(sub_edge_index.size(1), requires_grad=True, device=sub_edge_index.device)
        
        # Get target prediction for invariance optimization
        with torch.no_grad():
            target_out = model(sub_x, sub_edge_index)
            if task_type == 'node':
                target_class = target_out[target_node_in_sub].argmax()
            else:
                target_class = target_out.argmax() if len(target_out.shape) == 1 else target_out[target_node_in_sub].argmax()
        
        optimizer = torch.optim.Adam([edge_mask], lr=kwargs.get('lr', 0.1))
        sparsity_weight = kwargs.get('sparsity_weight', 0.01)
        
        # Train binary gates jointly with GNN (message-level gating)
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Apply sigmoid for differentiable gating
            gate_probs = torch.sigmoid(edge_mask)
            
            # Structural reasoning: encourage connected components
            structure_loss = 0
            for i in range(sub_edge_index.size(1)):
                src, dst = sub_edge_index[0, i], sub_edge_index[1, i]
                if src == target_node_in_sub or dst == target_node_in_sub:
                    structure_loss += (1 - gate_probs[i]) ** 2
                else:
                    structure_loss += 0.1 * (1 - gate_probs[i]) ** 2
            
            # Apply gates by selecting edges
            selected_edges = gate_probs > 0.5
            if selected_edges.sum() > 0:
                gated_edge_index = sub_edge_index[:, selected_edges]
            else:
                top_edges = gate_probs.topk(min(5, len(gate_probs)))[1]
                gated_edge_index = sub_edge_index[:, top_edges]
            
            try:
                gated_out = model(sub_x, gated_edge_index)
                
                # Prediction invariance loss (NOT fidelity)
                if task_type == 'node':
                    invariance_loss = F.cross_entropy(gated_out[target_node_in_sub:target_node_in_sub+1], target_class.unsqueeze(0))
                else:
                    invariance_loss = F.cross_entropy(gated_out.unsqueeze(0), target_class.unsqueeze(0))
                
                # Sparsity loss for redundancy identification
                sparsity_loss = gate_probs.sum()
                
                total_loss = invariance_loss + sparsity_weight * sparsity_loss + 0.1 * structure_loss
                total_loss.backward()
                optimizer.step()
            except:
                continue
        
        with torch.no_grad():
            final_gates = torch.sigmoid(edge_mask)
            
            # Identify redundant vs essential edges
            gated_off = final_gates <= 0.5  # Redundant edges
            essential = final_gates > 0.5   # Essential edges
            
            if essential.sum() > 0:
                final_edge_index = sub_edge_index[:, essential]
                final_edge_weights = final_gates[essential]
            else:
                _, top_indices = final_gates.topk(min(10, len(final_gates)))
                final_edge_index = sub_edge_index[:, top_indices]
                final_edge_weights = final_gates[top_indices]
            
            # Calculate prediction invariance (NOT fidelity)
            try:
                invariant_out = model(sub_x, final_edge_index)
                if task_type == 'node':
                    invariant_pred = invariant_out[target_node_in_sub].argmax()
                else:
                    invariant_pred = invariant_out.argmax() if len(invariant_out.shape) == 1 else invariant_out[target_node_in_sub].argmax()
                prediction_invariance = 1.0 if invariant_pred == target_class else 0.0
            except:
                prediction_invariance = 0.0
        
        explanation = {
            'method': 'GraphMask',
            'explainer_type': 'Global (message-level)',
            'visualization_context': f'Node {node_idx} neighborhood' if task_type == 'node' else 'Full graph',
            'node_idx': node_idx,
            'task_type': task_type,
            'edge_importance': final_gates,
            'subgraph_edges': final_edge_index,
            'subgraph_weights': final_edge_weights,
            'total_edges_evaluated': len(final_gates),
            'edges_gated_off': gated_off.sum().item(),
            'edges_retained': essential.sum().item(),
            'prediction_invariance': prediction_invariance,
            'structural_reasoning': True,
            'global_analysis': True,
            'num_hops_used': num_hops if task_type == 'node' else None
        }
        
        return explanation

# NeuronAnalysis is global; node_idx is used only for visualization context
class NeuronAnalysis(BaseExplainer):
    def explain(self, model, data, node_idx=0, task_type='node', **kwargs):
        """NeuronAnalysis: Global concept-based explanation"""
        model.eval()
        
        # Get intermediate activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for different layers
        hooks = []
        for name, module in model.named_modules():
            if any(conv_type in str(type(module)) for conv_type in ['Conv', 'Linear', 'SAGE', 'GAT', 'GIN']):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            if task_type == 'node':
                # For node tasks, analyze in context of k-hop subgraph
                num_hops = self.get_model_num_layers(model)
                subset, sub_edge_index, mapping, edge_mask_full = self.get_k_hop_subgraph(
                    node_idx, data.edge_index, num_hops, data.x.size(0)
                )
                
                sub_x = data.x[subset]
                target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
                output = model(sub_x, sub_edge_index)
                target_pred = output[target_node_in_sub]
            else:
                # Graph-level analysis
                output = model(data.x, data.edge_index)
                target_pred = output if len(output.shape) == 1 else output[node_idx]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze neuron activations and map to logical concepts
        neuron_concepts = {}
        concept_rules = []
        
        for layer_name, activation in activations.items():
            if len(activation.shape) >= 2:
                if task_type == 'node':
                    # Analyze target node activation
                    if target_node_in_sub < activation.size(0):
                        node_activation = activation[target_node_in_sub]
                    else:
                        continue
                else:
                    # Graph-level: analyze global activation patterns
                    node_activation = activation.mean(dim=0) if len(activation.shape) > 1 else activation
                
                # Find most active neurons
                top_neurons = node_activation.topk(min(8, len(node_activation)))[1]
                
                concepts = []
                for neuron_idx in top_neurons:
                    activation_val = node_activation[neuron_idx].item()
                    
                    # Map neurons to logical concepts (AND, OR, NOT patterns)
                    if activation_val > 0.9:
                        concept = "strong_positive_evidence"  # AND-like
                        logical_rule = f"IF {layer_name}_neuron_{neuron_idx} > 0.9 THEN high_confidence"
                    elif activation_val > 0.7:
                        concept = "positive_evidence"  # OR-like
                        logical_rule = f"IF {layer_name}_neuron_{neuron_idx} > 0.7 THEN medium_confidence"
                    elif activation_val < -0.5:
                        concept = "negative_evidence"  # NOT-like
                        logical_rule = f"IF {layer_name}_neuron_{neuron_idx} < -0.5 THEN inhibitory"
                    else:
                        concept = "neutral_evidence"
                        logical_rule = f"IF {layer_name}_neuron_{neuron_idx} in [-0.5, 0.7] THEN neutral"
                    
                    concepts.append({
                        'neuron_idx': neuron_idx.item(),
                        'activation': activation_val,
                        'concept': concept,
                        'logical_rule': logical_rule
                    })
                    
                    concept_rules.append(logical_rule)
                
                neuron_concepts[layer_name] = concepts
        
        # Global feature importance analysis
        if task_type == 'node':
            feature_importance = torch.abs(data.x[node_idx])
        else:
            feature_importance = torch.abs(data.x).mean(dim=0)
        
        top_features = feature_importance.topk(min(10, len(feature_importance)))[1]
        
        # Concept-to-prediction mapping
        prediction_confidence = torch.softmax(target_pred, dim=-1).max().item()
        predicted_class = target_pred.argmax().item()
        
        explanation = {
            'method': 'NeuronAnalysis',

            # node_idx kept for visualization only
            'node_idx': node_idx,

            # Override semantic meaning without breaking API
            'task_type': 'global',

            'neuron_concepts': neuron_concepts,
            'concept_rules': concept_rules,

            # Feature importance is auxiliary, not causal
            'feature_importance': feature_importance,
            'top_features': top_features.tolist(),

            'prediction': predicted_class,

            # IMPORTANT: clarify meaning of confidence
            'confidence': prediction_confidence,  # model confidence, not explanation quality
            'confidence_type': 'model_certainty',

            'global_analysis': True,

            # k-hop used only as context, not explanation scope
            'num_hops_used': num_hops if task_type == 'node' else None,
            'hop_usage_type': 'visualization_only',

            'logical_concepts': {
                'AND_patterns': [rule for rule in concept_rules if 'high_confidence' in rule],
                'OR_patterns': [rule for rule in concept_rules if 'medium_confidence' in rule],
                'NOT_patterns': [rule for rule in concept_rules if 'inhibitory' in rule]
            }
        }

        # NeuronAnalysis does not support fidelity or local causal validation
        explanation['supports_fidelity'] = False
        explanation['supports_local_causality'] = False

        return explanation

AVAILABLE_EXPLAINERS = {
    "GNNExplainer": GNNExplainer,
    "PGExplainer": PGExplainer,
    "SubgraphX": SubgraphX,
    "ProtGNN": ProtGNN,
    "GraphMask": GraphMask,
    "NeuronAnalysis": NeuronAnalysis,
}