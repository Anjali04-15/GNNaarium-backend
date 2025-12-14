import torch
import torch.nn.functional as F
import numpy as np

import random

class BaseExplainer:
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def explain(self, model, data, node_idx=None, **kwargs):
        raise NotImplementedError
    
    def get_explanation_metrics(self, explanation):
        """Calculate explanation quality metrics"""
        if 'edge_importance' in explanation:
            edge_imp = explanation['edge_importance']
            return {
                'sparsity': (edge_imp == 0).float().mean().item(),
                'max_importance': edge_imp.max().item(),
                'mean_importance': edge_imp.mean().item()
            }
        return {}

class GNNExplainer(BaseExplainer):
    def explain(self, model, data, node_idx=0, epochs=100, **kwargs):
        """GNNExplainer: Post-hoc explanation via edge mask optimization"""
        model.eval()
        
        # Initialize edge mask
        edge_index = data.edge_index
        edge_mask = torch.randn(edge_index.size(1), requires_grad=True, device=edge_index.device)
        
        # Get target prediction
        with torch.no_grad():
            target_out = model(data.x, edge_index)
            target_pred = target_out[node_idx].argmax()
        
        optimizer = torch.optim.Adam([edge_mask], lr=0.01)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Apply sigmoid to get probabilities
            mask_sigmoid = torch.sigmoid(edge_mask)
            
            # Apply mask by selecting edges
            selected_edges = mask_sigmoid > 0.5
            if selected_edges.sum() > 0:
                masked_edge_index = edge_index[:, selected_edges]
            else:
                # Keep at least one edge to avoid empty graph
                top_edge = mask_sigmoid.argmax()
                masked_edge_index = edge_index[:, [top_edge]]
            
            masked_out = model(data.x, masked_edge_index)
            
            # Loss: maintain prediction + sparsity
            pred_loss = F.cross_entropy(masked_out[node_idx:node_idx+1], target_pred.unsqueeze(0))
            sparsity_loss = mask_sigmoid.sum()
            
            loss = pred_loss + 0.01 * sparsity_loss
            loss.backward()
            optimizer.step()
        
        # Get final importance scores
        with torch.no_grad():
            edge_importance = torch.sigmoid(edge_mask)
            
            # Get important subgraph
            important_edges = edge_importance > 0.5
            if important_edges.sum() > 0:
                sub_edge_index = edge_index[:, important_edges]
                sub_edge_attr = edge_importance[important_edges]
            else:
                sub_edge_index = edge_index[:, :5]  # Keep top 5 edges
                sub_edge_attr = edge_importance[:5]
        
        return {
            'method': 'GNNExplainer',
            'node_idx': node_idx,
            'edge_importance': edge_importance,
            'subgraph_edges': sub_edge_index,
            'subgraph_weights': sub_edge_attr,
            'target_prediction': target_pred.item()
        }

class PGExplainer(BaseExplainer):
    def explain(self, model, data, node_idx=0, **kwargs):
        """PGExplainer: Parametric explainer using neural network"""
        model.eval()
        
        # Simple parametric explainer using node features
        x = data.x
        edge_index = data.edge_index
        
        # Create simple parametric model for edge importance
        edge_features = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Concatenate source and destination features
            edge_feat = torch.cat([x[src], x[dst]])
            edge_features.append(edge_feat)
        
        edge_features = torch.stack(edge_features)
        
        # Simple linear layer for edge importance
        with torch.no_grad():
            # Use cosine similarity as importance score
            target_feat = x[node_idx]
            edge_importance = []
            
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src == node_idx or dst == node_idx:
                    # High importance for direct connections
                    other_node = dst if src == node_idx else src
                    sim = F.cosine_similarity(target_feat.unsqueeze(0), x[other_node].unsqueeze(0))
                    edge_importance.append(sim.item())
                else:
                    # Lower importance for distant connections
                    edge_importance.append(0.1)
            
            edge_importance = torch.tensor(edge_importance, device=edge_index.device)
            
            # Get important subgraph
            important_edges = edge_importance > 0.3
            if important_edges.sum() > 0:
                sub_edge_index = edge_index[:, important_edges]
                sub_edge_attr = edge_importance[important_edges]
            else:
                sub_edge_index = edge_index[:, :5]
                sub_edge_attr = edge_importance[:5]
        
        return {
            'method': 'PGExplainer',
            'node_idx': node_idx,
            'edge_importance': edge_importance,
            'subgraph_edges': sub_edge_index,
            'subgraph_weights': sub_edge_attr,
            'parametric': True
        }

class SubgraphX(BaseExplainer):
    def explain(self, model, data, node_idx=0, num_samples=20, **kwargs):
        """SubgraphX: Shapley value-based subgraph explanation"""
        model.eval()
        
        # Simple Shapley value approximation for edges
        edge_index = data.edge_index
        num_edges = min(edge_index.size(1), 50)  # Limit for efficiency
        
        # Get baseline prediction
        with torch.no_grad():
            full_out = model(data.x, edge_index)
            full_pred = full_out[node_idx]
        
        # Calculate Shapley values for edges
        edge_shapley = []
        
        for edge_idx in range(num_edges):
            marginal_contributions = []
            
            for _ in range(num_samples):
                # Random subset of edges (excluding current edge)
                all_edges = list(range(num_edges))
                all_edges.remove(edge_idx)
                
                if len(all_edges) > 0:
                    subset_size = random.randint(0, min(len(all_edges), 10))
                    edge_subset = random.sample(all_edges, subset_size)
                    
                    # Prediction without current edge
                    if edge_subset:
                        subset_edges = edge_index[:, edge_subset]
                    else:
                        # Empty graph - use self-loop
                        subset_edges = torch.tensor([[node_idx], [node_idx]], device=edge_index.device)
                    
                    with torch.no_grad():
                        out_without = model(data.x, subset_edges)
                        pred_without = out_without[node_idx]
                    
                    # Prediction with current edge added
                    edge_subset_with = edge_subset + [edge_idx]
                    subset_edges_with = edge_index[:, edge_subset_with]
                    
                    with torch.no_grad():
                        out_with = model(data.x, subset_edges_with)
                        pred_with = out_with[node_idx]
                    
                    # Marginal contribution
                    contribution = torch.nn.functional.kl_div(
                        torch.log_softmax(pred_with, dim=0),
                        torch.softmax(pred_without, dim=0),
                        reduction='sum'
                    ).item()
                    marginal_contributions.append(abs(contribution))
                else:
                    marginal_contributions.append(0.0)
            
            edge_shapley.append(np.mean(marginal_contributions) if marginal_contributions else 0.0)
        
        # Pad with zeros for remaining edges
        while len(edge_shapley) < edge_index.size(1):
            edge_shapley.append(0.0)
        
        edge_importance = torch.tensor(edge_shapley, device=data.edge_index.device)
        
        # Get top important edges
        if len(edge_shapley) > 5:
            top_k = 5
            _, top_indices = edge_importance.topk(top_k)
            important_edges = edge_index[:, top_indices]
            important_weights = edge_importance[top_indices]
        else:
            important_edges = edge_index[:, :len(edge_shapley)]
            important_weights = edge_importance[:len(edge_shapley)]
        
        return {
            'method': 'SubgraphX',
            'node_idx': node_idx,
            'edge_importance': edge_importance,
            'subgraph_edges': important_edges,
            'subgraph_weights': important_weights,
            'shapley_values': edge_shapley[:num_edges]
        }

class ProtGNN(BaseExplainer):
    def explain(self, model, data, node_idx=0, num_prototypes=5, **kwargs):
        """ProtGNN: Prototype-based self-interpretable explanation"""
        model.eval()
        
        with torch.no_grad():
            # Get node embeddings
            embeddings = model.get_embeddings(data.x, data.edge_index) if hasattr(model, 'get_embeddings') else data.x
            
            # Find prototype nodes (most representative)
            target_embedding = embeddings[node_idx]
            
            # Calculate similarities to all nodes
            similarities = F.cosine_similarity(target_embedding.unsqueeze(0), embeddings, dim=1)
            
            # Get top-k most similar nodes as prototypes
            _, prototype_indices = similarities.topk(num_prototypes)
            prototype_similarities = similarities[prototype_indices]
            
            # Create prototype explanation
            prototypes = []
            for i, (proto_idx, sim) in enumerate(zip(prototype_indices, prototype_similarities)):
                prototypes.append({
                    'node_idx': proto_idx.item(),
                    'similarity': sim.item(),
                    'features': data.x[proto_idx].tolist(),
                    'label': data.y[proto_idx].item() if hasattr(data, 'y') else None
                })
            
            # Find edges connecting to prototypes
            edge_index = data.edge_index
            prototype_edges = []
            edge_importance = torch.zeros(edge_index.size(1), device=edge_index.device)
            
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src in prototype_indices or dst in prototype_indices:
                    if src == node_idx or dst == node_idx:
                        edge_importance[i] = 1.0  # High importance
                        prototype_edges.append(i)
                    elif src in prototype_indices and dst in prototype_indices:
                        edge_importance[i] = 0.7  # Medium importance
                        prototype_edges.append(i)
        
        return {
            'method': 'ProtGNN',
            'node_idx': node_idx,
            'prototypes': prototypes,
            'edge_importance': edge_importance,
            'prototype_edges': prototype_edges,
            'interpretable': True
        }

class GraphMask(BaseExplainer):
    def explain(self, model, data, node_idx=0, **kwargs):
        """GraphMask: Edge masking with structural reasoning"""
        model.eval()
        
        edge_index = data.edge_index
        
        # Initialize learnable edge masks
        edge_mask = torch.ones(edge_index.size(1), requires_grad=True, device=edge_index.device)
        
        # Get target prediction
        with torch.no_grad():
            target_out = model(data.x, edge_index)
            target_class = target_out[node_idx].argmax()
        
        optimizer = torch.optim.Adam([edge_mask], lr=0.1)
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Apply Gumbel softmax for differentiable masking
            mask_probs = torch.sigmoid(edge_mask)
            
            # Structural reasoning: encourage connected components
            structure_loss = 0
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src == node_idx or dst == node_idx:
                    # Encourage keeping edges connected to target node
                    structure_loss += (1 - mask_probs[i]) ** 2
            
            # Apply mask by selecting edges
            selected_edges = mask_probs > 0.5
            if selected_edges.sum() > 0:
                masked_edge_index = edge_index[:, selected_edges]
            else:
                # Keep top edges if none selected
                top_edges = mask_probs.topk(min(5, len(mask_probs)))[1]
                masked_edge_index = edge_index[:, top_edges]
            
            masked_out = model(data.x, masked_edge_index)
            
            # Fidelity loss
            fidelity_loss = F.cross_entropy(masked_out[node_idx:node_idx+1], target_class.unsqueeze(0))
            
            # Sparsity loss
            sparsity_loss = mask_probs.sum()
            
            total_loss = fidelity_loss + 0.01 * sparsity_loss + 0.1 * structure_loss
            total_loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            final_mask = torch.sigmoid(edge_mask)
            
            # Get important edges
            important_edges = final_mask > 0.5
            if important_edges.sum() > 0:
                sub_edge_index = edge_index[:, important_edges]
                sub_edge_weights = final_mask[important_edges]
            else:
                # Keep top 5 edges
                _, top_indices = final_mask.topk(min(5, len(final_mask)))
                sub_edge_index = edge_index[:, top_indices]
                sub_edge_weights = final_mask[top_indices]
        
        return {
            'method': 'GraphMask',
            'node_idx': node_idx,
            'edge_importance': final_mask,
            'subgraph_edges': sub_edge_index,
            'subgraph_weights': sub_edge_weights,
            'structural_reasoning': True
        }

class NeuronAnalysis(BaseExplainer):
    def explain(self, model, data, node_idx=0, **kwargs):
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
            if 'conv' in name.lower() or 'linear' in name.lower():
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            target_pred = output[node_idx]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze neuron activations
        neuron_concepts = {}
        
        for layer_name, activation in activations.items():
            if len(activation.shape) >= 2:
                # Get activation for target node
                if node_idx < activation.size(0):
                    node_activation = activation[node_idx]
                    
                    # Find most active neurons
                    top_neurons = node_activation.topk(min(5, len(node_activation)))[1]
                    
                    concepts = []
                    for neuron_idx in top_neurons:
                        # Simple concept mapping based on activation patterns
                        activation_val = node_activation[neuron_idx].item()
                        
                        if activation_val > 0.8:
                            concept = "high_confidence"
                        elif activation_val > 0.5:
                            concept = "medium_confidence"
                        else:
                            concept = "low_confidence"
                        
                        concepts.append({
                            'neuron_idx': neuron_idx.item(),
                            'activation': activation_val,
                            'concept': concept
                        })
                    
                    neuron_concepts[layer_name] = concepts
        
        # Global feature importance
        feature_importance = torch.abs(data.x[node_idx])
        top_features = feature_importance.topk(min(10, len(feature_importance)))[1]
        
        return {
            'method': 'NeuronAnalysis',
            'node_idx': node_idx,
            'neuron_concepts': neuron_concepts,
            'feature_importance': feature_importance,
            'top_features': top_features.tolist(),
            'prediction': target_pred.argmax().item(),
            'confidence': torch.softmax(target_pred, dim=0).max().item(),
            'global_analysis': True
        }

AVAILABLE_EXPLAINERS = {
    "GNNExplainer": GNNExplainer,
    "PGExplainer": PGExplainer,
    "SubgraphX": SubgraphX,
    "ProtGNN": ProtGNN,
    "GraphMask": GraphMask,
    "NeuronAnalysis": NeuronAnalysis,
}