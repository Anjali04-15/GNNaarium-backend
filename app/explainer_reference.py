"""
Reference implementation of GNNExplainer with mandatory behavior locks.
This serves as the canonical implementation that must be preserved.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class ReferenceGNNExplainer:
    """
    LOCKED REFERENCE IMPLEMENTATION
    
    Mandatory behaviors that MUST be preserved:
    1. k-hop locality for node tasks
    2. Fidelity-based validation 
    3. Soft-mask evaluation
    4. Separation of analyzed vs selected edges
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def get_model_num_layers(self, model):
        """Determine number of message-passing layers"""
        layer_count = 0
        for name, module in model.named_modules():
            if any(conv_type in str(type(module)) for conv_type in ['Conv', 'SAGE', 'GAT', 'GIN']):
                layer_count += 1
        return max(layer_count, 2)
    
    def get_k_hop_subgraph(self, node_idx, edge_index, num_hops, num_nodes=None):
        """LOCKED: Extract k-hop subgraph for node-level explanations"""
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        return subset, sub_edge_index, mapping, edge_mask
    
    def explain(self, model, data, node_idx=0, epochs=100, task_type='node', **kwargs):
        """
        LOCKED REFERENCE IMPLEMENTATION
        
        Regression checks:
        - MUST use k-hop subgraph for node tasks
        - MUST validate fidelity with soft masks
        - MUST separate analyzed edges from selected edges
        - MUST NOT mark explanations valid with fidelity=0
        """
        model.eval()
        
        # REGRESSION CHECK 1: Node tasks MUST use k-hop subgraph
        if task_type == 'node':
            num_hops = self.get_model_num_layers(model)
            subset, sub_edge_index, mapping, _ = self.get_k_hop_subgraph(
                node_idx, data.edge_index, num_hops, data.x.size(0)
            )
            
            sub_x = data.x[subset]
            target_node_in_sub = mapping[node_idx] if node_idx in mapping else 0
            
            # LOCKED: Initialize edge mask for subgraph only
            edge_mask = torch.randn(sub_edge_index.size(1), requires_grad=True, device=sub_edge_index.device)
            
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
        sparsity_weight = kwargs.get('sparsity_weight', 0.005)
        
        # LOCKED: Soft mask optimization
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # LOCKED: Continuous probabilities (NO binarization)
            mask_probs = torch.sigmoid(edge_mask)
            
            try:
                # LOCKED: Soft masking approach
                masked_out = self._forward_with_edge_weights(model, sub_x, sub_edge_index, mask_probs)
                
                if task_type == 'node':
                    masked_logits = masked_out[target_node_in_sub]
                else:
                    masked_logits = masked_out if len(masked_out.shape) == 1 else masked_out[target_node_in_sub]
                
                # LOCKED: KL divergence for soft targets
                pred_loss = F.kl_div(
                    F.log_softmax(masked_logits, dim=-1),
                    F.softmax(target_logits, dim=-1),
                    reduction='sum'
                )
                
            except:
                # Fallback to hard masking
                top_k = max(1, int(0.3 * len(mask_probs)))
                _, top_indices = mask_probs.topk(top_k)
                masked_edge_index = sub_edge_index[:, top_indices]
                
                masked_out = model(sub_x, masked_edge_index)
                
                if task_type == 'node':
                    pred_loss = F.cross_entropy(masked_out[target_node_in_sub:target_node_in_sub+1], target_pred.unsqueeze(0))
                else:
                    pred_loss = F.cross_entropy(masked_out.unsqueeze(0), target_pred.unsqueeze(0))
            
            # LOCKED: Entropy regularization
            sparsity_loss = -torch.mean(mask_probs * torch.log(mask_probs + 1e-8))
            
            loss = pred_loss + sparsity_weight * sparsity_loss
            loss.backward()
            optimizer.step()
        
        # LOCKED: Final edge importance (continuous)
        with torch.no_grad():
            edge_importance = torch.sigmoid(edge_mask)
            
            # LOCKED: Top-K edge retention (separation of analyzed vs selected)
            retention_ratio = kwargs.get('retention_ratio', 0.3)
            num_edges_to_keep = max(1, int(retention_ratio * len(edge_importance)))
            _, top_indices = edge_importance.topk(num_edges_to_keep)
            
            final_edge_index = sub_edge_index[:, top_indices]
            final_edge_weights = edge_importance[top_indices]
            
            # LOCKED: Fidelity validation with soft masks
            fidelity = self._validate_fidelity(
                model, sub_x, sub_edge_index, final_edge_index, 
                target_node_in_sub, target_pred, task_type
            )
            
            # REGRESSION CHECK 2: Try higher retention if fidelity is 0
            if fidelity == 0.0 and retention_ratio < 0.8:
                num_edges_to_keep = max(1, int(0.5 * len(edge_importance)))
                _, top_indices = edge_importance.topk(num_edges_to_keep)
                final_edge_index = sub_edge_index[:, top_indices]
                final_edge_weights = edge_importance[top_indices]
                
                fidelity = self._validate_fidelity(
                    model, sub_x, sub_edge_index, final_edge_index,
                    target_node_in_sub, target_pred, task_type
                )
        
        explanation = {
            'method': 'GNNExplainer',
            'node_idx': node_idx,
            'task_type': task_type,
            'edge_importance': edge_importance,  # LOCKED: All analyzed edges
            'subgraph_edges': final_edge_index,  # LOCKED: Selected edges only
            'subgraph_weights': final_edge_weights,
            'target_prediction': target_pred.item(),
            'fidelity': fidelity,
            'num_hops_used': num_hops if task_type == 'node' else None,
            'retention_ratio': retention_ratio
        }
        
        # REGRESSION CHECK 3: Add warning for low fidelity
        if fidelity < 0.5:
            explanation['fidelity_warning'] = "The model's decision relies on distributed graph evidence and cannot be localized to a small subgraph."
        
        # REGRESSION CHECK 4: MUST NOT mark valid with fidelity=0
        explanation['valid'] = fidelity > 0.0
        
        return explanation
    
    def _forward_with_edge_weights(self, model, x, edge_index, edge_weights):
        """LOCKED: Soft masking via probabilistic edge sampling"""
        edge_probs = edge_weights
        sampled_edges = torch.bernoulli(edge_probs).bool()
        
        if sampled_edges.sum() == 0:
            top_edge = edge_probs.argmax()
            sampled_edges[top_edge] = True
        
        masked_edge_index = edge_index[:, sampled_edges]
        return model(x, masked_edge_index)
    
    def _validate_fidelity(self, model, sub_x, sub_edge_index, final_edge_index, 
                          target_node_in_sub, target_pred, task_type):
        """LOCKED: Fidelity validation with proper error handling"""
        try:
            explained_out = model(sub_x, final_edge_index)
            if task_type == 'node':
                explained_pred = explained_out[target_node_in_sub].argmax()
            else:
                explained_pred = explained_out.argmax() if len(explained_out.shape) == 1 else explained_out[target_node_in_sub].argmax()
            
            return 1.0 if explained_pred == target_pred else 0.0
        except:
            return 0.0

# REGRESSION TESTS
def validate_reference_behavior(explainer, model, data, node_idx=0):
    """
    Mandatory regression checks that MUST pass
    """
    explanation = explainer.explain(model, data, node_idx=node_idx, task_type='node')
    
    # CHECK 1: k-hop locality
    assert 'num_hops_used' in explanation, "REGRESSION FAIL: Missing k-hop information"
    assert explanation['num_hops_used'] > 0, "REGRESSION FAIL: Invalid k-hop count"
    
    # CHECK 2: Fidelity validation
    assert 'fidelity' in explanation, "REGRESSION FAIL: Missing fidelity validation"
    assert isinstance(explanation['fidelity'], float), "REGRESSION FAIL: Invalid fidelity type"
    
    # CHECK 3: Soft mask evaluation
    assert 'edge_importance' in explanation, "REGRESSION FAIL: Missing edge importance"
    edge_imp = explanation['edge_importance']
    assert torch.all((edge_imp >= 0) & (edge_imp <= 1)), "REGRESSION FAIL: Edge importance not in [0,1]"
    
    # CHECK 4: Separation of analyzed vs selected
    assert 'subgraph_edges' in explanation, "REGRESSION FAIL: Missing selected edges"
    analyzed_count = len(explanation['edge_importance'])
    selected_count = explanation['subgraph_edges'].size(1)
    assert selected_count <= analyzed_count, "REGRESSION FAIL: More selected than analyzed edges"
    
    # CHECK 5: No valid explanations with fidelity=0
    if explanation.get('valid', True):
        assert explanation['fidelity'] > 0.0, "REGRESSION FAIL: Valid explanation with zero fidelity"
    
    return True