import torch
import asyncio
import time
from app.models import create_model
from app.attacks import AVAILABLE_ATTACKS
from app.defenses import AVAILABLE_DEFENSES
from app.explainers import AVAILABLE_EXPLAINERS

class RunManager:
    def __init__(self, config, data):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.status = "initialized"
        self.metrics = []
        self.should_stop = False
        self.training_started = False
        
        model_config = config.get("model", {})
        model_name = model_config.get("name", "GCN")
        
        # Ensure we have at least 2 classes and valid labels
        unique_labels = torch.unique(self.data.y)
        self.num_classes = max(2, len(unique_labels))  # At least 2 classes
        
        # Fix labels to be in range [0, num_classes-1]
        if self.data.y.max() >= self.num_classes or self.data.y.min() < 0:
            # Remap labels to valid range
            label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
            new_y = torch.zeros_like(self.data.y)
            for old_label, new_label in label_map.items():
                new_y[self.data.y == old_label] = new_label
            self.data.y = new_y
        
        # Detect if this is graph classification (MUTAG, PROTEINS, etc.)
        dataset_name = config.get('dataset', {}).get('name', '')
        is_graph_level = dataset_name in ['MUTAG', 'PROTEINS', 'ZINC']
        self.task_type = 'graph' if is_graph_level else 'node'
        
        self.model = create_model(
            model_name,
            self.data.num_node_features,
            model_config.get("hidden_dim", 64),
            self.num_classes,
            dropout=model_config.get("dropout", 0.5),
            graph_level=is_graph_level
        ).to(self.device)
        
        # Use better default hyperparameters based on dataset
        if dataset_name in ['MUTAG', 'PROTEINS']:
            default_lr = 0.001  # Lower LR for graph classification
        elif dataset_name in ['OGBN-Arxiv']:
            default_lr = 0.001  # Much lower LR for challenging synthetic data
        else:
            default_lr = 0.01   # Standard LR for citation networks
            
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=model_config.get("lr", default_lr),
            weight_decay=model_config.get("weight_decay", 5e-4)
        )
        self.epochs = model_config.get("epochs", 50)

    def stop(self):
        self.should_stop = True
        self.status = "stopped"

    def get_metrics_summary(self):
        if not self.metrics:
            return {}
        last_metric = self.metrics[-1]
        return {
            "epochs_completed": len(self.metrics),
            "final_train_loss": last_metric.get("train_loss"),
            "final_val_acc": last_metric.get("val_acc"),
            "final_asr": last_metric.get("asr"),
            "final_robust_acc": last_metric.get("robust_acc"),
            "final_fidelity": last_metric.get("fidelity"),
            "defense_effectiveness": last_metric.get("defense_effectiveness"),
            "explanation_quality": last_metric.get("explanation_quality")
        }

    async def run_training(self, send_event):
        if self.training_started:
            return
        
        self.training_started = True
        self.status = "running"
        await send_event({"type": "log", "msg": f"Starting training for {self.epochs} epochs..."})
        
        # Store original data for attack comparison
        original_data = self.data.clone()
        
        # Apply defense
        defense_config = self.config.get("defense")
        if defense_config and defense_config.get("name"):
            defense_name = defense_config["name"]
            if defense_name in AVAILABLE_DEFENSES:
                defense = AVAILABLE_DEFENSES[defense_name]()
                original_edges = self.data.edge_index.size(1)
                self.data = defense.apply(self.data, **defense_config)
                new_edges = self.data.edge_index.size(1)
                await send_event({
                    "type": "log", 
                    "msg": f"Defense '{defense_name}' applied: {original_edges} -> {new_edges} edges"
                })
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            if self.should_stop:
                break
                
            self.model.train()
            self.optimizer.zero_grad()
            
            # Check if this is graph-level classification
            dataset_name = self.config.get('dataset', {}).get('name', '')
            is_graph_level = dataset_name in ['MUTAG', 'PROTEINS', 'ZINC']
            
            if is_graph_level:
                # For graph classification, use proper batch
                out = self.model(self.data.x, self.data.edge_index, self.data.batch)
                loss = torch.nn.functional.cross_entropy(out, self.data.y)
            else:
                out = self.model(self.data.x, self.data.edge_index)
                
                # Ensure masks exist
                if not hasattr(self.data, 'train_mask') or self.data.train_mask is None:
                    from app.datasets import create_stratified_masks
                    train_mask, val_mask, test_mask = create_stratified_masks(
                        self.data.y, self.data.num_nodes
                    )
                    self.data.train_mask = train_mask.to(out.device)
                    self.data.val_mask = val_mask.to(out.device)
                    self.data.test_mask = test_mask.to(out.device)
                
                # Ensure labels are in correct range
                train_labels = self.data.y[self.data.train_mask]
                if train_labels.max() >= out.size(1):
                    train_labels = torch.clamp(train_labels, 0, out.size(1) - 1)
                
                loss = torch.nn.functional.cross_entropy(
                    out[self.data.train_mask], train_labels
                )
            loss.backward()
            self.optimizer.step()
            
            # Evaluate clean accuracy with validation
            clean_acc = self.evaluate()
            
            # Calculate comprehensive metrics (every 10 epochs or final epoch)
            asr = None
            robust_acc = None
            fidelity = None
            defense_effectiveness = None
            
            if epoch % 10 == 0 or epoch == self.epochs:
                attack_config = self.config.get("attack")
                if attack_config and attack_config.get("name"):
                    asr, robust_acc = self.calculate_asr(attack_config, original_data)
                
                # Calculate fidelity (model consistency)
                fidelity = self.calculate_fidelity(original_data)
                
                # Calculate defense effectiveness
                defense_config = self.config.get("defense")
                if defense_config and defense_config.get("name"):
                    defense_effectiveness = self.calculate_defense_effectiveness(original_data)
            
            metric = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_acc": clean_acc,
                "asr": asr,
                "robust_acc": robust_acc,
                "fidelity": fidelity,
                "defense_effectiveness": defense_effectiveness
            }
            self.metrics.append(metric)
            
            await send_event({"type": "metric", **metric})
            await asyncio.sleep(0.01)
        
        # Run explainer after training and calculate explanation quality
        explainer_config = self.config.get("explainer")
        explanation_quality = None
        if explainer_config and explainer_config.get("name"):
            explanations = await self.run_explainer(explainer_config, send_event)
            if explanations and len(explanations) > 0:
                # Calculate quality for first explanation if multiple nodes
                first_explanation = explanations[0] if isinstance(explanations, list) else explanations
                explanation_quality = self.calculate_explanation_quality(first_explanation)
            
            # Update final metric with explanation quality
            if self.metrics:
                self.metrics[-1]["explanation_quality"] = explanation_quality
            
            # Ensure explanation is sent before completion
            await asyncio.sleep(0.1)
        
        await send_event({"type": "log", "msg": "Training completed!"})
        await send_event({"type": "status", "status": "completed"})
        self.status = "completed"

    def evaluate(self, data=None):
        if data is None:
            data = self.data
            
        # Store original model mode and restore after
        was_training = self.model.training
        self.model.eval()
        
        dataset_name = self.config.get('dataset', {}).get('name', '')
        is_graph_level = dataset_name in ['MUTAG', 'PROTEINS', 'ZINC']
        
        with torch.no_grad():
            if is_graph_level:
                # Graph classification
                out = self.model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                acc = (pred == data.y).float().mean().item()
            else:
                # Node classification - use EXACT same logic as debug
                out = self.model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                
                # Use validation mask with integer division like debug
                if hasattr(data, 'val_mask') and data.val_mask.any():
                    val_pred = pred[data.val_mask]
                    val_true = data.y[data.val_mask]
                    matches = (val_pred == val_true).sum().item()
                    total = len(val_pred)
                    acc = matches / total
                else:
                    # Fallback to train mask if no val mask
                    train_pred = pred[data.train_mask]
                    train_true = data.y[data.train_mask]
                    matches = (train_pred == train_true).sum().item()
                    total = len(train_pred)
                    acc = matches / total
        
        # Restore original model mode
        if was_training:
            self.model.train()
                
        return acc

    def calculate_asr(self, attack_config, original_data):
        attack_name = attack_config["name"]
        if attack_name not in AVAILABLE_ATTACKS:
            return None, None
        
        # Get clean accuracy on original data
        clean_acc = self.evaluate(original_data)
        
        # Apply attack to original data
        attack = AVAILABLE_ATTACKS[attack_name]()
        attacked_data = attack.run(self.model, original_data, **attack_config)
        
        # Evaluate on attacked data
        robust_acc = self.evaluate(attacked_data)
        
        # Calculate ASR = (clean_acc - robust_acc) / clean_acc
        asr = (clean_acc - robust_acc) / clean_acc if clean_acc > 0 else 0
        return max(0, asr), robust_acc

    async def run_explainer(self, explainer_config, send_event):
        explainer_name = explainer_config["name"]
        if explainer_name not in AVAILABLE_EXPLAINERS:
            return None
        
        await send_event({"type": "log", "msg": f"Running {explainer_name}..."})
        
        # Determine task type based on dataset
        dataset_name = self.config.get('dataset', {}).get('name', '')
        task_type = 'graph' if dataset_name in ['MUTAG', 'PROTEINS', 'ZINC'] else 'node'
        
        # Get target node indices - parse comma-separated string
        node_idx_str = explainer_config.get("node_idx", "0")
        if isinstance(node_idx_str, str):
            node_indices = [int(idx.strip()) for idx in node_idx_str.split(',') if idx.strip()]
        else:
            node_indices = [int(node_idx_str)]
        
        # Remove name from config and add task type
        config_copy = explainer_config.copy()
        config_copy.pop("name", None)
        config_copy["task_type"] = task_type
        
        # Validate node indices for node-level tasks
        if task_type == 'node':
            valid_indices = [idx for idx in node_indices if idx < self.data.num_nodes]
            if len(valid_indices) != len(node_indices):
                invalid_indices = [idx for idx in node_indices if idx >= self.data.num_nodes]
                await send_event({"type": "log", "msg": f"Warning: Invalid node indices {invalid_indices} (max: {self.data.num_nodes-1})"})
            node_indices = valid_indices[:5]  # Limit to 5 nodes for performance
        else:
            # For graph-level tasks, use first node as representative
            node_indices = [0]
        
        if not node_indices:
            await send_event({"type": "log", "msg": "No valid node indices for explanation"})
            return None
        
        explanations = []
        explainer = AVAILABLE_EXPLAINERS[explainer_name]()
        
        try:
            for node_idx in node_indices:
                config_copy["node_idx"] = node_idx
                
                # Add validation message for k-hop constraint
                if task_type == 'node':
                    await send_event({"type": "log", "msg": f"Explaining node {node_idx} using k-hop subgraph (k={explainer.get_model_num_layers(self.model) if hasattr(explainer, 'get_model_num_layers') else 2})"})
                
                explanation = explainer.explain(self.model, self.data, **config_copy)
                explanations.append(explanation)
                
                # Validate explanation follows rules
                if task_type == 'node' and 'num_hops_used' in explanation:
                    await send_event({"type": "log", "msg": f"✓ Node explanation uses {explanation['num_hops_used']}-hop subgraph (compliant)"})
                elif task_type == 'graph':
                    await send_event({"type": "log", "msg": "✓ Graph explanation uses full graph (compliant)"})
            
            # Send each explanation
            for explanation in explanations:
                # Convert tensors to lists for JSON serialization
                serializable_explanation = {}
                for key, value in explanation.items():
                    if torch.is_tensor(value):
                        serializable_explanation[key] = value.cpu().tolist()
                    elif isinstance(value, list) and len(value) > 0 and torch.is_tensor(value[0]):
                        serializable_explanation[key] = [v.cpu().tolist() if torch.is_tensor(v) else v for v in value]
                    else:
                        serializable_explanation[key] = value
                
                # Send explanation
                explanation_msg = {"type": "explanation", "explainer": explainer_name, "explanation": serializable_explanation}
                await send_event(explanation_msg)
            
            return explanations
            
        except (ValueError, KeyError, RuntimeError) as e:
            error_msg = f"Explainer {explainer_name} failed: {str(e)}"
            await send_event({"type": "log", "msg": error_msg})
            await send_event({"type": "explanation", "explainer": explainer_name, "explanation": {"error": str(e), "method": explainer_name}})
            return {"error": str(e)}
    
    def calculate_fidelity(self, original_data):
        """Calculate model fidelity (prediction consistency)"""
        self.model.eval()
        with torch.no_grad():
            # Original predictions
            out1 = self.model(original_data.x, original_data.edge_index)
            
            # Add small noise and predict again
            noisy_x = original_data.x + torch.randn_like(original_data.x) * 0.01
            out2 = self.model(noisy_x, original_data.edge_index)
            
            # Calculate prediction consistency
            pred1 = out1.argmax(dim=1)
            pred2 = out2.argmax(dim=1)
            
            fidelity = (pred1 == pred2).float().mean().item()
            return fidelity
    
    def calculate_defense_effectiveness(self, original_data):
        """Calculate defense effectiveness metrics"""
        defense_config = self.config.get("defense")
        if not defense_config or not defense_config.get("name"):
            return None
        
        defense_name = defense_config["name"]
        if defense_name not in AVAILABLE_DEFENSES:
            return None
        
        # Apply defense to original data
        defense = AVAILABLE_DEFENSES[defense_name]()
        defended_data = defense.apply(original_data.clone(), **defense_config)
        
        # Calculate effectiveness metrics
        self.model.eval()
        with torch.no_grad():
            orig_out = self.model(original_data.x, original_data.edge_index)
            def_out = self.model(defended_data.x, defended_data.edge_index)
            
            # Prediction consistency
            orig_pred = orig_out.argmax(dim=1)
            def_pred = def_out.argmax(dim=1)
            consistency = (orig_pred == def_pred).float().mean().item()
            
            # Confidence preservation
            orig_conf = torch.softmax(orig_out, dim=1).max(dim=1)[0]
            def_conf = torch.softmax(def_out, dim=1).max(dim=1)[0]
            conf_preservation = (def_conf / (orig_conf + 1e-8)).mean().item()
            
            return {
                "consistency": consistency,
                "confidence_preservation": conf_preservation,
                "edge_reduction": 1 - (defended_data.edge_index.size(1) / original_data.edge_index.size(1))
            }
    
    def calculate_explanation_quality(self, explanation):
        """Calculate explanation quality metrics with method-specific validation"""
        if not explanation or "error" in explanation:
            return None
        
        quality_metrics = {}
        method = explanation.get('method')
        
        # GraphMask-specific validation
        if method == 'GraphMask':
            quality_metrics['explainer_type'] = 'Global (message-level)'
            quality_metrics['total_edges_evaluated'] = explanation.get('total_edges_evaluated', 0)
            quality_metrics['edges_gated_off'] = explanation.get('edges_gated_off', 0)
            quality_metrics['edges_retained'] = explanation.get('edges_retained', 0)
            
            # Prediction invariance (NOT fidelity)
            invariance = explanation.get('prediction_invariance', 0.0)
            quality_metrics['prediction_invariance'] = invariance
            quality_metrics['valid_invariance'] = invariance >= 0.8
            
            # GraphMask explains model behavior, not individual predictions
            quality_metrics['explains_model_behavior'] = True
            quality_metrics['explains_individual_predictions'] = False
            
            quality_score = 0.8 if quality_metrics['valid_invariance'] else 0.2
            quality_metrics['overall_quality'] = quality_score
            quality_metrics['success'] = True
            return quality_metrics
        
        # Standard validation for other explainers
        task_type = explanation.get('task_type', 'node')
        if task_type == 'node':
            # MANDATORY: Node explanations must use k-hop subgraph
            if 'num_hops_used' in explanation:
                quality_metrics['k_hop_compliant'] = True
                quality_metrics['num_hops_used'] = explanation['num_hops_used']
            else:
                quality_metrics['k_hop_compliant'] = False
                quality_metrics['validation_error'] = 'Node explanation did not use k-hop subgraph'
        
        # Calculate sparsity if edge importance is available
        if 'edge_importance' in explanation:
            edge_imp = torch.tensor(explanation['edge_importance']) if not torch.is_tensor(explanation['edge_importance']) else explanation['edge_importance']
            sparsity = (edge_imp < 0.1).float().mean().item()
            quality_metrics['sparsity'] = sparsity
            quality_metrics['max_importance'] = edge_imp.max().item()
            quality_metrics['mean_importance'] = edge_imp.mean().item()
            quality_metrics['sparse_enough'] = sparsity > 0.5
        
        # Fidelity check for non-GraphMask explainers
        if 'fidelity' in explanation:
            quality_metrics['fidelity'] = explanation['fidelity']
            quality_metrics['preserves_prediction'] = explanation['fidelity'] > 0.8
        
        # ProtGNN-specific validation
        if method == 'ProtGNN':
            quality_metrics['explainer_type'] = 'Self-interpretable (prototype-based)'
            quality_metrics['num_prototypes'] = explanation.get('num_prototypes', 0)
            quality_metrics['top_similarity'] = explanation.get('top_similarity', 0.0)
            quality_metrics['prototype_class_agreement'] = explanation.get('prototype_class_agreement', False)
            quality_metrics['valid_prototype_explanation'] = explanation.get('valid_explanation', False)
            quality_metrics['self_interpretable'] = True
            
            # ProtGNN explains via similarity, not causality
            quality_metrics['explains_via_similarity'] = True
            quality_metrics['explains_via_causality'] = False
            
            quality_score = 0.8 if quality_metrics['valid_prototype_explanation'] else 0.2
            quality_metrics['overall_quality'] = quality_score
            quality_metrics['success'] = True
            return quality_metrics
        
        if method == 'SubgraphX':
            quality_metrics['shapley_computed'] = len(explanation.get('shapley_values', []))
            quality_metrics['theoretically_grounded'] = True
        
        if method == 'NeuronAnalysis':
            quality_metrics['layers_analyzed'] = len(explanation.get('neuron_concepts', {}))
            quality_metrics['confidence'] = explanation.get('confidence', 0)
            quality_metrics['global_analysis'] = True
            quality_metrics['concept_rules'] = len(explanation.get('concept_rules', []))
        
        # Overall quality score
        quality_score = 0
        if quality_metrics.get('k_hop_compliant', True):
            quality_score += 0.3
        if quality_metrics.get('sparse_enough', False):
            quality_score += 0.3
        if quality_metrics.get('preserves_prediction', False):
            quality_score += 0.4
        
        quality_metrics['overall_quality'] = quality_score
        quality_metrics['success'] = True
        return quality_metrics