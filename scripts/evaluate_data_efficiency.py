#!/usr/bin/env python
"""
Data Efficiency Evaluation Script for PI-JEPA.

This script implements the data efficiency comparison framework that evaluates
PI-JEPA against baseline models (FNO, Geo-FNO, DeepONet) across varying amounts
of labeled training data.

Paper specifications:
- PI-JEPA: Pretrain on N_u unlabeled + finetune on N_l labeled
- Baselines: Train from scratch on same N_l labeled samples
- Sweep N_l ∈ {10, 25, 50, 100, 250, 500}
- Report relative L2 error on held-out test set
- Use identical train/test splits across all models

Validates: Requirement 6 (Data Efficiency Comparison Framework)
- AC 6.1: Train PI-JEPA with pretraining on N_u unlabeled + finetuning on N_l labeled
- AC 6.2: Train baselines (FNO, Geo-FNO, DeepONet) from scratch on same N_l labeled
- AC 6.3: Report relative L2 error on held-out test set
- AC 6.4: Sweep N_l ∈ {10, 25, 50, 100, 250, 500}
- AC 6.5: Use identical train/test splits across all models
- AC 6.6: Report results compatible with existing benchmark_comparison.json
"""

import os
import sys
import json
import argparse
import copy
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from models import ViTEncoder, PredictionHead, build_encoder
from utils import load_config
from benchmarks import FNOWrapper, DeepONetWrapper
from benchmarks.utils import set_seed, BenchmarkConfig, BenchmarkTrainer


# ============================================================================
# Data Efficiency Evaluator
# ============================================================================

class DataEfficiencyEvaluator:
    """
    Evaluator for data efficiency comparison between PI-JEPA and baselines.
    
    Validates: Requirement 6 (Data Efficiency Comparison Framework)
    - AC 6.1: Train PI-JEPA with pretraining on N_u unlabeled + finetuning on N_l labeled
    - AC 6.2: Train baselines (FNO, Geo-FNO, DeepONet) from scratch on same N_l labeled
    - AC 6.3: Report relative L2 error on held-out test set
    - AC 6.4: Sweep N_l ∈ {10, 25, 50, 100, 250, 500}
    - AC 6.5: Use identical train/test splits across all models
    - AC 6.6: Report results compatible with existing benchmark_comparison.json
    """
    
    # Default labeled sample counts per paper specification
    DEFAULT_N_LABELED_SWEEP = [10, 25, 50, 100, 250, 500]
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "outputs/data_efficiency",
        device: Optional[torch.device] = None
    ):
        """
        Initialize data efficiency evaluator.
        
        Args:
            config: Evaluation configuration
            output_dir: Directory for saving results
            device: Device for training/evaluation
        """
        self.config = config
        self.output_dir = output_dir
        
        # Set device
        if device is not None:
            self.device = device
        elif config.get("experiment", {}).get("device") is not None:
            self.device = torch.device(config["experiment"]["device"])
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        
        # Set seed for reproducibility
        self.seed = config.get("experiment", {}).get("seed", 42)
        set_seed(self.seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training parameters
        finetuning_cfg = config.get("finetuning", {})
        self.finetune_epochs = finetuning_cfg.get("epochs", 100)
        self.finetune_lr = float(finetuning_cfg.get("optim", {}).get("lr", 1e-3))
        self.batch_size = finetuning_cfg.get("batch_size", 32)
        
        # Baseline training parameters — match finetuning epochs for fair comparison
        self.baseline_epochs = config.get("evaluation", {}).get(
            "baseline_epochs", self.finetune_epochs
        )
        self.baseline_lr = config.get("evaluation", {}).get(
            "baseline_lr", 1e-3
        )
    
    def _create_data_splits(
        self,
        n_train: int = 1000,
        n_test: int = 200
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test data loaders with fixed splits.
        
        Validates: AC 6.5 - Use identical train/test splits across all models
        
        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Set seed for reproducible splits
        set_seed(self.seed)
        
        data_cfg = self.config.get("data", {})
        resolution = data_cfg.get("grid_size", 64)
        
        # First, check for generated data in data/darcy directory
        generated_train = os.path.join("data", "darcy", "darcy_train.pt")
        generated_test = os.path.join("data", "darcy", "darcy_test.pt")
        
        if os.path.exists(generated_train) and os.path.exists(generated_test):
            print(f"Loading from generated data: {generated_train}")
            
            train_data = torch.load(generated_train, weights_only=False)
            test_data = torch.load(generated_test, weights_only=False)
            
            # x: coefficient field, y: solution field
            x_train = train_data['x'][:n_train]  # (N, 1, H, W)
            y_train = train_data['y'][:n_train]  # (N, 1, H, W)
            x_test = test_data['x'][:n_test]
            y_test = test_data['y'][:n_test]
            
            print(f"  Train: {x_train.shape}, Test: {x_test.shape}")
            
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed)
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            return train_loader, test_loader
        
        # Fall back to neuralop
        try:
            try:
                from neuralop.data.datasets import load_darcy_flow_small
            except ImportError:
                from neuralop.datasets import load_darcy_flow_small
        
            train_loader, test_loaders, _ = load_darcy_flow_small(
                n_train=n_train,
                n_tests=[n_test],
                batch_size=self.batch_size,
                test_batch_sizes=[self.batch_size],
            )
            
            # test_loaders is a dict keyed by resolution, get the first one
            test_loader = list(test_loaders.values())[0]
            
            return train_loader, test_loader
            
        except ImportError:
            # Fallback: create synthetic data for testing
            print("Warning: neuralop not available, using synthetic data")
            
            # Create synthetic data with fixed seed
            torch.manual_seed(self.seed)
            
            x_train = torch.randn(n_train, 1, resolution, resolution)
            y_train = torch.randn(n_train, 1, resolution, resolution)
            
            x_test = torch.randn(n_test, 1, resolution, resolution)
            y_test = torch.randn(n_test, 1, resolution, resolution)
            
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed)
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            return train_loader, test_loader
    
    def _limit_dataset(
        self,
        data_loader: DataLoader,
        n_samples: int
    ) -> DataLoader:
        """
        Limit dataset to n_samples with fixed seed.
        
        Args:
            data_loader: Original data loader
            n_samples: Number of samples to use
            
        Returns:
            New DataLoader with limited samples
        """
        dataset = data_loader.dataset
        n_available = len(dataset)
        n_use = min(n_samples, n_available)
        
        # Use fixed indices for reproducibility
        indices = list(range(n_use))
        subset = Subset(dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=min(self.batch_size, n_use),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed)
        )
    
    def _prepare_batch(
        self,
        batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for training/evaluation.
        
        Args:
            batch: Batch from data loader
            
        Returns:
            Tuple of (x, y) tensors on device
        """
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else batch[0]
        elif isinstance(batch, dict):
            x = batch.get("x", batch.get("input", batch.get("coefficient")))
            y = batch.get("y", batch.get("target", batch.get("solution")))
            if x is None:
                raise KeyError("Batch must contain 'x', 'input', or 'coefficient' key")
            if y is None:
                raise KeyError("Batch must contain 'y', 'target', or 'solution' key")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        
        # Ensure shape is (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if y.dim() == 3:
            y = y.unsqueeze(1)
        
        return x, y
    
    def _compute_relative_l2(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> float:
        """
        Compute relative L2 error.
        
        Validates: AC 6.3 - Report relative L2 error on held-out test set
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            eps: Small constant for numerical stability
            
        Returns:
            Relative L2 error as float
        """
        diff = pred - target
        l2_error = torch.sqrt((diff ** 2).sum())
        l2_norm = torch.sqrt((target ** 2).sum()) + eps
        return (l2_error / l2_norm).item()
    
    def _finetune_pijepa(
        self,
        encoder: nn.Module,
        train_loader: DataLoader,
        n_labeled: int,
        full_finetune: bool = False
    ) -> nn.Module:
        """
        Finetune PI-JEPA on n_labeled samples.
        
        Validates: AC 6.1 - Train PI-JEPA with pretraining + finetuning on N_l labeled
        
        Args:
            encoder: Pretrained encoder (will be frozen unless full_finetune=True)
            train_loader: Training data loader
            n_labeled: Number of labeled samples to use
            full_finetune: If True, also finetune the encoder (not just prediction head)
            
        Returns:
            Trained prediction head
        """
        # Check if full finetuning is enabled
        finetuning_cfg = self.config.get("finetuning", {})
        full_finetune = full_finetune or finetuning_cfg.get("full_finetune", {}).get("enabled", False)
        
        if full_finetune:
            # Unfreeze encoder for full finetuning
            encoder.train()
            for param in encoder.parameters():
                param.requires_grad = True
            print("  [PI-JEPA] Full finetuning enabled (encoder unfrozen)")
        else:
            # Freeze encoder
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        
        # Build prediction head
        encoder_cfg = self.config.get("model", {}).get("encoder", {})
        head_cfg = finetuning_cfg.get("prediction_head", {})
        
        embed_dim = encoder_cfg.get("embed_dim", 384)
        hidden_dim = head_cfg.get("hidden_dim", 512)
        output_channels = head_cfg.get("output_channels", 1)
        image_size = encoder_cfg.get("image_size", 64)
        patch_size = encoder_cfg.get("patch_size", 8)
        
        prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            image_size=image_size,
            patch_size=patch_size
        ).to(self.device)
        
        # Limit dataset
        limited_loader = self._limit_dataset(train_loader, n_labeled)
        
        # Optimizer - include encoder if full finetuning
        if full_finetune:
            encoder_lr = self.finetune_lr * finetuning_cfg.get("full_finetune", {}).get("encoder_lr_multiplier", 0.1)
            optimizer = torch.optim.AdamW([
                {'params': prediction_head.parameters(), 'lr': self.finetune_lr},
                {'params': encoder.parameters(), 'lr': encoder_lr}
            ])
        else:
            optimizer = torch.optim.AdamW(
                prediction_head.parameters(),
                lr=self.finetune_lr
            )
        
        # Training loop
        prediction_head.train()
        
        # Debug: check shapes on first batch
        first_batch = True
        
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in limited_loader:
                x, y = self._prepare_batch(batch)
                
                optimizer.zero_grad()
                
                if full_finetune:
                    z = encoder(x)
                else:
                    with torch.no_grad():
                        z = encoder(x)
                
                if first_batch:
                    print(f"  [PI-JEPA] Input shape: {x.shape}, Target shape: {y.shape}")
                    print(f"  [PI-JEPA] Encoder output shape: {z.shape}")
                    first_batch = False
                
                y_pred = prediction_head(z)
                loss = F.mse_loss(y_pred, y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"  [PI-JEPA] Epoch {epoch+1}/{self.finetune_epochs} Loss: {avg_loss:.6f}")
        
        return prediction_head
    
    def _evaluate_pijepa(
        self,
        encoder: nn.Module,
        prediction_head: nn.Module,
        test_loader: DataLoader
    ) -> float:
        """
        Evaluate PI-JEPA on test set.
        
        Args:
            encoder: Pretrained encoder
            prediction_head: Trained prediction head
            test_loader: Test data loader
            
        Returns:
            Relative L2 error
        """
        encoder.eval()
        prediction_head.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = self._prepare_batch(batch)
                
                z = encoder(x)
                y_pred = prediction_head(z)
                
                all_preds.append(y_pred.cpu())
                all_targets.append(y.cpu())
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return self._compute_relative_l2(preds, targets)
    
    def _train_pijepa_scratch(
        self,
        train_loader: DataLoader,
        n_labeled: int
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Train PI-JEPA architecture from scratch (no pretraining).
        
        This provides a baseline to measure the benefit of pretraining.
        
        Args:
            train_loader: Training data loader
            n_labeled: Number of labeled samples to use
            
        Returns:
            Tuple of (encoder, prediction_head)
        """
        # Build fresh encoder using factory
        encoder = build_encoder(self.config, in_channels=1).to(self.device)
        encoder.train()
        
        # Build prediction head
        encoder_cfg = self.config.get("model", {}).get("encoder", {})
        finetuning_cfg = self.config.get("finetuning", {})
        head_cfg = finetuning_cfg.get("prediction_head", {})
        
        embed_dim = encoder_cfg.get("embed_dim", 384)
        hidden_dim = head_cfg.get("hidden_dim", 512)
        output_channels = head_cfg.get("output_channels", 1)
        image_size = encoder_cfg.get("image_size", 64)
        patch_size = encoder_cfg.get("patch_size", 8)
        
        prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            image_size=image_size,
            patch_size=patch_size
        ).to(self.device)
        
        # Limit dataset
        limited_loader = self._limit_dataset(train_loader, n_labeled)
        
        # Optimizer for both encoder and prediction head
        optimizer = torch.optim.AdamW([
            {'params': encoder.parameters(), 'lr': self.finetune_lr * 0.1},
            {'params': prediction_head.parameters(), 'lr': self.finetune_lr}
        ])
        
        # Training loop
        encoder.train()
        prediction_head.train()
        
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in limited_loader:
                x, y = self._prepare_batch(batch)
                
                optimizer.zero_grad()
                
                z = encoder(x)
                y_pred = prediction_head(z)
                loss = F.mse_loss(y_pred, y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"  [PI-JEPA scratch] Epoch {epoch+1}/{self.finetune_epochs} Loss: {avg_loss:.6f}")
        
        return encoder, prediction_head
    
    def _train_baseline(
        self,
        model_wrapper: Any,
        train_loader: DataLoader,
        n_labeled: int
    ) -> None:
        """
        Train baseline model from scratch on n_labeled samples.
        
        Validates: AC 6.2 - Train baselines from scratch on same N_l labeled
        
        Args:
            model_wrapper: Baseline model wrapper
            train_loader: Training data loader
            n_labeled: Number of labeled samples to use
        """
        # Limit dataset
        limited_loader = self._limit_dataset(train_loader, n_labeled)
        
        # Convert to dict format expected by benchmark wrappers
        dict_loader = self._convert_to_dict_loader(limited_loader)
        
        # Train model
        model_wrapper.train_model(
            dict_loader,
            epochs=self.baseline_epochs,
            lr=self.baseline_lr
        )
    
    def _convert_to_dict_loader(
        self,
        data_loader: DataLoader
    ) -> DataLoader:
        """
        Convert data loader to dict format expected by benchmark wrappers.
        
        Args:
            data_loader: Original data loader
            
        Returns:
            DataLoader yielding dict batches
        """
        class DictDataset:
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                if isinstance(item, (tuple, list)):
                    x = item[0]
                    y = item[1] if len(item) > 1 else item[0]
                elif isinstance(item, dict):
                    x = item.get("x", item.get("input"))
                    y = item.get("y", item.get("target"))
                else:
                    x = item
                    y = item
                return {"x": x, "y": y}
        
        dict_dataset = DictDataset(data_loader.dataset)
        
        return DataLoader(
            dict_dataset,
            batch_size=data_loader.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed)
        )
    
    def _evaluate_baseline(
        self,
        model_wrapper: Any,
        test_loader: DataLoader
    ) -> float:
        """
        Evaluate baseline model on test set.
        
        Args:
            model_wrapper: Trained baseline model wrapper
            test_loader: Test data loader
            
        Returns:
            Relative L2 error
        """
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = self._prepare_batch(batch)
                
                y_pred = model_wrapper.predict(x)
                
                all_preds.append(y_pred.cpu())
                all_targets.append(y.cpu())
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return self._compute_relative_l2(preds, targets)
    
    def _create_baseline_models(self) -> Dict[str, Any]:
        """
        Create baseline model wrappers.
        
        Validates: AC 6.2 - Train baselines (FNO, Geo-FNO, DeepONet) from scratch
        
        Returns:
            Dict mapping model name to wrapper
        """
        baselines = {}
        
        # FNO - Fourier Neural Operator
        baselines["fno"] = FNOWrapper(
            device=self.device,
            in_channels=1,
            out_channels=1,
            modes=(16, 16),
            hidden_channels=64,
            n_layers=4
        )
        
        # Geo-FNO (requires neuralop)
        try:
            from benchmarks.geo_fno import GeoFNOWrapper
            baselines["geo_fno"] = GeoFNOWrapper(device=self.device)
        except ImportError:
            print("Warning: Geo-FNO not available (requires neuralop)")
        except Exception as e:
            print(f"Warning: Geo-FNO initialization failed: {e}")
        
        # DeepONet - Deep Operator Network
        baselines["deeponet"] = DeepONetWrapper(device=self.device)
        
        return baselines
    
    def _create_fresh_baseline(self, model_name: str) -> Any:
        """
        Create a fresh baseline model instance for training.
        
        This ensures each N_l evaluation starts with a fresh model.
        
        Args:
            model_name: Name of the baseline model
            
        Returns:
            Fresh model wrapper instance
        """
        if model_name == "fno":
            return FNOWrapper(
                device=self.device,
                in_channels=1,
                out_channels=1,
                modes=(16, 16),
                hidden_channels=64,
                n_layers=4
            )
        elif model_name == "geo_fno":
            try:
                from benchmarks.geo_fno import GeoFNOWrapper
                return GeoFNOWrapper(device=self.device)
            except (ImportError, Exception):
                return None
        elif model_name == "deeponet":
            return DeepONetWrapper(device=self.device)
        else:
            raise ValueError(f"Unknown baseline model: {model_name}")
    
    def run_comparison(
        self,
        pretrained_encoder: nn.Module,
        n_unlabeled: int = 1000,
        n_labeled_sweep: Optional[List[int]] = None,
        baselines_to_run: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Run full data efficiency comparison.
        
        Validates: Requirement 6 (Data Efficiency Comparison Framework)
        - AC 6.1: Train PI-JEPA with pretraining on N_u unlabeled + finetuning on N_l labeled
        - AC 6.2: Train baselines (FNO, Geo-FNO, DeepONet) from scratch on same N_l labeled
        - AC 6.3: Report relative L2 error on held-out test set
        - AC 6.4: Sweep N_l ∈ {10, 25, 50, 100, 250, 500}
        - AC 6.5: Use identical train/test splits across all models
        
        Args:
            pretrained_encoder: Pretrained PI-JEPA encoder
            n_unlabeled: Number of unlabeled samples used for pretraining
            n_labeled_sweep: List of labeled sample counts to evaluate
            baselines_to_run: List of baseline names to run (default: all available)
            
        Returns:
            Dict mapping model name to {n_labeled: relative_l2_error}
        """
        if n_labeled_sweep is None:
            n_labeled_sweep = self.DEFAULT_N_LABELED_SWEEP
        
        if baselines_to_run is None:
            baselines_to_run = ["fno", "geo_fno", "deeponet"]
        
        print(f"\n{'='*60}")
        print("Data Efficiency Comparison")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"N_unlabeled (PI-JEPA pretraining): {n_unlabeled}")
        print(f"N_labeled sweep: {n_labeled_sweep}")
        print(f"Finetune epochs: {self.finetune_epochs}")
        print(f"Baseline epochs: {self.baseline_epochs}")
        print(f"Baselines: {baselines_to_run}")
        
        # Create data splits (AC 6.5: identical splits for all models)
        train_loader, test_loader = self._create_data_splits()
        
        # Initialize results
        results = {"pi_jepa": {}}
        for baseline in baselines_to_run:
            results[baseline] = {}
        
        # Run comparison for each n_labeled
        for n_labeled in n_labeled_sweep:
            print(f"\n{'-'*40}")
            print(f"N_labeled = {n_labeled}")
            print(f"{'-'*40}")
            
            # Reset seed for each n_labeled to ensure reproducibility
            set_seed(self.seed)
            
            # 1. Finetune and evaluate PI-JEPA (AC 6.1)
            print("Training PI-JEPA (finetuning)...")
            
            # Create a fresh copy of encoder for finetuning
            encoder_copy = copy.deepcopy(pretrained_encoder)
            encoder_copy.to(self.device)
            
            prediction_head = self._finetune_pijepa(
                encoder_copy,
                train_loader,
                n_labeled
            )
            pijepa_error = self._evaluate_pijepa(
                encoder_copy,
                prediction_head,
                test_loader
            )
            results["pi_jepa"][n_labeled] = pijepa_error
            print(f"  PI-JEPA relative L2 error: {pijepa_error:.6f}")
            
            # 2. Train PI-JEPA from scratch (no pretraining) for comparison
            print("Training PI-JEPA from scratch (no pretraining)...")
            set_seed(self.seed)
            
            scratch_encoder, scratch_head = self._train_pijepa_scratch(
                train_loader,
                n_labeled
            )
            scratch_error = self._evaluate_pijepa(
                scratch_encoder,
                scratch_head,
                test_loader
            )
            if "pi_jepa_scratch" not in results:
                results["pi_jepa_scratch"] = {}
            results["pi_jepa_scratch"][n_labeled] = scratch_error
            print(f"  PI-JEPA (scratch) relative L2 error: {scratch_error:.6f}")
            
            # 3. Train and evaluate baselines from scratch (AC 6.2)
            for baseline_name in baselines_to_run:
                print(f"Training {baseline_name} from scratch...")
                set_seed(self.seed)  # Reset seed for fair comparison
                
                # Create fresh model for each N_l
                wrapper = self._create_fresh_baseline(baseline_name)
                
                if wrapper is None:
                    print(f"  Skipping {baseline_name} (not available)")
                    continue
                
                try:
                    self._train_baseline(wrapper, train_loader, n_labeled)
                    error = self._evaluate_baseline(wrapper, test_loader)
                    results[baseline_name][n_labeled] = error
                    print(f"  {baseline_name} relative L2 error: {error:.6f}")
                except Exception as e:
                    print(f"  Error training {baseline_name}: {e}")
        
        # Remove empty results (models that weren't available)
        results = {k: v for k, v in results.items() if v}
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Dict[int, float]],
        filename: str = "benchmark_comparison.json",
        include_metadata: bool = True
    ) -> str:
        """
        Save results to JSON file.
        
        Validates: AC 6.6 - Report results compatible with existing benchmark_comparison.json
        
        Args:
            results: Results dict from run_comparison
            filename: Output filename
            include_metadata: Whether to include metadata in output
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert int keys to strings for JSON compatibility
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                str(k): v for k, v in model_results.items()
            }
        
        # Optionally add metadata
        if include_metadata:
            json_results["_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "seed": self.seed,
                "finetune_epochs": self.finetune_epochs,
                "baseline_epochs": self.baseline_epochs,
                "batch_size": self.batch_size,
                "device": str(self.device)
            }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def save_results_compatible(
        self,
        results: Dict[str, Dict[int, float]],
        filename: str = "benchmark_comparison.json"
    ) -> str:
        """
        Save results in format compatible with existing benchmark_comparison.json.
        
        This version excludes metadata for exact compatibility.
        
        Validates: AC 6.6 - Report results compatible with existing benchmark_comparison.json
        
        Args:
            results: Results dict from run_comparison
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        return self.save_results(results, filename, include_metadata=False)
    
    def print_summary(
        self,
        results: Dict[str, Dict[int, float]]
    ) -> None:
        """
        Print summary table of results.
        
        Args:
            results: Results dict from run_comparison
        """
        print(f"\n{'='*60}")
        print("Data Efficiency Summary")
        print(f"{'='*60}")
        
        # Get all n_labeled values
        all_n_labeled = sorted(set(
            n for model_results in results.values()
            for n in model_results.keys()
        ))
        
        # Print header
        header = "N_labeled".ljust(12)
        for model in results.keys():
            header += model.ljust(15)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for n_labeled in all_n_labeled:
            row = str(n_labeled).ljust(12)
            for model, model_results in results.items():
                error = model_results.get(n_labeled, float('nan'))
                row += f"{error:.4f}".ljust(15)
            print(row)
        
        # Print improvement summary
        print(f"\n{'='*60}")
        print("PI-JEPA Improvement over Baselines")
        print(f"{'='*60}")
        
        if "pi_jepa" in results:
            for model in results.keys():
                if model == "pi_jepa":
                    continue
                
                improvements = []
                for n_labeled in all_n_labeled:
                    pijepa_error = results["pi_jepa"].get(n_labeled)
                    baseline_error = results[model].get(n_labeled)
                    
                    if pijepa_error is not None and baseline_error is not None:
                        improvement = (baseline_error - pijepa_error) / baseline_error * 100
                        improvements.append(improvement)
                
                if improvements:
                    avg_improvement = sum(improvements) / len(improvements)
                    print(f"  vs {model}: {avg_improvement:.1f}% average improvement")


# ============================================================================
# Model Loading
# ============================================================================

def load_pretrained_encoder(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> ViTEncoder:
    """
    Load pretrained encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to pretraining checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded ViTEncoder
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Verify checkpoint type
    checkpoint_type = checkpoint.get('checkpoint_type', 'unknown')
    if checkpoint_type != 'pretraining':
        print(f"Warning: Expected pretraining checkpoint, got '{checkpoint_type}'")
    
    # Build encoder using factory (supports vit, fourier, multiscale_fourier)
    encoder = build_encoder(config, in_channels=1).to(device)
    
    # Load weights
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"Loaded encoder weights from {checkpoint_path}")
    else:
        raise KeyError("Checkpoint does not contain 'encoder_state_dict'")
    
    return encoder


# ============================================================================
# Main Entry Point
# ============================================================================

def evaluate_data_efficiency(
    pretrain_checkpoint: str,
    config_path: str = "configs/darcy.yaml",
    output_dir: str = "outputs/data_efficiency",
    n_labeled_sweep: Optional[List[int]] = None,
    baselines: Optional[List[str]] = None,
    compatible_format: bool = True
) -> Dict[str, Dict[int, float]]:
    """
    Run data efficiency evaluation.
    
    Validates: Requirement 6 (Data Efficiency Comparison Framework)
    - AC 6.1: Train PI-JEPA with pretraining on N_u unlabeled + finetuning on N_l labeled
    - AC 6.2: Train baselines (FNO, Geo-FNO, DeepONet) from scratch on same N_l labeled
    - AC 6.3: Report relative L2 error on held-out test set
    - AC 6.4: Sweep N_l ∈ {10, 25, 50, 100, 250, 500}
    - AC 6.5: Use identical train/test splits across all models
    - AC 6.6: Report results compatible with existing benchmark_comparison.json
    
    Args:
        pretrain_checkpoint: Path to pretraining checkpoint
        config_path: Path to YAML configuration file
        output_dir: Directory to save results
        n_labeled_sweep: List of labeled sample counts (default: [10, 25, 50, 100, 250, 500])
        baselines: List of baseline models to run (default: ["fno", "geo_fno", "deeponet"])
        compatible_format: Whether to save in format compatible with existing benchmark_comparison.json
        
    Returns:
        Results dict mapping model name to {n_labeled: relative_l2_error}
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    if config["experiment"].get("device") is not None:
        device = torch.device(config["experiment"]["device"])
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(pretrain_checkpoint, config, device)
    
    # Create evaluator
    evaluator = DataEfficiencyEvaluator(
        config=config,
        output_dir=output_dir,
        device=device
    )
    
    # Get n_unlabeled from config or checkpoint
    pretraining_cfg = config.get("pretraining", {})
    n_unlabeled = pretraining_cfg.get("n_unlabeled", 1000)
    
    # Run comparison
    results = evaluator.run_comparison(
        pretrained_encoder=encoder,
        n_unlabeled=n_unlabeled,
        n_labeled_sweep=n_labeled_sweep,
        baselines_to_run=baselines
    )
    
    # Save results (AC 6.6: compatible format)
    if compatible_format:
        evaluator.save_results_compatible(results)
    else:
        evaluator.save_results(results)
    
    # Print summary
    evaluator.print_summary(results)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data efficiency evaluation for PI-JEPA vs baselines"
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        required=True,
        help="Path to pretraining checkpoint"
    )
    parser.add_argument(
        "--config",
        default="configs/darcy.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        default="outputs/data_efficiency",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-labeled",
        type=int,
        nargs="+",
        default=None,
        help="Labeled sample counts to evaluate (default: 10 25 50 100 250 500)"
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=None,
        help="Baseline models to run (default: fno geo_fno deeponet)"
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata in output JSON (not compatible with existing format)"
    )
    args = parser.parse_args()
    
    results = evaluate_data_efficiency(
        pretrain_checkpoint=args.pretrain_checkpoint,
        config_path=args.config,
        output_dir=args.output,
        n_labeled_sweep=args.n_labeled,
        baselines=args.baselines,
        compatible_format=not args.include_metadata
    )
    
    print("\nData efficiency evaluation complete.")
