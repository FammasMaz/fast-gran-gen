"""
Evaluation tools for assessing conditioning effectiveness in 3D voxel diffusion.

This module provides utilities for evaluating how well the conditional model
follows the specified conditioning inputs and generating diagnostic reports.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
import pandas as pd
from pathlib import Path
import h5py
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import warnings


class ConditioningEvaluator:
    """
    Evaluates conditioning effectiveness for voxel diffusion models.
    """
    
    def __init__(self, normalization_params_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            normalization_params_path: Path to conditioning normalization parameters
        """
        self.normalization_params = None
        self.feature_names = None
        self.means = None
        self.stds = None
        
        if normalization_params_path:
            self.load_normalization_params(normalization_params_path)
    
    def load_normalization_params(self, params_path: str):
        """Load normalization parameters."""
        with open(params_path, 'r') as f:
            self.normalization_params = json.load(f)
        
        self.feature_names = self.normalization_params['feature_names']
        self.means = np.array(self.normalization_params['means'])
        self.stds = np.array(self.normalization_params['stds'])
    
    def denormalize_conditioning(self, normalized_conditioning: np.ndarray) -> np.ndarray:
        """
        Denormalize conditioning features.
        
        Args:
            normalized_conditioning: Normalized conditioning values
            
        Returns:
            Denormalized conditioning values
        """
        if self.means is None or self.stds is None:
            return normalized_conditioning
        
        return normalized_conditioning * self.stds + self.means
    
    def compute_voxel_statistics(self, voxel_grids: np.ndarray) -> List[Dict[str, float]]:
        """
        Compute statistical features for voxel grids.
        
        Args:
            voxel_grids: Array of voxel grids (N, D, H, W)
            
        Returns:
            List of statistics dictionaries
        """
        from scipy import ndimage
        
        stats_list = []
        
        for voxel_grid in voxel_grids:
            binary_grid = (voxel_grid > 0.5).astype(np.uint8)
            
            stats = {}
            
            # Basic statistics
            stats['density'] = np.mean(binary_grid)
            stats['volume'] = np.sum(binary_grid)
            
            # Shape characteristics
            D, H, W = binary_grid.shape
            stats['aspect_ratio_dh'] = D / H if H > 0 else 0.0
            stats['aspect_ratio_dw'] = D / W if W > 0 else 0.0
            stats['aspect_ratio_hw'] = H / W if W > 0 else 0.0
            
            # Connected components
            labeled_grid, num_components = ndimage.label(binary_grid)
            stats['num_components'] = num_components
            
            if num_components > 0:
                component_sizes = np.bincount(labeled_grid.ravel())[1:]
                stats['largest_component_size'] = np.max(component_sizes)
                stats['largest_component_ratio'] = stats['largest_component_size'] / stats['volume'] if stats['volume'] > 0 else 0.0
            else:
                stats['largest_component_size'] = 0
                stats['largest_component_ratio'] = 0.0
            
            # Surface area approximation
            if stats['volume'] > 0:
                padded = np.pad(binary_grid, 1, mode='constant', constant_values=0)
                neighbors_sum = (
                    padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1] +
                    padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1] +
                    padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:]
                )
                surface_voxels = binary_grid & (neighbors_sum < 6)
                stats['surface_area_approx'] = np.sum(surface_voxels)
                stats['surface_to_volume_ratio'] = stats['surface_area_approx'] / stats['volume']
            else:
                stats['surface_area_approx'] = 0
                stats['surface_to_volume_ratio'] = 0.0
            
            # Bounding box
            if stats['volume'] > 0:
                coords = np.where(binary_grid)
                min_coords = [np.min(c) for c in coords]
                max_coords = [np.max(c) for c in coords]
                bbox_size = [max_coords[i] - min_coords[i] + 1 for i in range(3)]
                bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
                stats['bbox_volume'] = bbox_volume
                stats['bbox_fill_ratio'] = stats['volume'] / bbox_volume if bbox_volume > 0 else 0.0
            else:
                stats['bbox_volume'] = 0
                stats['bbox_fill_ratio'] = 0.0
            
            stats_list.append(stats)
        
        return stats_list
    
    def evaluate_conditioning_accuracy(
        self,
        target_conditioning: np.ndarray,
        generated_voxels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate how accurately the model follows conditioning inputs.
        
        Args:
            target_conditioning: Target conditioning values (N, num_features)
            generated_voxels: Generated voxel grids (N, D, H, W)
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if feature_names is None:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(target_conditioning.shape[1])]
        
        # Denormalize target conditioning if needed
        if self.means is not None:
            target_denorm = self.denormalize_conditioning(target_conditioning)
        else:
            target_denorm = target_conditioning
        
        # Compute actual statistics
        actual_stats_list = self.compute_voxel_statistics(generated_voxels)
        
        # Convert to array format
        actual_stats_array = np.array([[stats.get(name, 0.0) for name in feature_names] 
                                     for stats in actual_stats_list])
        
        # Compute metrics
        metrics = {}
        
        # Overall metrics
        mse = mean_squared_error(target_denorm, actual_stats_array)
        metrics['overall_mse'] = mse
        metrics['overall_rmse'] = np.sqrt(mse)
        
        # Per-feature metrics
        feature_metrics = {}
        for i, feature_name in enumerate(feature_names):
            target_values = target_denorm[:, i]
            actual_values = actual_stats_array[:, i]
            
            # Skip features with zero variance
            if np.var(target_values) > 1e-8:
                feature_mse = mean_squared_error(target_values, actual_values)
                feature_r2 = r2_score(target_values, actual_values)
                correlation, p_value = stats.pearsonr(target_values, actual_values)
                
                feature_metrics[feature_name] = {
                    'mse': feature_mse,
                    'rmse': np.sqrt(feature_mse),
                    'r2_score': feature_r2,
                    'correlation': correlation,
                    'correlation_p_value': p_value,
                    'mean_absolute_error': np.mean(np.abs(target_values - actual_values)),
                    'relative_error': np.mean(np.abs(target_values - actual_values) / (np.abs(target_values) + 1e-8))
                }
            else:
                feature_metrics[feature_name] = {
                    'mse': 0.0, 'rmse': 0.0, 'r2_score': 1.0,
                    'correlation': 1.0, 'correlation_p_value': 0.0,
                    'mean_absolute_error': 0.0, 'relative_error': 0.0
                }
        
        metrics['feature_metrics'] = feature_metrics
        
        return metrics
    
    def plot_conditioning_accuracy(
        self,
        target_conditioning: np.ndarray,
        generated_voxels: np.ndarray,
        output_path: str,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Create plots showing conditioning accuracy.
        
        Args:
            target_conditioning: Target conditioning values
            generated_voxels: Generated voxel grids
            output_path: Path to save plots
            feature_names: Names of features
            figsize: Figure size
        """
        if feature_names is None:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(target_conditioning.shape[1])]
        
        # Denormalize and compute actual stats
        if self.means is not None:
            target_denorm = self.denormalize_conditioning(target_conditioning)
        else:
            target_denorm = target_conditioning
        
        actual_stats_list = self.compute_voxel_statistics(generated_voxels)
        actual_stats_array = np.array([[stats.get(name, 0.0) for name in feature_names] 
                                     for stats in actual_stats_list])
        
        # Create subplots
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature_name in enumerate(feature_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            target_values = target_denorm[:, i]
            actual_values = actual_stats_array[:, i]
            
            # Scatter plot
            ax.scatter(target_values, actual_values, alpha=0.6, s=50)
            
            # Perfect accuracy line
            min_val = min(np.min(target_values), np.min(actual_values))
            max_val = max(np.max(target_values), np.max(actual_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect accuracy')
            
            # Compute and display R²
            if np.var(target_values) > 1e-8:
                r2 = r2_score(target_values, actual_values)
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_xlabel(f'Target {feature_name}')
            ax.set_ylabel(f'Actual {feature_name}')
            ax.set_title(f'{feature_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(
        self,
        generated_voxels: np.ndarray,
        output_path: str,
        compare_with: Optional[np.ndarray] = None,
        compare_label: str = "Reference",
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot distributions of features in generated voxels.
        
        Args:
            generated_voxels: Generated voxel grids
            output_path: Path to save plots
            compare_with: Optional reference voxels for comparison
            compare_label: Label for reference data
            feature_names: Names of features
            figsize: Figure size
        """
        if feature_names is None:
            feature_names = self.feature_names or ["density", "volume", "num_components"]
        
        # Compute statistics
        generated_stats = self.compute_voxel_statistics(generated_voxels)
        generated_df = pd.DataFrame(generated_stats)
        
        compare_df = None
        if compare_with is not None:
            compare_stats = self.compute_voxel_statistics(compare_with)
            compare_df = pd.DataFrame(compare_stats)
        
        # Create plots
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name not in generated_df.columns:
                continue
                
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot generated distribution
            generated_values = generated_df[feature_name]
            ax.hist(generated_values, bins=20, alpha=0.7, label='Generated', density=True)
            
            # Plot reference distribution if available
            if compare_df is not None and feature_name in compare_df.columns:
                compare_values = compare_df[feature_name]
                ax.hist(compare_values, bins=20, alpha=0.7, label=compare_label, density=True)
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_guidance_scale_sensitivity(
        self,
        sampler,
        target_conditioning: Dict[str, float],
        guidance_scales: List[float],
        num_samples: int = 5,
        output_dir: str = "guidance_sensitivity"
    ) -> Dict[str, Dict]:
        """
        Evaluate how different guidance scales affect generation quality.
        
        Args:
            sampler: Conditional sampler instance
            target_conditioning: Target conditioning specification
            guidance_scales: List of guidance scales to test
            num_samples: Number of samples per guidance scale
            output_dir: Directory to save results
            
        Returns:
            Dictionary of results for each guidance scale
        """
        from pathlib import Path
        import os
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for guidance_scale in guidance_scales:
            print(f"Evaluating guidance scale: {guidance_scale}")
            
            # Generate samples
            generated_voxels = sampler.generate_conditional(
                target_conditioning=target_conditioning,
                batch_size=num_samples,
                guidance_scale=guidance_scale
            )
            
            # Postprocess
            generated_voxels = sampler.postprocess_voxels(generated_voxels)
            
            # Compute target conditioning array
            target_array = np.array([[target_conditioning.get(name, 0.0) for name in self.feature_names]
                                   for _ in range(num_samples)])
            
            # Evaluate accuracy
            metrics = self.evaluate_conditioning_accuracy(
                target_array, generated_voxels, self.feature_names
            )
            
            # Save plots
            plot_path = os.path.join(output_dir, f"guidance_{guidance_scale:.1f}_accuracy.png")
            self.plot_conditioning_accuracy(
                target_array, generated_voxels, plot_path, self.feature_names
            )
            
            results[guidance_scale] = {
                'metrics': metrics,
                'generated_voxels': generated_voxels,
                'plot_path': plot_path
            }
        
        # Create comparison plot
        self._plot_guidance_comparison(results, output_dir)
        
        return results
    
    def _plot_guidance_comparison(self, results: Dict, output_dir: str):
        """Plot comparison of guidance scales."""
        guidance_scales = list(results.keys())
        overall_rmse = [results[gs]['metrics']['overall_rmse'] for gs in guidance_scales]
        
        plt.figure(figsize=(10, 6))
        plt.plot(guidance_scales, overall_rmse, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Guidance Scale')
        plt.ylabel('Overall RMSE')
        plt.title('Conditioning Accuracy vs Guidance Scale')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        best_idx = np.argmin(overall_rmse)
        best_scale = guidance_scales[best_idx]
        best_rmse = overall_rmse[best_idx]
        plt.annotate(f'Best: {best_scale} (RMSE={best_rmse:.3f})',
                    xy=(best_scale, best_rmse),
                    xytext=(best_scale + 1, best_rmse + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.savefig(os.path.join(output_dir, 'guidance_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(
        self,
        target_conditioning: np.ndarray,
        generated_voxels: np.ndarray,
        output_dir: str,
        report_name: str = "conditioning_evaluation_report"
    ):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            target_conditioning: Target conditioning values
            generated_voxels: Generated voxel grids
            output_dir: Directory to save report
            report_name: Base name for report files
        """
        from pathlib import Path
        import os
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Evaluate accuracy
        metrics = self.evaluate_conditioning_accuracy(
            target_conditioning, generated_voxels, self.feature_names
        )
        
        # Create plots
        accuracy_plot = os.path.join(output_dir, f"{report_name}_accuracy.png")
        self.plot_conditioning_accuracy(
            target_conditioning, generated_voxels, accuracy_plot, self.feature_names
        )
        
        distribution_plot = os.path.join(output_dir, f"{report_name}_distributions.png")
        self.plot_feature_distributions(
            generated_voxels, distribution_plot, feature_names=self.feature_names
        )
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"{report_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Generate summary
        summary = self._generate_summary(metrics)
        summary_file = os.path.join(output_dir, f"{report_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"Evaluation report generated in: {output_dir}")
        print(f"Summary: {summary_file}")
        print(f"Metrics: {metrics_file}")
        print(f"Plots: {accuracy_plot}, {distribution_plot}")
    
    def _generate_summary(self, metrics: Dict) -> str:
        """Generate text summary of evaluation results."""
        summary = "Conditioning Evaluation Summary\n"
        summary += "=" * 40 + "\n\n"
        
        summary += f"Overall Performance:\n"
        summary += f"  RMSE: {metrics['overall_rmse']:.4f}\n"
        summary += f"  MSE: {metrics['overall_mse']:.4f}\n\n"
        
        summary += "Feature-wise Performance:\n"
        for feature, feature_metrics in metrics['feature_metrics'].items():
            summary += f"  {feature}:\n"
            summary += f"    R² Score: {feature_metrics['r2_score']:.3f}\n"
            summary += f"    Correlation: {feature_metrics['correlation']:.3f}\n"
            summary += f"    RMSE: {feature_metrics['rmse']:.4f}\n"
            summary += f"    Relative Error: {feature_metrics['relative_error']:.3f}\n\n"
        
        # Identify best and worst performing features
        r2_scores = {name: metrics['feature_metrics'][name]['r2_score'] 
                    for name in metrics['feature_metrics']}
        best_feature = max(r2_scores, key=r2_scores.get)
        worst_feature = min(r2_scores, key=r2_scores.get)
        
        summary += f"Best controlled feature: {best_feature} (R² = {r2_scores[best_feature]:.3f})\n"
        summary += f"Worst controlled feature: {worst_feature} (R² = {r2_scores[worst_feature]:.3f})\n"
        
        return summary


def load_evaluation_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load evaluation data from HDF5 file.
    
    Args:
        data_path: Path to HDF5 file containing evaluation data
        
    Returns:
        Tuple of (target_conditioning, generated_voxels)
    """
    with h5py.File(data_path, 'r') as f:
        target_conditioning = f['target_conditioning'][:]
        generated_voxels = f['generated_voxels'][:]
    
    return target_conditioning, generated_voxels


def compare_models(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    normalization_params_path: str,
    output_dir: str
):
    """
    Compare conditioning performance across multiple models.
    
    Args:
        model_results: Dict mapping model names to (target, generated) tuples
        normalization_params_path: Path to normalization parameters
        output_dir: Directory to save comparison results
    """
    evaluator = ConditioningEvaluator(normalization_params_path)
    
    comparison_results = {}
    
    for model_name, (target, generated) in model_results.items():
        metrics = evaluator.evaluate_conditioning_accuracy(target, generated)
        comparison_results[model_name] = metrics
    
    # Create comparison plots and report
    # Implementation would depend on specific comparison needs
    
    return comparison_results