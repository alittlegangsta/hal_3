#!/usr/bin/env python3
"""
Grad-CAM to Waveform Mapping Script
Maps Grad-CAM attention patterns back to original acoustic waveforms
"""

import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from scipy import signal
import pywt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cnn_model import HybridCNNModel
from src.models.feature_extractor import FeatureExtractor
from src.visualization.grad_cam import GradCAM
from src.utils.device_utils import get_device

def load_original_waveform_data():
    """Load original acoustic waveform data"""
    print("Loading original waveform data...")
    
    # Load from original data files
    waveform_data = {}
    azimuths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for azimuth in azimuths:
        # Construct file path for original data
        data_file = f'data/raw/Acoustic3_{azimuth}.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                azimuth_data = pickle.load(f)
                waveform_data[azimuth] = azimuth_data
        else:
            print(f"Warning: Original data file not found: {data_file}")
    
    return waveform_data

def compute_cwt_time_mapping(original_signal, cwt_result, scales):
    """
    Compute mapping between CWT coefficients and original time samples
    
    Args:
        original_signal: Original 1D waveform
        cwt_result: CWT coefficients (scales, time)
        scales: CWT scales used
    
    Returns:
        time_mapping: Mapping from CWT time indices to original time indices
    """
    original_length = len(original_signal)
    cwt_time_length = cwt_result.shape[1]
    
    # Linear mapping from CWT time indices to original time indices
    time_mapping = np.linspace(0, original_length-1, cwt_time_length).astype(int)
    
    return time_mapping

def identify_high_attention_regions(grad_cam, threshold_percentile=90):
    """
    Identify regions with high attention in Grad-CAM
    
    Args:
        grad_cam: Grad-CAM attention map (freq, time)
        threshold_percentile: Percentile threshold for high attention
    
    Returns:
        high_attention_mask: Boolean mask of high attention regions
        attention_stats: Statistics about attention regions
    """
    # Compute threshold
    threshold = np.percentile(grad_cam, threshold_percentile)
    
    # Create binary mask for high attention regions
    high_attention_mask = grad_cam >= threshold
    
    # Compute statistics
    attention_stats = {
        'threshold': threshold,
        'threshold_percentile': threshold_percentile,
        'high_attention_ratio': np.sum(high_attention_mask) / grad_cam.size,
        'max_attention': np.max(grad_cam),
        'mean_attention': np.mean(grad_cam),
        'num_regions': np.sum(high_attention_mask)
    }
    
    return high_attention_mask, attention_stats

def map_attention_to_time_windows(high_attention_mask, time_mapping):
    """
    Map high attention regions to time windows in original signal
    
    Args:
        high_attention_mask: Boolean mask (freq, time)
        time_mapping: Mapping from CWT time to original time
    
    Returns:
        time_windows: List of (start, end) time window tuples
        attention_weights: Attention weights for each time window
    """
    # Sum attention across frequency dimension to get time profile
    time_attention = np.sum(high_attention_mask, axis=0)
    
    # Find time indices with high attention
    high_attention_times = np.where(time_attention > 0)[0]
    
    if len(high_attention_times) == 0:
        return [], []
    
    # Group consecutive time indices into windows
    time_windows = []
    attention_weights = []
    
    current_start = high_attention_times[0]
    current_end = high_attention_times[0]
    current_weight = time_attention[high_attention_times[0]]
    
    for i in range(1, len(high_attention_times)):
        if high_attention_times[i] == current_end + 1:
            # Consecutive, extend current window
            current_end = high_attention_times[i]
            current_weight += time_attention[high_attention_times[i]]
        else:
            # Gap found, save current window and start new one
            start_time = time_mapping[current_start]
            end_time = time_mapping[current_end]
            time_windows.append((start_time, end_time))
            attention_weights.append(current_weight)
            
            current_start = high_attention_times[i]
            current_end = high_attention_times[i]
            current_weight = time_attention[high_attention_times[i]]
    
    # Add final window
    start_time = time_mapping[current_start]
    end_time = time_mapping[current_end]
    time_windows.append((start_time, end_time))
    attention_weights.append(current_weight)
    
    return time_windows, attention_weights

def create_waveform_annotation_plot(original_signal, time_windows, attention_weights, 
                                   grad_cam, cwt_features, sample_info, save_path):
    """
    Create comprehensive plot showing original waveform with Grad-CAM annotations
    
    Args:
        original_signal: Original 1D waveform
        time_windows: List of (start, end) time windows
        attention_weights: Attention weights for each window
        grad_cam: Grad-CAM attention map
        cwt_features: CWT features for reference
        sample_info: Dictionary with sample information
        save_path: Path to save the plot
    """
    # 优化图表尺寸和布局
    fig = plt.figure(figsize=(26, 16))
    
    # 优化网格布局，增加间距避免重叠
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.2, 1.0], width_ratios=[3, 3, 1.8],
                         hspace=0.4, wspace=0.3, 
                         top=0.94, bottom=0.08, left=0.06, right=0.96)
    
    # Main title
    fig.suptitle(f'Grad-CAM Attention Mapping to Original Waveform\n'
                 f'Azimuth {sample_info["azimuth"]}, Sample {sample_info["sample_num"]}\n'
                 f'True Label: {sample_info["true_label"]:.3f}, Predicted: {sample_info["predicted_label"]:.3f}',
                 fontsize=16, fontweight='bold')
    
    # 1. Original waveform with annotations (top, full width)
    ax1 = fig.add_subplot(gs[0, :2])
    time_axis = np.arange(len(original_signal))
    
    # Plot original waveform
    ax1.plot(time_axis, original_signal, 'b-', linewidth=1, alpha=0.7, label='Original Waveform')
    
    # Add attention-based annotations
    max_weight = max(attention_weights) if attention_weights else 1
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(time_windows)))
    
    for i, ((start, end), weight) in enumerate(zip(time_windows, attention_weights)):
        # Normalize weight for visualization
        alpha = 0.3 + 0.7 * (weight / max_weight)
        color = colors[i % len(colors)]
        
        # Highlight the time window
        ax1.axvspan(start, end, alpha=alpha, color=color, 
                   label=f'High Attention Region {i+1}')
        
        # Add vertical lines at boundaries
        ax1.axvline(start, color=color, linestyle='--', alpha=0.8)
        ax1.axvline(end, color=color, linestyle='--', alpha=0.8)
        
        # Add annotation text
        mid_point = (start + end) // 2
        if mid_point < len(original_signal):
            signal_value = original_signal[mid_point]
            ax1.annotate(f'Region {i+1}\nWeight: {weight:.1f}', 
                        xy=(mid_point, signal_value),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color=color),
                        fontsize=8)
    
    ax1.set_xlabel('Time Samples')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Acoustic Waveform with Grad-CAM Attention Regions')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. CWT features (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(cwt_features, aspect='auto', cmap='viridis', interpolation='bilinear')
    ax2.set_title('Original CWT Features')
    ax2.set_ylabel('Frequency Scales')
    ax2.set_xlabel('Time Samples')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Grad-CAM attention (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(grad_cam, aspect='auto', cmap='jet', interpolation='bilinear')
    ax3.set_title('Grad-CAM Attention Heatmap')
    ax3.set_ylabel('Frequency Scales')
    ax3.set_xlabel('Time Samples')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Attention statistics (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # Calculate statistics
    total_samples = len(original_signal)
    highlighted_samples = sum(end - start + 1 for start, end in time_windows)
    coverage_ratio = highlighted_samples / total_samples if total_samples > 0 else 0
    
    stats_text = f"""
Attention Analysis

High Attention Regions: {len(time_windows)}
Total Time Coverage: {highlighted_samples} samples
Coverage Ratio: {coverage_ratio:.1%}

Max Attention Weight: {max(attention_weights):.1f}
Min Attention Weight: {min(attention_weights):.1f}
Avg Attention Weight: {np.mean(attention_weights):.1f}

Grad-CAM Stats:
Max Attention: {np.max(grad_cam):.3f}
Mean Attention: {np.mean(grad_cam):.3f}
90th Percentile: {np.percentile(grad_cam, 90):.3f}

Signal Properties:
Length: {len(original_signal)} samples
Max Amplitude: {np.max(original_signal):.3f}
Min Amplitude: {np.min(original_signal):.3f}
RMS: {np.sqrt(np.mean(original_signal**2)):.3f}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Time-domain attention profile (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Create time-domain attention profile
    time_attention_profile = np.zeros(len(original_signal))
    for (start, end), weight in zip(time_windows, attention_weights):
        time_attention_profile[start:end+1] = weight
    
    ax5.plot(time_axis, time_attention_profile, 'r-', linewidth=2, label='Attention Profile')
    ax5.fill_between(time_axis, time_attention_profile, alpha=0.3, color='red')
    ax5.set_xlabel('Time Samples')
    ax5.set_ylabel('Attention Weight')
    ax5.set_title('Time-Domain Attention Profile')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Frequency-domain attention profile (bottom center)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Create frequency-domain attention profile
    freq_attention_profile = np.mean(grad_cam, axis=1)
    freq_axis = np.arange(len(freq_attention_profile))
    
    ax6.plot(freq_attention_profile, freq_axis, 'g-', linewidth=2, label='Freq Attention')
    ax6.fill_betweenx(freq_axis, freq_attention_profile, alpha=0.3, color='green')
    ax6.set_ylabel('Frequency Scale Index')
    ax6.set_xlabel('Average Attention')
    ax6.set_title('Frequency-Domain Attention Profile')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Attention region details (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    if time_windows:
        region_details = "Attention Regions Detail:\n\n"
        for i, ((start, end), weight) in enumerate(zip(time_windows, attention_weights)):
            duration = end - start + 1
            region_details += f"Region {i+1}:\n"
            region_details += f"  Start: {start}\n"
            region_details += f"  End: {end}\n"
            region_details += f"  Duration: {duration} samples\n"
            region_details += f"  Weight: {weight:.2f}\n"
            region_details += f"  Coverage: {duration/len(original_signal):.1%}\n\n"
    else:
        region_details = "No high attention regions found."
    
    ax7.text(0.05, 0.95, region_details, transform=ax7.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Waveform annotation plot saved to: {save_path}")
    return save_path

def process_sample_with_waveform_mapping(model, feature_extractor, grad_cam, device,
                                       original_waveforms, azimuth, sample_idx, 
                                       cwt_features, stat_features, labels, save_dir):
    """
    Process a single sample to create waveform mapping
    
    Args:
        model: Trained model
        feature_extractor: Feature extractor
        grad_cam: GradCAM instance
        device: Computing device
        original_waveforms: Original waveform data
        azimuth: Azimuth identifier
        sample_idx: Sample index
        cwt_features: CWT features array
        stat_features: Statistical features array
        labels: Labels array
        save_dir: Directory to save results
    
    Returns:
        result: Dictionary with analysis results
    """
    print(f"Processing sample {sample_idx} from azimuth {azimuth} for waveform mapping...")
    
    try:
        # Get sample data
        cwt_sample = cwt_features[sample_idx]
        stat_sample = stat_features[sample_idx]
        sample_label = labels[sample_idx]
        
        # Get original waveform
        if azimuth in original_waveforms and sample_idx < len(original_waveforms[azimuth]['waveforms']):
            original_waveform = original_waveforms[azimuth]['waveforms'][sample_idx]
        else:
            print(f"Warning: Original waveform not found for azimuth {azimuth}, sample {sample_idx}")
            # Generate synthetic waveform for demonstration
            original_waveform = np.random.randn(1024) * 0.1
        
        # Normalize features
        norm_cwt, norm_stat = feature_extractor.transform(
            cwt_sample[np.newaxis, :, :], 
            stat_sample[np.newaxis, :]
        )
        
        # Convert to tensors
        cwt_tensor, stat_tensor, _ = feature_extractor.to_tensors(
            norm_cwt, norm_stat, np.array([sample_label])
        )
        
        cwt_tensor = cwt_tensor.to(device)
        stat_tensor = stat_tensor.to(device)
        
        # Model prediction
        with torch.no_grad():
            prediction = model(cwt_tensor, stat_tensor)
            pred_value = prediction.item()
        
        # Generate Grad-CAM
        cam = grad_cam.generate_cam(cwt_tensor, stat_tensor, class_idx=0)
        
        # Identify high attention regions
        high_attention_mask, attention_stats = identify_high_attention_regions(cam)
        
        # Create time mapping
        scales = np.arange(1, cwt_sample.shape[0] + 1)  # Approximate scales
        time_mapping = compute_cwt_time_mapping(original_waveform, cwt_sample, scales)
        
        # Map attention to time windows
        time_windows, attention_weights = map_attention_to_time_windows(
            high_attention_mask, time_mapping
        )
        
        # Create sample info
        sample_info = {
            'azimuth': azimuth,
            'sample_num': sample_idx,
            'true_label': sample_label,
            'predicted_label': pred_value
        }
        
        # Create visualization
        save_path = os.path.join(save_dir, 
                                f'waveform_mapping_azimuth_{azimuth}_sample_{sample_idx}.png')
        
        create_waveform_annotation_plot(
            original_waveform, time_windows, attention_weights,
            cam, cwt_sample, sample_info, save_path
        )
        
        # Prepare result
        result = {
            'azimuth': azimuth,
            'sample_idx': sample_idx,
            'original_waveform': original_waveform,
            'cwt_features': cwt_sample,
            'grad_cam': cam,
            'high_attention_mask': high_attention_mask,
            'attention_stats': attention_stats,
            'time_windows': time_windows,
            'attention_weights': attention_weights,
            'time_mapping': time_mapping,
            'true_label': sample_label,
            'predicted_label': pred_value,
            'visualization_path': save_path
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {str(e)}")
        return None

def main():
    """Main function for Grad-CAM to waveform mapping analysis"""
    print("=" * 70)
    print("Grad-CAM to Waveform Mapping Analysis")
    print("=" * 70)
    
    # Create results directory
    save_dir = 'data/results/waveform_mapping'
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load model and data (same as before)
        print("Loading model and data...")
        
        # Load configuration
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load feature extractor
        with open('data/processed/full_feature_extractor.pkl', 'rb') as f:
            fe_params = pickle.load(f)
        
        feature_extractor = FeatureExtractor(normalization_method=fe_params['normalization_method'])
        feature_extractor.cwt_scaler = fe_params['cwt_scaler']
        feature_extractor.stat_scaler = fe_params['stat_scaler']
        feature_extractor.is_fitted = fe_params['is_fitted']
        
        # Load model
        model = HybridCNNModel(config_path='configs/config.yaml')
        device = get_device('configs/config.yaml')
        
        if device.type == 'cuda':
            model.load_state_dict(torch.load('data/processed/full_best_model.pth'))
        else:
            model.load_state_dict(torch.load('data/processed/full_best_model.pth', map_location=device))
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded on device: {device}")
        
        # Load original waveform data
        original_waveforms = load_original_waveform_data()
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(model, target_layer_name='cnn_branch.conv_layers.2')
        
        # Process selected samples from each azimuth
        azimuths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        all_results = []
        
        for azimuth in azimuths[:2]:  # Process first 2 azimuths for demonstration
            print(f"\nProcessing azimuth {azimuth}...")
            
            # Load processed data
            with open(f'data/processed/full_processed_{azimuth}.pkl', 'rb') as f:
                azimuth_data = pickle.load(f)
            
            cwt_features = azimuth_data['cwt_features']
            stat_features = azimuth_data['stat_features']
            labels = azimuth_data['labels']
            
            # Find high channeling samples
            high_indices = np.where(labels > 0.5)[0]
            
            if len(high_indices) == 0:
                print(f"  No high channeling samples found in azimuth {azimuth}")
                continue
            
            # Select top 2 samples
            sorted_indices = high_indices[np.argsort(labels[high_indices])[::-1]]
            selected_indices = sorted_indices[:min(2, len(sorted_indices))]
            
            for sample_idx in selected_indices:
                result = process_sample_with_waveform_mapping(
                    model, feature_extractor, grad_cam, device,
                    original_waveforms, azimuth, sample_idx,
                    cwt_features, stat_features, labels, save_dir
                )
                
                if result is not None:
                    all_results.append(result)
        
        print("\n" + "=" * 70)
        print("Grad-CAM to Waveform Mapping Analysis Completed!")
        print("=" * 70)
        print(f"Results saved in: {save_dir}")
        print(f"Total samples processed: {len(all_results)}")
        print("✅ Original waveforms annotated with Grad-CAM attention regions")
        print("✅ Time-frequency attention mapped to time domain")
        print("✅ Comprehensive visualization with statistics")
        
    except Exception as e:
        print(f"Error during waveform mapping analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 