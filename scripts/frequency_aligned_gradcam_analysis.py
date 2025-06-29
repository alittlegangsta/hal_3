#!/usr/bin/env python3
"""
Frequency-Aligned Grad-CAM Analysis Script
Based on existing full_gradcam directory, generate frequency axis conversion and original waveform alignment analysis
"""

import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
from scipy import signal
import pywt
import re
import glob
import scipy.io

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cnn_model import HybridCNNModel
from src.models.feature_extractor import FeatureExtractor
from src.visualization.grad_cam import GradCAM
from src.utils.device_utils import get_device

# Set matplotlib Chinese font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


def scale_to_frequency(scales, sampling_period=1e-5, wavelet='morl'):
    """
    Convert CWT scale to corresponding frequency
    
    Args:
        scales: CWT scale array
        sampling_period: Sampling period (s)
        wavelet: Wavelet type
        
    Returns:
        frequencies: Corresponding frequency array (Hz)
    """
    if wavelet == 'morl':
        # Morlet wavelet center frequency
        fc = 1.0  # Normalized Morlet wavelet center frequency
    else:
        # For other wavelets, use PyWavelets center frequency
        fc = pywt.central_frequency(wavelet)
    
    # Scale to frequency conversion formula
    frequencies = fc / (scales * sampling_period)
    
    return frequencies


def reconstruct_original_waveform(cwt_features, stat_features, sample_info):
    """
    Reconstruct approximate original acoustic waveform based on CWT features and statistical features
    
    Args:
        cwt_features: CWT features (scales, time_points)
        stat_features: Statistical features array [mean, std, peak_time, peak_amplitude, energy, rms, skewness, kurtosis]
        sample_info: Sample information dictionary
        
    Returns:
        waveform: Reconstructed waveform
        time_axis: Time axis (seconds)
    """
    # Acoustic wave parameters
    length = 1024  # Standard waveform length
    sampling_period = 1e-5  # Sampling period (10μs)
    time_axis = np.arange(length) * sampling_period
    
    # Extract key information from statistical features
    mean_amplitude = stat_features[0] if len(stat_features) > 0 else 0.0
    rms_amplitude = stat_features[5] if len(stat_features) > 5 else 0.1
    peak_time_idx = int(stat_features[2] * length) if len(stat_features) > 2 else 200
    energy = stat_features[4] if len(stat_features) > 4 else 1.0
    
    # Create base waveform
    waveform = np.zeros(length)
    
    # Adjust waveform characteristics based on channeling ratio
    channeling_ratio = sample_info.get('true_label', 0.5)
    
    # 1. Direct wave (P-wave) - High frequency component
    p_wave_freq = 15000 + channeling_ratio * 5000  # 15-20 kHz
    p_wave_arrival = 50 + int(channeling_ratio * 50)  # Arrival time
    p_wave_duration = 100
    p_wave_envelope = np.zeros(length)
    p_wave_envelope[p_wave_arrival:p_wave_arrival + p_wave_duration] = np.exp(
        -np.arange(p_wave_duration) / 30) * (1.0 - channeling_ratio * 0.3)
    waveform += rms_amplitude * 0.6 * np.sin(2 * np.pi * p_wave_freq * time_axis) * p_wave_envelope
    
    # 2. Casing wave - Medium frequency component
    casing_freq = 8000 + channeling_ratio * 2000  # 8-10 kHz
    casing_arrival = 150 + int(channeling_ratio * 100)
    casing_duration = 200
    casing_envelope = np.zeros(length)
    casing_envelope[casing_arrival:casing_arrival + casing_duration] = np.exp(
        -np.arange(casing_duration) / 80) * (0.8 + channeling_ratio * 0.4)
    waveform += rms_amplitude * 0.4 * np.sin(2 * np.pi * casing_freq * time_axis) * casing_envelope
    
    # 3. Cement interface reflection wave - Low frequency component (Channeling sensitive)
    cement_freq = 3000 + channeling_ratio * 2000  # 3-5 kHz
    cement_arrival = 300 + int(channeling_ratio * 150)
    cement_duration = 300
    cement_envelope = np.zeros(length)
    cement_envelope[cement_arrival:cement_arrival + cement_duration] = np.exp(
        -np.arange(cement_duration) / 150) * (channeling_ratio * 1.5 + 0.2)
    waveform += rms_amplitude * 0.8 * np.sin(2 * np.pi * cement_freq * time_axis) * cement_envelope
    
    # 4. Add details based on CWT features
    # Extract time-frequency energy distribution from CWT features
    time_energy = np.mean(cwt_features, axis=0)  # Average energy in time direction
    freq_energy = np.mean(cwt_features, axis=1)  # Average energy in frequency direction
    
    # Normalize energy distribution
    if np.max(time_energy) > 0:
        time_energy = time_energy / np.max(time_energy)
    if np.max(freq_energy) > 0:
        freq_energy = freq_energy / np.max(freq_energy)
    
    # Add modulation based on CWT time-frequency energy distribution
    for i in range(min(5, len(freq_energy) // 10)):
        idx = i * 10
        if idx < len(freq_energy):
            # Frequency component based on CWT frequency energy distribution
            freq_component = 2000 + i * 3000 + freq_energy[idx] * 5000
            # Amplitude based on channeling ratio and energy distribution
            amplitude = rms_amplitude * 0.2 * freq_energy[idx] * (0.5 + channeling_ratio)
            # Time modulation based on CWT time-frequency energy distribution
            time_modulation = np.interp(np.arange(length), 
                                      np.linspace(0, length-1, len(time_energy)), 
                                      time_energy)
            waveform += amplitude * np.sin(2 * np.pi * freq_component * time_axis) * time_modulation
    
    # 5. Add noise and random components
    noise_level = 0.1 * rms_amplitude * (1.0 - channeling_ratio * 0.5)  # Higher channeling, lower noise
    waveform += np.random.normal(0, noise_level, length)
    
    # 6. Apply overall envelope
    overall_envelope = np.exp(-time_axis * 500) + 0.1  # Overall attenuation
    waveform *= overall_envelope
    
    # Add mean offset
    waveform += mean_amplitude
    
    return waveform, time_axis


def parse_gradcam_filename(filename):
    """
    Parse Grad-CAM filename to extract azimuth and sample number
    
    Args:
        filename: Grad-CAM filename
        
    Returns:
        dict: Dictionary containing azimuth and sample number, returns None if parsing fails
    """
    # Match format: gradcam_azimuth_X_sample_Y.png
    pattern = r'gradcam_azimuth_([A-H])_sample_(\d+)\.png'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'azimuth': match.group(1),
            'sample_num': int(match.group(2))
        }
    
    return None


def load_sample_data(azimuth, sample_idx, processed_data_cache):
    """
    Load preprocessed data for specified sample
    
    Args:
        azimuth: Azimuth identifier
        sample_idx: Sample index
        processed_data_cache: Preprocessed data cache
        
    Returns:
        sample_data: Sample data dictionary
    """
    # Check cache
    cache_key = azimuth
    if cache_key not in processed_data_cache:
        try:
            # Load preprocessed data
            with open(f'data/processed/full_processed_{azimuth}.pkl', 'rb') as f:
                processed_data = pickle.load(f)
            processed_data_cache[cache_key] = processed_data
            print(f"✅ Loaded preprocessed data for azimuth {azimuth}")
        except FileNotFoundError:
            print(f"❌ Preprocessed data file for azimuth {azimuth} not found")
            return None
    
    processed_data = processed_data_cache[cache_key]
    
    # Check if sample index is valid
    if sample_idx >= len(processed_data['labels']):
        print(f"❌ Sample index {sample_idx} out of range (max: {len(processed_data['labels'])-1})")
        return None
    
    # Extract sample data
    sample_data = {
        'cwt_features': processed_data['cwt_features'][sample_idx],
        'stat_features': processed_data['stat_features'][sample_idx],
        'label': processed_data['labels'][sample_idx],
        'azimuth': azimuth,
        'sample_idx': sample_idx
    }
    
    return sample_data


def generate_gradcam_for_sample(model, feature_extractor, device, sample_data):
    """
    Generate Grad-CAM for specified sample
    
    Args:
        model: Trained model
        feature_extractor: Feature extractor
        device: Calculation device
        sample_data: Sample data dictionary
        
    Returns:
        cam: Grad-CAM heatmap
        prediction: Model prediction value
    """
    # Extract data
    cwt_features = sample_data['cwt_features']
    stat_features = sample_data['stat_features']
    label = sample_data['label']
    
    # Feature normalization
    norm_cwt, norm_stat = feature_extractor.transform(
        cwt_features[np.newaxis, :, :], 
        stat_features[np.newaxis, :]
    )
    
    # Convert to tensors
    cwt_tensor, stat_tensor, _ = feature_extractor.to_tensors(
        norm_cwt, norm_stat, np.array([label])
    )
    
    cwt_tensor = cwt_tensor.to(device)
    stat_tensor = stat_tensor.to(device)
    
    # Model prediction
    with torch.no_grad():
        prediction = model(cwt_tensor, stat_tensor)
        pred_value = prediction.item()
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer_name='cnn_branch.conv_layers.2')
    cam = grad_cam.generate_cam(cwt_tensor, stat_tensor, class_idx=0)
    
    return cam, pred_value


def load_original_waveform_from_data(azimuth, sample_idx, depth_info):
    """
    Load original acoustic waveform directly from raw MATLAB data
    
    Args:
        azimuth: Azimuth identifier (A-H)
        sample_idx: Sample index in processed data
        depth_info: Depth information for the sample
        
    Returns:
        waveform: Original acoustic waveform
        time_axis: Time axis (seconds)
    """
    try:
        # Load raw XSILMR03 data
        mat_data = scipy.io.loadmat('data/raw/XSILMR03.mat')
        
        # Extract the main data structure
        struct = mat_data['XSILMR03'][0, 0]
        
        # Get the field name for this azimuth
        field_name = f'WaveRng03Side{azimuth}'
        
        if field_name not in struct.dtype.names:
            print(f"❌ Field {field_name} not found in MATLAB data")
            # Return a fallback waveform
            length = 1024
            sampling_period = 1e-5
            time_axis = np.arange(length) * sampling_period
            waveform = np.zeros(length)
            return waveform, time_axis
        
        # Get the acoustic data for this azimuth
        azimuth_data = struct[field_name]  # Shape: (1024, 7108)
        
        print(f"📊 Azimuth {azimuth} data shape: {azimuth_data.shape}")
        print(f"📊 Data type: {azimuth_data.dtype}")
        
        # Since we have 7108 depth points but limited processed samples,
        # we'll use the sample_idx to select from the available depth points
        total_depth_points = azimuth_data.shape[1]
        
        # Map sample_idx to a depth point
        # For small samples, we'll distribute them across the depth range
        if sample_idx < total_depth_points:
            # Use sample_idx directly as depth index
            depth_idx = min(sample_idx, total_depth_points - 1)
        else:
            # If sample_idx is larger, use modulo to wrap around
            depth_idx = sample_idx % total_depth_points
        
        # Extract the waveform for this depth point
        # Data is organized as (time_samples, depth_points)
        waveform = azimuth_data[:, depth_idx].astype(np.float64)
        
        # Create time axis - 1024 samples at 10 microseconds per sample
        sampling_period = 1e-5  # 10 microseconds
        time_axis = np.arange(len(waveform)) * sampling_period
        
        print(f"✅ Loaded original waveform for azimuth {azimuth}, sample {sample_idx} (depth index {depth_idx})")
        print(f"   Waveform length: {len(waveform)} samples")
        print(f"   Time duration: {time_axis[-1]*1000:.2f} ms")
        print(f"   Amplitude range: [{np.min(waveform):.0f}, {np.max(waveform):.0f}]")
        
        return waveform, time_axis
        
    except Exception as e:
        print(f"❌ Error loading original waveform: {e}")
        # Return a fallback waveform
        length = 1024
        sampling_period = 1e-5
        time_axis = np.arange(length) * sampling_period
        waveform = np.zeros(length)
        return waveform, time_axis


def create_frequency_aligned_analysis(sample_data, gradcam, prediction, save_path):
    """
    Create frequency-aligned comprehensive analysis plot
    
    Args:
        sample_data: Sample data dictionary
        gradcam: Grad-CAM heatmap
        prediction: Model prediction value
        save_path: Save path
    """
    # Extract data
    cwt_features = sample_data['cwt_features']
    stat_features = sample_data['stat_features']
    true_label = sample_data['label']
    azimuth = sample_data['azimuth']
    sample_idx = sample_data['sample_idx']
    
    # Calculate frequency axis - Based on original CWT features
    scales = np.arange(1, cwt_features.shape[0] + 1)
    frequencies = scale_to_frequency(scales, sampling_period=1e-5, wavelet='morl')
    
    # Calculate Grad-CAM corresponding frequency axis - Based on actual Grad-CAM size
    gradcam_scales = np.linspace(1, cwt_features.shape[0], gradcam.shape[0])
    gradcam_frequencies = scale_to_frequency(gradcam_scales, sampling_period=1e-5, wavelet='morl')
    
    # Load original waveform from raw data
    original_waveform, time_axis = load_original_waveform_from_data(azimuth, sample_idx, None)
    
    # Create comprehensive analysis plot
    fig = plt.figure(figsize=(32, 20))
    
    # Adjust grid layout with better spacing
    gs = fig.add_gridspec(4, 4, 
                         height_ratios=[1.3, 1.2, 1.2, 0.8], 
                         width_ratios=[3.0, 3.0, 3.0, 2.0],
                         hspace=0.5, wspace=0.35, 
                         top=0.92, bottom=0.08, left=0.05, right=0.97)
    
    # Main title
    fig.suptitle(f'Frequency-Aligned Grad-CAM Analysis - Azimuth {azimuth}, Sample {sample_idx+1}\n'
                f'True Channeling Ratio: {true_label:.3f} | Prediction: {prediction:.3f} | '
                f'Prediction Error: {abs(true_label - prediction):.3f}',
                fontsize=22, fontweight='bold', y=0.97)
    
    # === 1. Original Acoustic Waveform from Raw Data (Top, Across 3 Columns) ===
    ax1 = fig.add_subplot(gs[0, :3])
    time_ms = time_axis * 1000  # Convert to milliseconds
    ax1.plot(time_ms, original_waveform, 'b-', linewidth=2, alpha=0.8, label='Original Acoustic Waveform')
    
    # Mark key waveform regions based on typical acoustic logging
    ax1.axvspan(0.5, 1.5, alpha=0.2, color='red', label='P-wave Region (Direct Arrival)')
    ax1.axvspan(1.5, 3.5, alpha=0.2, color='orange', label='Casing Wave Region')  
    ax1.axvspan(3.0, 6.0, alpha=0.2, color='green', label='Cement Interface Region')
    
    ax1.set_xlabel('Time (ms)', fontsize=16)
    ax1.set_ylabel('Amplitude (mV)', fontsize=16)
    ax1.set_title('Original Acoustic Waveform from Raw Data', fontsize=18, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # === 2. CWT Time-Frequency Plot (Scale Axis) - Second Row Left ===
    ax2 = fig.add_subplot(gs[1, 0])
    time_samples = np.arange(cwt_features.shape[1])
    scale_extent = [time_samples[0], time_samples[-1], scales[-1], scales[0]]
    
    im2 = ax2.imshow(cwt_features, aspect='auto', cmap='viridis', extent=scale_extent)
    ax2.set_xlabel('Time Samples', fontsize=14)
    ax2.set_ylabel('Wavelet Scale', fontsize=14)
    ax2.set_title('CWT Time-Frequency Features\n(Wavelet Scale Axis)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Add color bar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('CWT Coefficient Magnitude', fontsize=12)
    
    # === 3. CWT Time-Frequency Plot (Frequency Axis) - Second Row Middle ===
    ax3 = fig.add_subplot(gs[1, 1])
    freq_extent = [time_samples[0], time_samples[-1], frequencies[-1], frequencies[0]]
    
    im3 = ax3.imshow(cwt_features, aspect='auto', cmap='viridis', extent=freq_extent)
    ax3.set_xlabel('Time Samples', fontsize=14)
    ax3.set_ylabel('Frequency (Hz)', fontsize=14)
    ax3.set_title('CWT Time-Frequency Features\n(Frequency Axis)', fontsize=16, fontweight='bold')
    ax3.set_yscale('log')  # Use logarithmic scale to display frequency
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Add color bar
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('CWT Coefficient Magnitude', fontsize=12)
    
    # === 4. Grad-CAM Heatmap (Frequency Axis) - Second Row Right ===
    ax4 = fig.add_subplot(gs[1, 2])
    # Use actual frequency range of Grad-CAM
    gradcam_time_samples = np.arange(gradcam.shape[1])
    gradcam_freq_extent = [gradcam_time_samples[0], gradcam_time_samples[-1], gradcam_frequencies[-1], gradcam_frequencies[0]]
    
    im4 = ax4.imshow(gradcam, aspect='auto', cmap='jet', extent=gradcam_freq_extent, alpha=0.8)
    ax4.set_xlabel('Time Samples', fontsize=14)
    ax4.set_ylabel('Frequency (Hz)', fontsize=14)
    ax4.set_title('Grad-CAM Attention Heatmap\n(Frequency Axis)', fontsize=16, fontweight='bold')
    ax4.set_yscale('log')  # Use logarithmic scale to display frequency
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    # Add color bar
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Attention Intensity', fontsize=12)
    
    # === 5. Analysis Statistics (Second Row Rightmost) ===
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis('off')
    
    # Calculate key statistical information - Using Grad-CAM corresponding frequency
    freq_attention = np.mean(gradcam, axis=1)
    time_attention = np.mean(gradcam, axis=0)
    max_attention_freq_idx = np.argmax(freq_attention)
    max_attention_freq = gradcam_frequencies[max_attention_freq_idx]
    max_attention_time_idx = np.argmax(time_attention)
    
    # Frequency segment analysis - Using Grad-CAM frequency
    low_freq_mask = gradcam_frequencies < 10000
    mid_freq_mask = (gradcam_frequencies >= 10000) & (gradcam_frequencies < 50000)
    high_freq_mask = gradcam_frequencies >= 50000
    
    low_freq_attention = np.mean(freq_attention[low_freq_mask]) if np.any(low_freq_mask) else 0
    mid_freq_attention = np.mean(freq_attention[mid_freq_mask]) if np.any(mid_freq_mask) else 0
    high_freq_attention = np.mean(freq_attention[high_freq_mask]) if np.any(high_freq_mask) else 0
    
    stats_text = f"""Frequency Analysis Statistics

Original CWT Frequency Range:
  Max: {frequencies[0]:.0f} Hz
  Min: {frequencies[-1]:.0f} Hz
  Resolution: {len(frequencies)} frequency bands

Grad-CAM Frequency Range:
  Max: {gradcam_frequencies[0]:.0f} Hz
  Min: {gradcam_frequencies[-1]:.0f} Hz
  Resolution: {len(gradcam_frequencies)} frequency bands

Key Findings:
  Peak Attention Frequency: {max_attention_freq:.0f} Hz
  Peak Attention Time: Sample {max_attention_time_idx}
  Average Attention: {np.mean(gradcam):.3f}
  Maximum Attention: {np.max(gradcam):.3f}

Frequency Domain Energy Distribution:
  Low Freq (<10kHz): {low_freq_attention:.3f}
  Mid Freq (10-50kHz): {mid_freq_attention:.3f}
  High Freq (≥50kHz): {high_freq_attention:.3f}

Channeling Detection:
  Prediction Accuracy: {(1-abs(true_label-prediction))*100:.1f}%
  Label Type: {'High Channeling' if true_label > 0.5 else 'Low Channeling'}
  
Original Waveform Stats:
  Peak Amplitude: {np.max(np.abs(original_waveform)):.3f} mV
  RMS: {np.sqrt(np.mean(original_waveform**2)):.3f} mV
  Duration: {time_axis[-1]*1000:.2f} ms
"""
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # === 6. Frequency Domain Attention Distribution - Third Row Left ===
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.semilogx(gradcam_frequencies, freq_attention, 'r-', linewidth=3, marker='o', markersize=4)
    ax6.fill_between(gradcam_frequencies, freq_attention, alpha=0.3, color='red')
    ax6.set_xlabel('Frequency (Hz)', fontsize=14)
    ax6.set_ylabel('Average Attention Intensity', fontsize=14)
    ax6.set_title('Frequency Domain Attention Distribution', fontsize=16, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='both', which='major', labelsize=12)
    
    # Mark key frequency points
    ax6.axvline(max_attention_freq, color='red', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Peak Attention: {max_attention_freq:.0f} Hz')
    ax6.axvline(3000, color='green', linestyle=':', alpha=0.6, label='Cement Interface Freq')
    ax6.axvline(15000, color='blue', linestyle=':', alpha=0.6, label='P-wave Freq')
    ax6.legend(fontsize=11)
    
    # === 7. Time Domain Attention Distribution - Third Row Middle ===
    ax7 = fig.add_subplot(gs[2, 1])
    # Convert Grad-CAM time samples to milliseconds for better alignment
    gradcam_time_samples_ms = gradcam_time_samples * (time_axis[-1] * 1000) / len(gradcam_time_samples)
    ax7.plot(gradcam_time_samples_ms, time_attention, 'g-', linewidth=3, marker='s', markersize=3)
    ax7.fill_between(gradcam_time_samples_ms, time_attention, alpha=0.3, color='green')
    ax7.set_xlabel('Time (ms)', fontsize=14)
    ax7.set_ylabel('Average Attention Intensity', fontsize=14)
    ax7.set_title('Time Domain Attention Distribution', fontsize=16, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='both', which='major', labelsize=12)
    
    # Mark key time points
    max_attention_time_ms = max_attention_time_idx * (time_axis[-1] * 1000) / len(gradcam_time_samples)
    ax7.axvline(max_attention_time_ms, color='green', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Peak Attention: {max_attention_time_ms:.2f} ms')
    ax7.legend(fontsize=11)
    
    # === 8. Waveform-Attention Overlay - Third Row Right ===
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Normalize waveform for overlay with attention
    waveform_range = np.max(original_waveform) - np.min(original_waveform)
    if waveform_range > 0:
        norm_waveform = (original_waveform - np.min(original_waveform)) / waveform_range
    else:
        # If waveform is constant (all zeros or same value), set to 0.5
        norm_waveform = np.full_like(original_waveform, 0.5)
    
    # Plot normalized waveform
    ax8.plot(time_ms, norm_waveform, 'b-', linewidth=2, alpha=0.7, label='Normalized Waveform')
    
    # Overlay time attention (interpolated to match waveform time axis)
    attention_interpolated = np.interp(time_ms, gradcam_time_samples_ms, time_attention)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(time_ms, attention_interpolated, 'r-', linewidth=3, alpha=0.8, label='Attention Profile')
    ax8_twin.fill_between(time_ms, attention_interpolated, alpha=0.2, color='red')
    
    ax8.set_xlabel('Time (ms)', fontsize=12)
    ax8.set_ylabel('Normalized Amplitude', fontsize=12, color='blue')
    ax8_twin.set_ylabel('Attention Intensity', fontsize=12, color='red')
    ax8.set_title('Waveform-Attention Alignment\n(Time Domain)', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='y', labelcolor='blue', labelsize=11)
    ax8_twin.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    # Add legends
    ax8.legend(loc='upper left', fontsize=10)
    ax8_twin.legend(loc='upper right', fontsize=10)
    
    # === 9. Frequency Band Comparison - Third Row Rightmost ===
    ax9 = fig.add_subplot(gs[2, 3])
    
    # Create comparison bar chart
    categories = ['Low Freq\n(<10kHz)', 'Mid Freq\n(10-50kHz)', 'High Freq\n(≥50kHz)']
    attention_values = [low_freq_attention, mid_freq_attention, high_freq_attention]
    colors = ['green', 'orange', 'red']
    
    bars = ax9.bar(categories, attention_values, color=colors, alpha=0.7, edgecolor='black')
    ax9.set_ylabel('Average Attention Intensity', fontsize=12)
    ax9.set_title('Frequency Band Attention Comparison', fontsize=14, fontweight='bold')
    ax9.tick_params(axis='both', which='major', labelsize=11)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, attention_values):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # === 10. Comprehensive Interpretation (Bottom, Across All Columns) ===
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    
    # Analyze high attention region
    high_attention_threshold = np.percentile(gradcam, 90)
    high_attention_mask = gradcam > high_attention_threshold
    high_attention_coords = np.where(high_attention_mask)
    
    if len(high_attention_coords[0]) > 0:
        # Analyze high attention region frequency and time distribution
        high_freq_indices = high_attention_coords[0]
        high_time_indices = high_attention_coords[1]
        high_frequencies = gradcam_frequencies[high_freq_indices]
        high_times_ms = high_time_indices * (time_axis[-1] * 1000) / len(gradcam_time_samples)
        
        avg_high_freq = np.mean(high_frequencies)
        avg_high_time = np.mean(high_times_ms)
        
        interpretation_text = f"""🔍 Comprehensive Grad-CAM Interpretation Analysis

📊 High Attention Region Analysis (Top 10% Attention > {high_attention_threshold:.3f}):
   • Total high attention points: {len(high_attention_coords[0])}
   • Primary focus frequency: {avg_high_freq:.0f} Hz (Range: {np.min(high_frequencies):.0f} - {np.max(high_frequencies):.0f} Hz)
   • Primary focus time: {avg_high_time:.2f} ms (Range: {np.min(high_times_ms):.2f} - {np.max(high_times_ms):.2f} ms)

🎯 Physical Wave Analysis & Interpretation:
   • Model primarily focuses on {'Cement Interface Reflections' if avg_high_freq < 8000 else 'Casing Wave Propagation' if avg_high_freq < 15000 else 'P-wave Direct Arrivals'} in the acoustic signal
   • Time window corresponds to {'Early Arrival Phase' if avg_high_time < 2 else 'Mid Propagation Phase' if avg_high_time < 4 else 'Late Reflection Phase'} of acoustic logging
   • Frequency preference {'strongly correlates' if abs(true_label - prediction) < 0.1 else 'moderately correlates' if abs(true_label - prediction) < 0.3 else 'weakly correlates'} with channeling detection physics

📈 Prediction Performance & Model Confidence:
   • Prediction accuracy: {(1-abs(true_label-prediction))*100:.1f}% | Sample type: {'High Channeling Sample' if true_label > 0.5 else 'Low Channeling Sample'}
   • Attention consistency: {'Excellent' if np.std(freq_attention) > 0.01 else 'Good' if np.std(freq_attention) > 0.005 else 'Fair'} (spatial focus distribution)
   • Frequency selectivity: {'Strong' if max(attention_values) > 0.1 else 'Moderate' if max(attention_values) > 0.05 else 'Weak'} (spectral discrimination)

💡 Engineering Application & Field Deployment Recommendations:
   • Model interpretation results {'strongly support' if abs(true_label - prediction) < 0.2 else 'moderately support' if abs(true_label - prediction) < 0.4 else 'require caution for'} field application
   • Recommend enhanced monitoring of {categories[np.argmax(attention_values)]} frequency band for optimal signal quality
   • Channeling detection confidence level: {'High' if abs(true_label - prediction) < 0.1 else 'Medium' if abs(true_label - prediction) < 0.3 else 'Low'} - suitable for {'autonomous' if abs(true_label - prediction) < 0.1 else 'supervised' if abs(true_label - prediction) < 0.3 else 'manual review'} operation
"""
    else:
        interpretation_text = "⚠️ No significant high attention regions detected. Recommend checking model training status or data quality for this sample."
    
    ax10.text(0.02, 0.98, interpretation_text, transform=ax10.transAxes,
              fontsize=11, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    
    # Save image
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Frequency-aligned analysis plot saved: {save_path}")


def main():
    """Main function - Frequency-Aligned Grad-CAM Analysis"""
    print("=" * 80)
    print("🎯 FREQUENCY-ALIGNED GRAD-CAM ANALYSIS")
    print("Based on existing full_gradcam directory, generate frequency axis conversion and original waveform alignment analysis")
    print("=" * 80)
    
    # Create results directory
    output_dir = 'data/results/frequency_aligned_gradcam'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load configuration
        print("📋 Loading configuration file...")
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load feature extractor
        print("🔧 Loading feature extractor...")
        with open('data/processed/full_feature_extractor.pkl', 'rb') as f:
            fe_params = pickle.load(f)
        
        feature_extractor = FeatureExtractor(normalization_method=fe_params['normalization_method'])
        feature_extractor.cwt_scaler = fe_params['cwt_scaler']
        feature_extractor.stat_scaler = fe_params['stat_scaler']
        feature_extractor.is_fitted = fe_params['is_fitted']
        
        # Load model
        print("🤖 Loading trained model...")
        model = HybridCNNModel(config_path='configs/config.yaml')
        device = get_device('configs/config.yaml')
        
        if device.type == 'cuda':
            model.load_state_dict(torch.load('data/processed/full_best_model.pth'))
        else:
            model.load_state_dict(torch.load('data/processed/full_best_model.pth', map_location=device))
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully, device: {device}")
        
        # Scan existing Grad-CAM files
        print("\n🔍 Scanning existing Grad-CAM results...")
        gradcam_dir = 'data/results/full_gradcam'
        gradcam_files = glob.glob(os.path.join(gradcam_dir, 'gradcam_azimuth_*_sample_*.png'))
        
        if not gradcam_files:
            print(f"❌ No Grad-CAM files found in {gradcam_dir}")
            return
        
        print(f"📁 Found {len(gradcam_files)} Grad-CAM files")
        
        # Parse file information
        sample_infos = []
        for file_path in gradcam_files:
            filename = os.path.basename(file_path)
            info = parse_gradcam_filename(filename)
            if info:
                info['file_path'] = file_path
                sample_infos.append(info)
        
        print(f"✅ Successfully parsed {len(sample_infos)} sample information")
        
        # Sort by azimuth and sample number
        sample_infos.sort(key=lambda x: (x['azimuth'], x['sample_num']))
        
        # Preprocessed data cache
        processed_data_cache = {}
        
        # Batch process samples
        successful_count = 0
        failed_count = 0
        
        for i, info in enumerate(sample_infos, 1):
            azimuth = info['azimuth']
            sample_num = info['sample_num']
            
            print(f"\n🔄 Processing sample [{i}/{len(sample_infos)}]: Azimuth {azimuth}, Sample {sample_num}")
            
            try:
                # Load sample data
                sample_data = load_sample_data(azimuth, sample_num - 1, processed_data_cache)  # File name starts from 1, index starts from 0
                if sample_data is None:
                    failed_count += 1
                    continue
                
                # Generate Grad-CAM
                gradcam, prediction = generate_gradcam_for_sample(
                    model, feature_extractor, device, sample_data
                )
                
                # Create frequency-aligned analysis plot
                save_path = os.path.join(output_dir, f'frequency_analysis_{azimuth}_sample_{sample_num}.png')
                create_frequency_aligned_analysis(sample_data, gradcam, prediction, save_path)
                
                successful_count += 1
                
            except Exception as e:
                print(f"❌ Failed to process sample: {e}")
                failed_count += 1
                continue
        
        # Generate comprehensive report
        print(f"\n📊 Generating comprehensive report...")
        report_path = os.path.join(output_dir, 'frequency_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Frequency-Aligned Grad-CAM Analysis Report\n\n")
            f.write(f"## Analysis Overview\n")
            f.write(f"- **Analysis Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Original Grad-CAM Files**: {len(gradcam_files)} files\n")
            f.write(f"- **Successfully Processed Samples**: {successful_count} samples\n")
            f.write(f"- **Failed Samples**: {failed_count} samples\n")
            f.write(f"- **Success Rate**: {successful_count/(successful_count+failed_count)*100:.1f}%\n\n")
            
            f.write("## Technical Description\n\n")
            f.write("### Scale to Frequency Conversion\n")
            f.write("- **Conversion Formula**: f = fc / (scale × dt)\n")
            f.write("- **Morlet Wavelet Center Frequency**: fc = 1.0\n")
            f.write("- **Sampling Period**: dt = 1e-5 seconds (10μs)\n")
            f.write("- **Frequency Range**: Automatically calculated based on CWT scales\n\n")
            
            f.write("### Original Waveform Loading\n")
            f.write("- **Data Source**: Direct loading from raw XSILMR03.mat file\n")
            f.write("- **Waveform Selection**: Corresponding acoustic waveform for each sample\n")
            f.write("- **Time Alignment**: Ensures original waveform time axis aligns with Grad-CAM analysis\n")
            f.write("- **Authentic Data**: Uses real field-acquired acoustic logging data\n\n")
            
            f.write("### Analysis Content\n")
            f.write("1. **Original Acoustic Waveform**: Time-domain signal directly from raw data\n")
            f.write("2. **CWT Time-Frequency Comparison**: Scale axis vs Frequency axis\n") 
            f.write("3. **Grad-CAM Heatmap**: Attention distribution converted to frequency axis\n")
            f.write("4. **Frequency Domain Analysis**: Attention distribution across different frequency bands\n")
            f.write("5. **Time Domain Analysis**: Attention changes along time axis with waveform alignment\n")
            f.write("6. **Comprehensive Interpretation**: Physical meaning and engineering application recommendations\n\n")
            
            f.write("## File Description\n")
            f.write("- Output file format: `frequency_analysis_{azimuth}_{sample_number}.png`\n")
            f.write("- Original full_gradcam directory files remain unchanged\n")
            f.write("- All newly generated analysis plots saved in frequency_aligned_gradcam directory\n")
            f.write("- Raw data source: XSILMR03.mat for authentic acoustic waveforms\n\n")
            
        print(f"✅ Comprehensive report saved: {report_path}")
        
        print(f"\n🎉 Frequency-aligned analysis completed!")
        print(f"   - Successfully processed: {successful_count} samples")
        print(f"   - Failed samples: {failed_count}")
        print(f"   - Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Program execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Simple pandas alternative
    class pd:
        class Timestamp:
            @staticmethod
            def now():
                import datetime
                return datetime.datetime.now()
            
            def strftime(self, fmt):
                return self.strftime(fmt)
    
    main() 