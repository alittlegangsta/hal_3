# Frequency-Aligned Grad-CAM Analysis Report

## Analysis Overview
- **Analysis Time**: 2025-06-28 22:28:34
- **Original Grad-CAM Files**: 24 files
- **Successfully Processed Samples**: 24 samples
- **Failed Samples**: 0 samples
- **Success Rate**: 100.0%

## Technical Description

### Scale to Frequency Conversion
- **Conversion Formula**: f = fc / (scale × dt)
- **Morlet Wavelet Center Frequency**: fc = 1.0
- **Sampling Period**: dt = 1e-5 seconds (10μs)
- **Frequency Range**: Automatically calculated based on CWT scales

### Original Waveform Loading
- **Data Source**: Direct loading from raw XSILMR03.mat file
- **Waveform Selection**: Corresponding acoustic waveform for each sample
- **Time Alignment**: Ensures original waveform time axis aligns with Grad-CAM analysis
- **Authentic Data**: Uses real field-acquired acoustic logging data

### Analysis Content
1. **Original Acoustic Waveform**: Time-domain signal directly from raw data
2. **CWT Time-Frequency Comparison**: Scale axis vs Frequency axis
3. **Grad-CAM Heatmap**: Attention distribution converted to frequency axis
4. **Frequency Domain Analysis**: Attention distribution across different frequency bands
5. **Time Domain Analysis**: Attention changes along time axis with waveform alignment
6. **Comprehensive Interpretation**: Physical meaning and engineering application recommendations

## File Description
- Output file format: `frequency_analysis_{azimuth}_{sample_number}.png`
- Original full_gradcam directory files remain unchanged
- All newly generated analysis plots saved in frequency_aligned_gradcam directory
- Raw data source: XSILMR03.mat for authentic acoustic waveforms

