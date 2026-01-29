"""
Calculate and plot psychoacoustic masking effect frequency curves using MPEG/ISO standards

Based on MPEG-1 Layer III (MP3) and ISO/IEC 11172-3 psychoacoustic model
Includes:
1. Tonality Analysis
2. Masking Threshold Calculation
3. Absolute Threshold of Hearing (ATH)
4. Simultaneous Masking

Important: All calculations are performed in linear power domain, converted to dB only for visualization
"""

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# Set font for plotting
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def bark_scale(f):
    """
    Convert frequency to Bark scale
    Bark scale is a psychoacoustic scale based on human critical bands
    
    Args:
        f: Frequency (Hz)
    Returns:
        Bark value
    """
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def bark_to_hz(bark):
    """
    Convert Bark scale back to frequency
    Using approximate inverse transformation
    
    Args:
        bark: Bark value
    Returns:
        Frequency (Hz)
    """
    # Using Traunmüller (1990) approximation formula
    return 600 * np.sinh(bark / 6.0)


def absolute_threshold_hearing(f):
    """
    Absolute Threshold of Hearing (ATH) - Minimum sound pressure level audible in quiet
    Based on ISO 226 standard and MPEG psychoacoustic model
    
    Args:
        f: Frequency (Hz)
    Returns:
        Threshold (dB SPL)
    """
    # MPEG-1 psychoacoustic model ATH formula
    f_khz = f / 1000.0
    ath = 3.64 * (f_khz ** -0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + \
          1e-3 * (f_khz ** 4)
    return ath


def spreading_function(bark_distance):
    """
    Spreading Function
    Describes spreading characteristics of masking effect along frequency axis
    Based on MPEG-1 Layer III standard
    
    Args:
        bark_distance: Frequency distance on Bark scale
    Returns:
        Spreading attenuation value (dB)
    """
    # MPEG-1 psychoacoustic model spreading function
    # Upward spreading (high frequency) attenuates faster than downward (low frequency)
    spreading = np.zeros_like(bark_distance)
    
    # Downward spreading (bark_distance < 0)
    mask_lower = bark_distance < 0
    spreading[mask_lower] = 27 * bark_distance[mask_lower]
    
    # Upward spreading (0 < bark_distance < 1)
    mask_middle = (bark_distance > 0) & (bark_distance < 1)
    if np.any(mask_middle):
        spreading[mask_middle] = -bark_distance[mask_middle] * (24 + 230/bark_distance[mask_middle] - 0.2*bark_distance[mask_middle])
    
    # Case where bark_distance == 0
    mask_zero = bark_distance == 0
    spreading[mask_zero] = 0
    
    # Upward spreading far end (bark_distance >= 1)
    mask_upper = bark_distance >= 1
    spreading[mask_upper] = -bark_distance[mask_upper] * (24 + 230/bark_distance[mask_upper])
    
    return spreading


def compute_tonality(magnitude_spectrum, freqs, window_size=5):
    """
    Calculate tonality of spectrum
    High tonality components (e.g., pure tones) produce stronger masking effects
    
    Args:
        magnitude_spectrum: Magnitude spectrum
        freqs: Corresponding frequencies
        window_size: Local window size
    Returns:
        tonality: Tonality coefficient (0-1)
    """
    n_bins = len(magnitude_spectrum)
    tonality = np.zeros(n_bins)
    
    for i in range(window_size, n_bins - window_size):
        # Calculate geometric and arithmetic mean within local window
        window = magnitude_spectrum[i-window_size:i+window_size+1]
        geometric_mean = np.exp(np.mean(np.log(window + 1e-10)))
        arithmetic_mean = np.mean(window)
        
        # Tonality coefficient: geometric mean close to arithmetic mean indicates pure tone (high tonality)
        if arithmetic_mean > 1e-10:
            tonality[i] = geometric_mean / arithmetic_mean
        else:
            tonality[i] = 0.0
    
    return tonality


def compute_masking_threshold(audio, sr=16000, n_fft=2048, hop_length=512, frame_idx=None):
    """
    Calculate psychoacoustic masking threshold (MPEG/ISO standard - linear domain calculation)
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        frame_idx: Frame index to analyze, if None analyze average of entire audio
    Returns:
        freqs: Frequency axis (Hz)
        masking_threshold: Masking threshold (linear power)
        power_spectrum: Power spectrum (linear power)
        ath_db: Absolute threshold of hearing (dB, for visualization only)
        tonality: Tonality coefficient (0-1)
    """
    # Calculate STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)
    power = magnitude ** 2  # Keep in linear domain, conforming to MPEG/ISO standard
    
    # If frame index specified, analyze only that frame; otherwise take average
    if frame_idx is not None and frame_idx < power.shape[1]:
        power_frame = power[:, frame_idx]  # Linear power
        magnitude_frame = magnitude[:, frame_idx]
    else:
        power_frame = np.mean(power, axis=1)  # Linear power
        magnitude_frame = np.mean(magnitude, axis=1)
    
    # Frequency axis
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Calculate Bark scale
    bark_freqs = bark_scale(freqs)
    
    # Calculate tonality
    tonality = compute_tonality(magnitude_frame, freqs, window_size=3)
    
    # Initialize masking threshold to absolute threshold of hearing (convert to linear domain)
    ath_db = absolute_threshold_hearing(freqs)
    ath_linear = 10 ** (ath_db / 10.0)  # dB -> linear power
    masking_threshold = ath_linear.copy()
    
    # Find significant spectral peaks (potential masking sources)
    # Use local maximum detection (in linear domain)
    power_frame_max = np.max(power_frame)
    peak_threshold = power_frame_max * 10 ** (-60/10)  # Peaks within 60dB range (linear domain)
    is_peak = np.zeros(len(power_frame), dtype=bool)
    
    for i in range(2, len(power_frame) - 2):
        if power_frame[i] > peak_threshold and \
           power_frame[i] > power_frame[i-1] and \
           power_frame[i] > power_frame[i+1] and \
           power_frame[i] > power_frame[i-2] and \
           power_frame[i] > power_frame[i+2]:
            is_peak[i] = True
    
    # For each peak, calculate its masking effect (in linear domain)
    for i in np.where(is_peak)[0]:
        peak_power = power_frame[i]  # Linear power
        peak_bark = bark_freqs[i]
        peak_tonality = tonality[i]
        
        # Tonality modulation factor: high tonality signals produce stronger masking
        tonality_factor = 1.0 + 0.5 * peak_tonality
        
        # Calculate masking contribution of this peak to all frequencies
        bark_distance = bark_freqs - peak_bark
        spreading_db = spreading_function(bark_distance)  # Spreading function still in dB
        spreading_linear = 10 ** (spreading_db / 10.0)  # Convert to linear gain
        
        # Masking threshold = peak power × spreading factor × tonality factor × offset factor
        # Offset considers tonality: pure tones have stronger masking ability
        offset_db = 14.5 + 11 * (1 - peak_tonality)  # Tonal signal has smaller offset (stronger masking)
        offset_linear = 10 ** (-offset_db / 10.0)  # Negative sign for attenuation
        
        # Linear domain multiplication (physically meaningful)
        masking_contribution = peak_power * spreading_linear * tonality_factor * offset_linear
        
        # Take maximum of all masking contributions
        masking_threshold = np.maximum(masking_threshold, masking_contribution)
    
    # Final masking threshold cannot be lower than absolute threshold of hearing
    masking_threshold = np.maximum(masking_threshold, ath_linear)
    
    return freqs, masking_threshold, power_frame, ath_db, tonality


def plot_masking_curve(audio_path, output_path=None, frame_time=None, 
                       sr=16000, n_fft=2048, hop_length=512, 
                       freq_range=(20, 8000)):
    """
    Plot masking effect curve
    
    Args:
        audio_path: Input audio file path
        output_path: Output image path
        frame_time: Time point to analyze (seconds), if None analyze entire audio
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        freq_range: Frequency range to display (Hz)
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Calculate frame index
    frame_idx = None
    if frame_time is not None:
        frame_idx = int(frame_time * sr / hop_length)
    
    # Calculate masking threshold (returns linear domain)
    freqs, masking_threshold_linear, power_spectrum_linear, ath_db, tonality = \
        compute_masking_threshold(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, frame_idx=frame_idx)
    
    # Convert to dB for visualization
    power_spectrum_db = 10 * np.log10(power_spectrum_linear + 1e-10)
    masking_threshold_db = 10 * np.log10(masking_threshold_linear + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Frequency range mask
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[freq_mask]
    
    # Subplot 1: Power spectrum, masking threshold and absolute threshold of hearing
    ax1 = axes[0]
    ax1.plot(freqs_plot, power_spectrum_db[freq_mask], 'b-', linewidth=2, label='Power Spectrum', alpha=0.7)
    ax1.plot(freqs_plot, masking_threshold_db[freq_mask], 'r-', linewidth=2.5, label='Masking Threshold (MPEG Standard)')
    ax1.plot(freqs_plot, ath_db[freq_mask], 'g--', linewidth=1.5, label='Absolute Threshold of Hearing (ATH)', alpha=0.7)
    
    # Fill masking region
    min_threshold = np.min(masking_threshold_db[freq_mask]) - 20
    ax1.fill_between(freqs_plot, masking_threshold_db[freq_mask], 
                     min_threshold * np.ones_like(freqs_plot),
                     alpha=0.2, color='red', label='Masking Region')
    
    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Psychoacoustic Masking Effect Curve (MPEG/ISO Standard - Linear Domain Calculation)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(freq_range)
    
    # Use logarithmic scale to match human perception
    ax1.set_xscale('log')
    ax1.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 8000])
    ax1.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '8k'])
    
    # Subplot 2: Tonality analysis
    ax2 = axes[1]
    ax2.plot(freqs_plot, tonality[freq_mask], 'purple', linewidth=2, label='Tonality Coefficient')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tonality', fontsize=12, fontweight='bold')
    ax2.set_title('Spectral Tonality Analysis', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(freq_range)
    ax2.set_ylim([0, 1])
    ax2.set_xscale('log')
    ax2.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 8000])
    ax2.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '8k'])
    
    # Add text information
    info_text = f"Sample Rate: {sr} Hz\n"
    info_text += f"FFT Size: {n_fft}\n"
    if frame_time is not None:
        info_text += f"Analysis Time: {frame_time:.3f}s"
    else:
        info_text += "Analysis: Global Average"
    
    fig.text(0.02, 0.02, info_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Masking curve saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return freqs, masking_threshold_linear, power_spectrum_linear


def compute_snr_masked(audio_path, noise_audio_path=None, sr=16000, n_fft=2048, hop_length=512):
    """
    Calculate perceptual SNR considering psychoacoustic masking
    
    Args:
        audio_path: Original audio path
        noise_audio_path: Noisy audio path (if None, calculate masking margin of itself)
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
    Returns:
        perceptual_snr: Perceptual SNR (dB)
        masked_ratio: Ratio of masked frequency components
    """
    # Load audio
    audio_clean, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Calculate masking threshold (returns linear power)
    freqs, masking_threshold_linear, power_spectrum_linear, ath_db, tonality = \
        compute_masking_threshold(audio_clean, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    if noise_audio_path:
        # Load noisy audio
        audio_noisy, _ = librosa.load(noise_audio_path, sr=sr, mono=True)
        
        # Calculate noise power spectrum (linear domain)
        stft_noisy = librosa.stft(audio_noisy, n_fft=n_fft, hop_length=hop_length)
        stft_clean = librosa.stft(audio_clean, n_fft=n_fft, hop_length=hop_length)
        
        noise_power_linear = np.mean(np.abs(stft_noisy - stft_clean) ** 2, axis=1)
        
        # Calculate ratio of masked noise (compare in linear domain)
        masked_noise = noise_power_linear < masking_threshold_linear
        masked_ratio = np.sum(masked_noise) / len(masked_noise)
        
        # Perceptual SNR: only consider unmasked noise (convert to dB)
        signal_power_db = 10 * np.log10(np.mean(power_spectrum_linear) + 1e-10)
        if np.any(~masked_noise):
            unmasked_noise_power_db = 10 * np.log10(np.mean(noise_power_linear[~masked_noise]) + 1e-10)
        else:
            unmasked_noise_power_db = -np.inf
        perceptual_snr = signal_power_db - unmasked_noise_power_db
    else:
        # Calculate ratio of signal components above masking threshold (compare in linear domain)
        above_threshold = power_spectrum_linear > masking_threshold_linear
        masked_ratio = 1.0 - np.sum(above_threshold) / len(above_threshold)
        
        # Masking margin: average difference between signal power and masking threshold (convert to dB)
        margin_linear = np.mean(power_spectrum_linear - masking_threshold_linear)
        perceptual_snr = 10 * np.log10(margin_linear + 1e-10)
    
    return perceptual_snr, masked_ratio


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and plot psychoacoustic masking effect curves using MPEG/ISO standards'
    )
    parser.add_argument('input_audio', type=str, help='Input audio file path')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output image path (default: <input>_masking.png)')
    parser.add_argument('--time', '-t', type=float, default=None,
                       help='Time point to analyze (seconds), analyze entire audio if not specified')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (Hz)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--freq_min', type=float, default=20, help='Minimum display frequency (Hz)')
    parser.add_argument('--freq_max', type=float, default=8000, help='Maximum display frequency (Hz)')
    parser.add_argument('--compute_snr', action='store_true', 
                       help='Calculate perceptual SNR and masking ratio')
    parser.add_argument('--noise_audio', type=str, default=None,
                       help='Noisy audio path (for SNR calculation)')
    
    args = parser.parse_args()
    
    # Generate default output path
    if args.output is None:
        import os
        base_name = os.path.splitext(args.input_audio)[0]
        args.output = f"{base_name}_masking.png"
    
    # Plot masking curve
    print(f"Analyzing audio: {args.input_audio}")
    if args.time is not None:
        print(f"Analysis time point: {args.time}s")
    else:
        print("Analysis mode: Global average")
    
    freqs, masking_threshold_linear, power_spectrum_linear = plot_masking_curve(
        audio_path=args.input_audio,
        output_path=args.output,
        frame_time=args.time,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_range=(args.freq_min, args.freq_max)
    )
    
    # Convert to dB for statistical display
    power_spectrum_db = 10 * np.log10(power_spectrum_linear + 1e-10)
    masking_threshold_db = 10 * np.log10(masking_threshold_linear + 1e-10)
    
    # Calculate statistics
    print(f"\n=== Masking Effect Statistics ===")
    print(f"Frequency range: {args.freq_min} - {args.freq_max} Hz")
    print(f"Average power spectrum: {np.mean(power_spectrum_db):.2f} dB")
    print(f"Average masking threshold: {np.mean(masking_threshold_db):.2f} dB")
    print(f"Masking margin: {np.mean(power_spectrum_db - masking_threshold_db):.2f} dB")
    
    # Calculate perceptual SNR
    if args.compute_snr:
        print(f"\n=== Perceptual SNR Analysis ===")
        perceptual_snr, masked_ratio = compute_snr_masked(
            args.input_audio, 
            noise_audio_path=args.noise_audio,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length
        )
        print(f"Perceptual SNR: {perceptual_snr:.2f} dB")
        print(f"Masking ratio: {masked_ratio*100:.2f}%")


if __name__ == '__main__':
    main()
