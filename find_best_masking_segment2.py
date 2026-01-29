"""
Traverse audio directory, cut audio into 2s segments, calculate how much of the command spectrum each segment's masking curve can cover
Select the 2s segment that can cover the largest area of the command among all audio files

Based on MPEG/ISO standard psychoacoustic masking model
"""

import argparse
import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

os.environ['TORCH_HOME'] = ''
# Import masking calculation functions
from psychoacoustic_masking import (
    bark_scale, 
    absolute_threshold_hearing
)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# Global VAD model variables (avoid reloading)
_vad_model = None
_vad_utils = None


def get_model2_partitions(sr, n_fft):
    """
    Get MPEG-1 Model 2 critical band (Partition) divisions
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # Bark scale approximation
    barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
    
    # Model 2 uses approximately 1/3 Bark resolution
    partition_indices = np.round(3.0 * barks).astype(int)
    n_partitions = np.max(partition_indices) + 1
    
    return partition_indices, n_partitions, barks

def spreading_function_model2(dz):
    """
    MPEG-1 Model 2 spreading function
    """
    # Typical spreading function shape
    val = 15.81 + 7.5 * dz - 17.5 * np.sqrt(1 + dz**2)
    return 10 ** (val / 10.0)

def compute_masking_threshold_model2(audio, sr, n_fft, hop_length):
    """
    Calculate masking threshold using MPEG-1 Model 2 standard
    Includes: partition energy, spreading function, prediction-based tonality detection, SNR offset
    """
    # 1. STFT (keep center=True to align time axis)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    X = stft
    power = np.abs(X) ** 2
    n_freqs, n_frames = power.shape
    
    # 2. Map to Partitions
    partition_indices, n_partitions, barks = get_model2_partitions(sr, n_fft)
    
    # Build mapping matrix [n_partitions, n_freqs]
    map_matrix = np.zeros((n_partitions, n_freqs))
    for k, p in enumerate(partition_indices):
        map_matrix[p, k] = 1.0
        
    # Calculate partition energy [n_partitions, n_frames]
    P_part = np.dot(map_matrix, power)
    
    # 3. Calculate spreading matrix
    # Calculate center Bark value for each partition
    part_barks = np.zeros(n_partitions)
    for p in range(n_partitions):
        mask = partition_indices == p
        if np.any(mask):
            part_barks[p] = np.mean(barks[mask])
        else:
            part_barks[p] = p / 3.0
            
    S = np.zeros((n_partitions, n_partitions))
    for i in range(n_partitions): # masker
        for j in range(n_partitions): # masked
            dz = part_barks[j] - part_barks[i]
            # Limit spreading range (-3 Bark to +8 Bark)
            if -3 <= dz <= 8:
                S[j, i] = spreading_function_model2(dz)
                
    # Spread energy
    E_spr = np.dot(S, P_part)
    
    # 4. Tonality - based on Unpredictability
    # c[k] = |X_t - X_pred| / (|X_t| + |X_pred|)
    # X_pred = 2*X_{t-1} - X_{t-2}
    
    c = np.ones_like(X, dtype=float) # Default high unpredictability (Noise)
    
    if n_frames > 2:
        X_curr = X[:, 2:]
        X_prev1 = X[:, 1:-1]
        X_prev2 = X[:, 0:-2]
        
        # Complex domain linear prediction
        X_pred = 2 * X_prev1 - X_prev2
        
        num = np.abs(X_curr - X_pred)
        den = np.abs(X_curr) + np.abs(X_pred) + 1e-10
        c[:, 2:] = num / den
        
    # Calculate partition weighted unpredictability
    cP = c * power
    cP_part = np.dot(map_matrix, cP)
    c_part = cP_part / (P_part + 1e-10)
    
    # Tonality index alpha (0=Noise, 1=Tone)
    # alpha = -0.299 - 0.43 * ln(c_part)
    alpha = -0.299 - 0.43 * np.log(c_part + 1e-10)
    alpha = np.clip(alpha, 0, 1)
    
    # 5. SNR offset
    # Model 2 empirical formula: SNR = alpha * 18.5 + (1-alpha) * 6.0 (dB)
    SNR_db = alpha * 18.5 + (1 - alpha) * 6.0
    SNR_linear = 10 ** (SNR_db / 10.0)
    
    # 6. Calculate partition masking threshold
    T_part = E_spr / SNR_linear
    
    # 7. Map back to frequency bins (Per-bin Threshold)
    # Assume uniform energy distribution within partition, distribute threshold to each bin
    bins_per_part = np.sum(map_matrix, axis=1)
    bins_per_part[bins_per_part == 0] = 1.0
    
    T_bin_part = T_part / bins_per_part[:, np.newaxis]
    
    global_masking = np.dot(map_matrix.T, T_bin_part)
    
    # 8. Absolute Threshold of Hearing (ATH)
    freqs_fft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    ath_db = absolute_threshold_hearing(freqs_fft)
    ath_linear = 10 ** (ath_db / 10.0)
    
    global_masking = np.maximum(global_masking, ath_linear[:, np.newaxis])
    
    return global_masking, freqs_fft, power, ath_db, alpha


def load_silero_vad_model():
    """
    Load Silero VAD model (singleton pattern)
    
    Returns:
        model: Silero VAD model
        utils: Utility functions
    """
    global _vad_model, _vad_utils
    
    if _vad_model is None:
        #print("Loading Silero VAD model...")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        print("VAD model loaded successfully!")
    
    return _vad_model, _vad_utils


def compute_coverage_area(masking_threshold, command_spectrum, freqs, freq_range=(0, 3000), threshold_margin_db=0.0):
    """
    Calculate the proportion of frequency bins in the command that are below the masking curve
    
    Args:
        masking_threshold: Masking threshold of music segment (linear power)
        command_spectrum: Power spectrum of command (linear power)
        freqs: Frequency axis
        freq_range: Frequency range to consider (default: 20-7000Hz)
        threshold_margin_db: Safety margin for masking judgment (dB), positive value means command needs to be lower than masking threshold by this much to be considered masked
                             For example threshold_margin_db=10 means command needs to be 10dB lower than masking threshold to be considered masked
                             0 means just below threshold is considered masked (default behavior)
    Returns:
        coverage_ratio: Ratio of command spectrum frequency bins below masking curve
        covered_count: Number of covered frequency bins
    """
    # Filter frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    
    masking_in_range = masking_threshold[freq_mask]
    command_in_range = command_spectrum[freq_mask]
    freqs_in_range = freqs[freq_mask]
    
    # Convert safety margin from dB to linear scale factor
    # threshold_margin_db > 0: command needs to be lower to be considered masked (stricter, safer)
    # threshold_margin_db < 0: command can be slightly higher and still be considered masked (looser)
    # For example margin=10dB means masking threshold is reduced to 10^(-10/10) = 0.1 times the original
    margin_linear = 10 ** (-threshold_margin_db / 10.0)
    
    # Adjust masking threshold: lower threshold makes judgment stricter
    # adjusted_threshold = original_threshold × 10^(-margin_db/10)
    adjusted_masking = masking_in_range * margin_linear
    
    # Calculate masked regions: command power lower than adjusted masking threshold (linear domain comparison)
    covered_mask = command_in_range < adjusted_masking
    
    # Calculate coverage ratio: number of covered frequency bins / total frequency bins
    covered_count = np.sum(covered_mask)
    total_count = len(covered_mask)
    coverage_ratio = covered_count / total_count if total_count > 0 else 0.0
    
    return coverage_ratio, covered_count


def precompute_command_features(command_audio, sr, n_fft, hop_length, vad_threshold=0.5):
    """
    Precompute command audio features: power spectrum and VAD mask
    """
    # 1. Resample to 16k (if needed)
    if sr != 16000:
        cmd_16k = librosa.resample(command_audio, orig_sr=sr, target_sr=16000)
    else:
        cmd_16k = command_audio
        
    # 2. Calculate STFT and power spectrum
    stft = librosa.stft(cmd_16k, n_fft=n_fft, hop_length=hop_length, window='hann')
    power_spec = np.abs(stft) ** 2
    
    # 3. Calculate VAD Mask
    model, _ = load_silero_vad_model()
    n_frames = power_spec.shape[1]
    vad_mask = np.zeros(n_frames, dtype=bool)
    
    # Batch VAD calculation might be faster, but keeping logic consistent here
    # Each frame corresponds to hop_length step size
    for i in range(n_frames):
        start = i * hop_length
        end = start + 512
        if end > len(cmd_16k):
            segment = np.pad(cmd_16k[start:], (0, 512 - len(cmd_16k[start:])), mode='constant')
        else:
            segment = cmd_16k[start:end]
            
        tensor = torch.from_numpy(segment).float()
        score = model(tensor, 16000).item()
        if score >= vad_threshold:
            vad_mask[i] = True
            
    return power_spec, vad_mask


def compute_global_masking_threshold(music_audio, sr, n_fft, hop_length, freqs):
    """
    Calculate frame-by-frame masking threshold matrix for entire music audio (using Model 2)
    """
    # Directly call Model 2 implementation
    global_masking, _, _, _, _ = compute_masking_threshold_model2(music_audio, sr, n_fft, hop_length)
    return global_masking


def process_single_audio(audio_path, command_features, freqs, segment_length=2.0, 
                         sr=16000, n_fft=2048, hop_length=512, overlap=0.5, threshold_margin_db=10.0, music_start_offset=1.0, vad_threshold=0.5, max_segments_per_audio=3):
    """
    Process single audio file, using full calculation + sliding window search for best segments
    
    Args:
        audio_path: Audio file path
        command_features: Precomputed command features (power_spec, vad_mask)
        freqs: Frequency axis
        segment_length: Output segment length (seconds)
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        overlap: (Deprecated, determined by sliding step)
        threshold_margin_db: Safety margin for masking judgment (dB)
        music_start_offset: Command start offset in segment (seconds)
        vad_threshold: (Used in precomputation)
        max_segments_per_audio: Maximum number of segments to keep per audio file
    Returns:
        top_segment_infos: Top max_segments_per_audio best segments for this audio file
    """
    try:
        # Read audio
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Skip if audio is too short
        segment_samples = int(segment_length * sr)
        if len(audio) < segment_samples:
            return []
            
        # Unpack command features
        cmd_power, cmd_vad_mask = command_features
        n_cmd_frames = cmd_power.shape[1]
        
        # 1. Calculate full masking threshold
        # print(f"  Calculate full masking threshold: {os.path.basename(audio_path)}")
        global_masking = compute_global_masking_threshold(audio, sr, n_fft, hop_length, freqs)
        
        # 2. Sliding window search
        # We need to slide on global_masking, window size is n_cmd_frames
        
        n_music_frames = global_masking.shape[1]
        
        # Calculate effective search range
        # Each search point i represents the position where command audio starts
        # This position corresponds to music_start_offset in the output segment
        # So output segment start time is time(i) - music_start_offset
        # Output segment end time is time(i) - music_start_offset + segment_length
        # We need to ensure the output segment is within the audio range
        
        offset_frames = int(music_start_offset * sr / hop_length)
        segment_frames = int(segment_length * sr / hop_length)
        
        # Calculate sliding step based on overlap
        # Smaller step means finer search, higher computational cost
        search_step = max(1, int(segment_frames * (1 - overlap)))
        
        # i is the frame index where command starts
        # Segment start frame: i - offset_frames
        # Segment end frame: i - offset_frames + segment_frames
        
        min_start_frame = offset_frames
        max_start_frame = n_music_frames - (segment_frames - offset_frames)
        
        # Also need to ensure i + n_cmd_frames <= n_music_frames
        max_start_frame = min(max_start_frame, n_music_frames - n_cmd_frames)
        
        if max_start_frame <= min_start_frame:
            return []
            
        all_segment_infos = []
        
        # Precompute margin factor
        margin_linear = 10 ** (-threshold_margin_db / 10.0)
        
        # Frequency range mask
        freq_range = (0, 3000)
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        # Optimization: only take the frequency range of interest for comparison
        cmd_power_masked = cmd_power[freq_mask, :]
        global_masking_masked = global_masking[freq_mask, :]
        
        # Count frames with speech
        vocal_frames_count = np.sum(cmd_vad_mask)
        if vocal_frames_count == 0:
            return []

        for i in range(min_start_frame, max_start_frame, search_step):
            # Extract corresponding masking threshold segment
            masking_seg = global_masking_masked[:, i : i + n_cmd_frames]
            
            # Adjust threshold
            adjusted_masking = masking_seg * margin_linear
            
            # Compare (Broadcasting)
            # cmd_power_masked: [F, T], adjusted_masking: [F, T]
            covered_matrix = cmd_power_masked < adjusted_masking
            
            # Calculate coverage ratio per frame
            # sum over freq axis -> [T]
            covered_bins_per_frame = np.sum(covered_matrix, axis=0)
            total_bins = covered_matrix.shape[0]
            frame_ratios = covered_bins_per_frame / total_bins
            
            # Determine if frame is masked (threshold 0.9)
            is_covered_frame = frame_ratios >= 0.9
            
            # Combine with VAD Mask
            # Only count frames that have speech and are masked
            effective_covered = is_covered_frame & cmd_vad_mask
            covered_count = np.sum(effective_covered)
            
            frame_coverage_ratio = covered_count / vocal_frames_count
            
            # Construct result
            # Only need to record key information, no need for detailed frame_results (takes too much memory)
            # Only when finally selected, if visualization is needed, regenerate detailed information
            
            start_frame_of_segment = i - offset_frames
            start_sample = start_frame_of_segment * hop_length
            end_sample = start_sample + segment_samples
            start_time = start_sample / sr
            
            # Simple fingerprint: take mean vector of masking threshold for this segment (average along time axis) -> [F]
            # Or take Chroma of audio. For speed, don't calculate Chroma here,
            # wait until Top N are selected, then calculate Chroma for deduplication.
            
            segment_info = {
                'audio_path': audio_path,
                'start_time': start_time,
                'end_time': start_time + segment_length,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'frame_coverage_ratio': frame_coverage_ratio,
                'covered_frames': covered_count,
                # 'frame_results': ... # Not stored for now
                'cmd_start_frame_in_global': i # Record command position in global, convenient for later recovery
            }
            all_segment_infos.append(segment_info)
        
        # Sort
        all_segment_infos.sort(key=lambda x: x['frame_coverage_ratio'], reverse=True)
        
        # Initial screening Top 50 (to avoid Chroma calculation being too slow)
        candidates = all_segment_infos[:50]
        
        # Supplement Chroma fingerprint and detailed information
        final_candidates = []
        for seg in candidates:
            # Extract audio segment to calculate Chroma
            seg_audio = audio[seg['start_sample']:seg['end_sample']]
            try:
                chroma = librosa.feature.chroma_stft(y=seg_audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
                fingerprint = np.mean(chroma, axis=1)
            except:
                fingerprint = np.zeros(12)
            seg['fingerprint'] = fingerprint
            
            # Restore frame_results (for compatibility with visualization code)
            # We don't need to recalculate masking here, just construct based on previous calculation results
            # But for code simplicity, and only doing this for a few Top ones, some repetition is acceptable
            # Or simply construct a simplified frame_results
            
            # The cost of recalculating detailed information is not large, since we only do it for a few
            # Here we can simply construct a fake frame_results or
            # Better way: recalculate in visualize_best_segment.
            # But select_diverse_segments doesn't need frame_results.
            # Only output in main function needs it.
            
            # We won't add frame_results first, when main function needs to save,
            # if visualize_best_segment needs it, it will read and analyze audio itself.
            # Wait, visualize_best_segment depends on best_segment['frame_results'] to draw figure 2.
            # So we need to supplement here.
            
            # Supplement frame_results
            i = seg['cmd_start_frame_in_global']
            masking_seg = global_masking[:, i : i + n_cmd_frames] # Take full frequency range
            # ... Repeat comparison logic above ...
            # For code reuse, we can extract the comparison logic, but just write it directly here
            
            # Recalculate detailed coverage for this segment
            seg_frame_results = []
            
            # Extract corresponding command part
            # Note: cmd_power is full frequency range
            cmd_power_seg = cmd_power
            
            # Adjust threshold
            adj_mask = masking_seg * margin_linear
            
            # Only calculate coverage in frequency range of interest
            mask_f = freq_mask
            cov_mat = cmd_power_seg[mask_f] < adj_mask[mask_f]
            ratios = np.sum(cov_mat, axis=0) / cov_mat.shape[0]
            
            for f_idx in range(n_cmd_frames):
                is_cov = ratios[f_idx] >= 0.9
                seg_frame_results.append({
                    'frame_idx': f_idx,
                    'coverage_ratio': ratios[f_idx],
                    'is_covered': is_cov,
                    'has_vocal': bool(cmd_vad_mask[f_idx]),
                    'vad_score': 1.0 if cmd_vad_mask[f_idx] else 0.0
                })
            
            seg['frame_results'] = seg_frame_results
            final_candidates.append(seg)
            
        # Diversity filtering
        min_time_dist = segment_length * 0.8
        top_segment_infos = select_diverse_segments(
            final_candidates, 
            max_segments_per_audio, 
            min_time_dist=min_time_dist,
            max_similarity=0.95
        )
        
        return top_segment_infos
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing audio {audio_path}: {e}")
        return []


def visualize_best_segment(best_segment, command_audio_path, output_path, 
                          sr=16000, n_fft=2048, hop_length=512, threshold_margin_db=0.0, music_start_offset=2.0):
    """
    Visualize the masking effect of the best segment
    
    Args:
        best_segment: Best segment information
        command_audio_path: Command audio path
        output_path: Output image path
        sr: Sample rate
        n_fft: FFT size
        hop_length: Frame shift
        threshold_margin_db: Safety margin for masking judgment (dB)
        music_start_offset: Start calculating masking effect from which second of the music (default 2.0 seconds)
    """
    # Load the best segment audio
    full_audio, _ = librosa.load(best_segment['audio_path'], sr=sr, mono=True)
    segment_audio = full_audio[best_segment['start_sample']:best_segment['end_sample']]
    
    # Extract music segment from specified position for masking calculation
    offset_samples = int(music_start_offset * sr)
    if offset_samples < len(segment_audio):
        segment_audio_for_masking = segment_audio[offset_samples:]
    else:
        segment_audio_for_masking = segment_audio
    
    # Load command audio
    command_audio, _ = librosa.load(command_audio_path, sr=sr, mono=True)
    
    # Calculate STFT of music segment to get actual power spectrum (for visualization, using part from offset)
    stft_seg = librosa.stft(segment_audio_for_masking, n_fft=n_fft, hop_length=hop_length, window='hann')
    power_seg_real = np.mean(np.abs(stft_seg) ** 2, axis=1)  # Average over all time frames (linear domain)
    power_seg_db = 10 * np.log10(power_seg_real + 1e-10)  # Convert to dB for display
    
    # Calculate masking threshold (using Model 2)
    masking_threshold_linear, freqs_seg, _, ath_db, _ = \
        compute_masking_threshold_model2(segment_audio_for_masking, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # Take average masking threshold for display
    masking_threshold_linear_avg = np.mean(masking_threshold_linear, axis=1)
    masking_threshold_db = 10 * np.log10(masking_threshold_linear_avg + 1e-10)
    
    # Calculate power spectrum of command audio (return linear domain)
    # Note: Also using Model 2 here, though a bit overkill, to obtain freqs_cmd and power_cmd_linear
    # Could actually calculate directly using stft
    stft_cmd = librosa.stft(command_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    power_cmd_linear_frames = np.abs(stft_cmd) ** 2
    power_cmd_linear = np.mean(power_cmd_linear_frames, axis=1)
    power_cmd_db = 10 * np.log10(power_cmd_linear + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Frequency range
    freq_range = (20, 7000)
    freq_mask = (freqs_seg >= freq_range[0]) & (freqs_seg <= freq_range[1])
    freqs_plot = freqs_seg[freq_mask]
    
    # Subplot 1: Masking curve and command spectrum (both displayed in dB)
    ax1 = axes[0]
    ax1.plot(freqs_plot, power_seg_db[freq_mask], 'b-', linewidth=2, label='Music Segment Power Spectrum (Average)', alpha=0.7)
    ax1.plot(freqs_plot, masking_threshold_db[freq_mask], 'r-', linewidth=2.5, label='Masking Threshold')
    
    # If there is a safety margin, draw the adjusted threshold
    if threshold_margin_db > 0:
        margin_linear = 10 ** (-threshold_margin_db / 10.0)
        adjusted_masking_linear = masking_threshold_linear * margin_linear
        adjusted_masking_db = 10 * np.log10(adjusted_masking_linear + 1e-10)
        ax1.plot(freqs_plot, adjusted_masking_db[freq_mask], 'r--', linewidth=2, 
                label=f'Adjusted Masking Threshold (-{threshold_margin_db}dB)', alpha=0.7)
    
    ax1.plot(freqs_plot, power_cmd_db[freq_mask], 'orange', linewidth=2, label='Command Power Spectrum', alpha=0.8)
    ax1.plot(freqs_plot, ath_db[freq_mask], 'g--', linewidth=1.5, label='Absolute Threshold of Hearing', alpha=0.5)
    
    # Fill covered region (where command power spectrum is below adjusted masking threshold, compared in linear domain)
    margin_linear = 10 ** (-threshold_margin_db / 10.0)
    adjusted_masking = masking_threshold_linear_avg[freq_mask] * margin_linear
    covered_mask = power_cmd_linear[freq_mask] < adjusted_masking
    if np.any(covered_mask):
        ax1.fill_between(freqs_plot[covered_mask], 
                        power_cmd_db[freq_mask][covered_mask],
                        masking_threshold_db[freq_mask][covered_mask],
                        alpha=0.3, color='green', label='Covered Region (Command Masked)')
    
    # Note: Masking threshold is usually higher than music power spectrum because:
    # 1. Masking threshold = peak power × spreading factor × tonality factor × offset factor (linear domain multiplication)
    # 2. In dB domain: masking_db = peak_db + spreading_db + tonality_db - offset_db
    # 3. Indicates that when music is present, the command must reach this threshold to be heard
    # 4. **MPEG/ISO standard is based on linear domain power spectrum calculation, only converted to dB for visualization**
    
    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (dB)', fontsize=12, fontweight='bold')
    title_text = 'Best Masking Segment Analysis (MPEG-1 Model 2 Standard)'
    if threshold_margin_db > 0:
        title_text += f' [Safety Margin: {threshold_margin_db}dB]'
    ax1.set_title(title_text, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(freq_range)
    ax1.set_xscale('log')
    ax1.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 7000])
    ax1.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '7k'])
    
    # Subplot 2: Coverage ratio per frame (frame-by-frame comparison results)
    ax2 = axes[1]
    frame_results = best_segment['frame_results']
    frame_indices = [r['frame_idx'] for r in frame_results]
    coverage_ratios = [r['coverage_ratio'] for r in frame_results]
    is_covered_list = [r['is_covered'] for r in frame_results]
    has_vocal_list = [r.get('has_vocal', True) for r in frame_results]  # Compatible with old results
    
    frame_times = np.array(frame_indices) * hop_length / sr
    
    # Draw coverage ratio curve
    ax2.plot(frame_times, coverage_ratios, 'purple', linewidth=2, marker='o', markersize=4, label='Frame Coverage Ratio')
    
    # Mark masked, unmasked, and no-vocal frames with different colors
    covered_times = []
    uncovered_times = []
    no_vocal_times = []
    
    for i in range(len(is_covered_list)):
        if not has_vocal_list[i]:
            no_vocal_times.append(frame_times[i])
        elif is_covered_list[i]:
            covered_times.append(frame_times[i])
        else:
            uncovered_times.append(frame_times[i])
    
    if covered_times:
        covered_indices = [frame_indices.index(int(t*sr/hop_length)) for t in covered_times]
        ax2.scatter(covered_times, [coverage_ratios[i] for i in covered_indices], 
                   color='green', s=100, marker='o', alpha=0.6, label='Masked Frames', zorder=5)
    if uncovered_times:
        uncovered_indices = [frame_indices.index(int(t*sr/hop_length)) for t in uncovered_times]
        ax2.scatter(uncovered_times, [coverage_ratios[i] for i in uncovered_indices], 
                   color='red', s=100, marker='x', alpha=0.6, label='Unmasked Frames (With Vocal)', zorder=5)
    if no_vocal_times:
        no_vocal_indices = [frame_indices.index(int(t*sr/hop_length)) for t in no_vocal_times]
        ax2.scatter(no_vocal_times, [coverage_ratios[i] for i in no_vocal_indices], 
                   color='gray', s=100, marker='s', alpha=0.6, label='No Vocal Frames', zorder=5)
    
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% Threshold')
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency Point Coverage Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Frame-by-Frame Masking Analysis (Using VAD to Filter No-Vocal Frames)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    
    # Add information text
    total_frames = len(frame_results)
    vocal_frames = sum(has_vocal_list)
    no_vocal_frames = total_frames - vocal_frames
    frame_coverage_ratio = best_segment.get('frame_coverage_ratio', 0.0)
    
    info_text = f"Audio file: {os.path.basename(best_segment['audio_path'])}\n"
    info_text += f"Time range: {best_segment['start_time']:.2f}s - {best_segment['end_time']:.2f}s\n"
    info_text += f"Frame coverage ratio: {frame_coverage_ratio:.2%}\n"
    info_text += f"Number of masked frames: {best_segment['covered_frames']}/{total_frames}\n"
    info_text += f"With vocal frames: {vocal_frames}, No vocal frames: {no_vocal_frames}"
    
    fig.text(0.02, 0.02, info_text, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization result saved to: {output_path}")
    plt.close()


def save_best_segment(best_segment, output_audio_path, sr=16000):
    """
    Save the best segment as an audio file
    
    Args:
        best_segment: Best segment information
        output_audio_path: Output audio path
        sr: Sample rate
    """
    import soundfile as sf
    
    # Load complete audio
    audio, _ = librosa.load(best_segment['audio_path'], sr=sr, mono=True)
    
    # Extract segment
    segment = audio[best_segment['start_sample']:best_segment['end_sample']]
    
    # Save
    sf.write(output_audio_path, segment, sr)
    print(f"Best segment saved to: {output_audio_path}")


def compute_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def select_diverse_segments(segment_infos, max_count, min_time_dist=2.0, max_similarity=0.95):
    """
    Filter out non-overlapping segments with significant content differences from the segment list
    Note: Deduplication is only performed within the same file
    
    Args:
        segment_infos: Segment list sorted in descending order by score
        max_count: Maximum number to retain
        min_time_dist: Minimum time interval (seconds), less than this is considered time overlap
        max_similarity: Maximum content similarity (0-1), greater than this is considered content duplication
    Returns:
        selected: Filtered segment list
    """
    if not segment_infos:
        return []
        
    selected = []
    
    for seg in segment_infos:
        if len(selected) >= max_count:
            break
            
        is_duplicate = False
        for sel in selected:
            # 0. Only check duplicates within the same file
            if seg['audio_path'] != sel['audio_path']:
                continue
            
            # 2. Check content similarity (if fingerprint data exists)
            if 'fingerprint' in seg and 'fingerprint' in sel:
                sim = compute_cosine_similarity(seg['fingerprint'], sel['fingerprint'])
                if sim > max_similarity:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            selected.append(seg)
            
    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Find 2-second segments in audio directory that can maximally mask the command'
    )
    parser.add_argument('audio_dir', type=str, help='Audio directory path')
    parser.add_argument('command_audio', type=str, help='Command audio file path')
    parser.add_argument('--output_dir', '-o', type=str, default='best_segments_3_otw2_musicall',
                       help='Output directory (default: best_segments_loose)')
    parser.add_argument('--segment_length', type=float, default=3.0,
                       help='Segment length (seconds, default: 3.0)')
    parser.add_argument('--overlap', type=float, default=0.95,
                       help='Segment overlap ratio (default: 0.95)')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (Hz)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--top_k', type=int, default=20, help='Save top K best segments')
    parser.add_argument('--pattern', type=str, default='*.wav', help='Audio file match pattern')
    parser.add_argument('--threshold_margin', type=float, default=0.0,
                       help='Safety margin for masking judgment (dB), command needs to be this much lower than masking threshold to be considered masked (default: 0.0)')
    parser.add_argument('--vad_threshold', type=float, default=0.5,
                       help='VAD threshold (0-1), used to determine if command frame contains speech (default: 0.5)')
    parser.add_argument('--max_segments_per_audio', type=int, default=3,
                       help='Maximum number of segments to keep per audio file (default: 3)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read command audio (keep original signal for frame-by-frame comparison)
    print(f"Reading command audio: {args.command_audio}")
    command_audio, _ = librosa.load(args.command_audio, sr=args.sr, mono=True)
    
    # Precompute command features (Power Spec, VAD Mask)
    print("Precomputing command audio features...")
    command_features = precompute_command_features(
        command_audio, args.sr, args.n_fft, args.hop_length, args.vad_threshold
    )
    
    # Get frequency axis (for masking calculation)
    freqs = librosa.fft_frequencies(sr=args.sr, n_fft=args.n_fft)
    
    # Get all audio files
    audio_files = glob.glob(os.path.join(args.audio_dir, args.pattern))
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return
    
    # Process all audio files
    all_segments = []
    
    print("\nStarting audio file analysis...")
    if args.threshold_margin > 0:
        print(f"Using safety margin: {args.threshold_margin} dB (command needs to be {args.threshold_margin}dB lower than masking threshold to be considered masked)")
    print(f"VAD threshold: {args.vad_threshold} (speech detection sensitivity)")
    print(f"Maximum segments per audio: {args.max_segments_per_audio}")
    
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        segment_infos = process_single_audio(
            audio_file, command_features, freqs,
            segment_length=args.segment_length,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            overlap=args.overlap,
            threshold_margin_db=args.threshold_margin,
            music_start_offset=1.0,
            vad_threshold=args.vad_threshold,
            max_segments_per_audio=args.max_segments_per_audio
        )
        
        # Add all segments from this audio file to the total list
        all_segments.extend(segment_infos)
    
    if len(all_segments) == 0:
        print("No valid audio segments found!")
        return
    
    # Sort by frame coverage ratio (higher is better)
    all_segments.sort(key=lambda x: x['frame_coverage_ratio'], reverse=True)
    
    # Output results
    print(f"\n{'='*80}")
    print(f"Found {len(all_segments)} valid segments")
    print(f"{'='*80}\n")
    
    print(f"Top {min(args.top_k, len(all_segments))} best segments:\n")
    
    for rank, segment in enumerate(all_segments[:args.top_k], 1):
        total_frames = len(segment['frame_results'])
        print(f"Rank {rank}:")
        print(f"  File: {os.path.basename(segment['audio_path'])}")
        print(f"  Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
        print(f"  Frame coverage ratio: {segment['frame_coverage_ratio']:.2%}")
        print(f"  Number of masked frames: {segment['covered_frames']}/{total_frames}")
        print()
        
        # Save segment audio
        output_audio = os.path.join(
            args.output_dir,
            f"rank{rank}_{os.path.basename(segment['audio_path'])}"
        )
        save_best_segment(segment, output_audio, sr=args.sr)
        
        # Visualization
        output_viz = os.path.join(
            args.output_dir,
            f"rank{rank}_visualization.png"
        )
        visualize_best_segment(
            segment, args.command_audio, output_viz,
            sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
            threshold_margin_db=args.threshold_margin, music_start_offset=2.0
        )
    
    # Save statistics
    stats_file = os.path.join(args.output_dir, 'statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Command audio: {args.command_audio}\n")
        f.write(f"Audio directory: {args.audio_dir}\n")
        f.write(f"Segment length: {args.segment_length}s\n")
        f.write(f"Safety margin: {args.threshold_margin} dB\n")
        f.write(f"Total analyzed files: {len(audio_files)}\n")
        f.write(f"Valid segments: {len(all_segments)}\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Top {min(args.top_k, len(all_segments))} best segments:\n")
        f.write("="*80 + "\n\n")
        
        for rank, segment in enumerate(all_segments[:args.top_k], 1):
            total_frames = len(segment['frame_results'])
            f.write(f"Rank {rank}:\n")
            f.write(f"  File: {segment['audio_path']}\n")
            f.write(f"  Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s\n")
            f.write(f"  Frame coverage ratio: {segment['frame_coverage_ratio']:.2%}\n")
            f.write(f"  Number of masked frames: {segment['covered_frames']}/{total_frames}\n\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"All results saved to directory: {args.output_dir}")


if __name__ == '__main__':
    main()
