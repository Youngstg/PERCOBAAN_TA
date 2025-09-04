import numpy as np
import librosa
import torch
from scipy import signal

def normalize_audio(audio):
    """
    Normalize audio data to have a mean of 0 and a standard deviation of 1
    Args:
        audio: Input audio array
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    # Avoid division by zero
    std_val = np.std(audio)
    if std_val == 0:
        return audio - np.mean(audio)
    
    return (audio - np.mean(audio)) / std_val

def extract_features(audio, sample_rate=16000, n_mfcc=13):
    """
    Extract features from audio data (e.g., MFCCs)
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        n_mfcc: Number of MFCC coefficients
    Returns:
        Extracted MFCC features
    """
    try:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Extract additional features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Combine features
        features = np.vstack([
            mfccs,
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.array([])

def preprocess_audio(file_path, target_sample_rate=16000, duration=None):
    """
    Load and preprocess audio file
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate for resampling
        duration: Maximum duration in seconds (None for full audio)
    Returns:
        Preprocessed features and raw audio
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(
            file_path, 
            sr=target_sample_rate, 
            duration=duration
        )
        
        if len(audio) == 0:
            print(f"Warning: Empty audio file {file_path}")
            return None, None
        
        # Remove silence from beginning and end
        audio = trim_silence(audio, sample_rate)
        
        # Normalize audio
        normalized_audio = normalize_audio(audio)
        
        # Extract features
        features = extract_features(normalized_audio, sample_rate)
        
        return features, normalized_audio
        
    except Exception as e:
        print(f"Error preprocessing audio file {file_path}: {e}")
        return None, None

def trim_silence(audio, sample_rate, threshold=0.01):
    """
    Remove silence from the beginning and end of audio
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        threshold: Silence threshold
    Returns:
        Trimmed audio
    """
    try:
        # Use librosa to trim silence
        trimmed_audio, _ = librosa.effects.trim(
            audio, 
            top_db=20,  # Consider anything 20dB below peak as silence
            frame_length=2048,
            hop_length=512
        )
        return trimmed_audio
    except Exception as e:
        print(f"Error trimming silence: {e}")
        return audio

def resample_audio(audio, original_sr, target_sr):
    """
    Resample audio to target sample rate
    Args:
        audio: Input audio array
        original_sr: Original sample rate
        target_sr: Target sample rate
    Returns:
        Resampled audio
    """
    if original_sr == target_sr:
        return audio
    
    try:
        resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        return resampled_audio
    except Exception as e:
        print(f"Error resampling audio: {e}")
        return audio

def apply_noise_reduction(audio, sample_rate):
    """
    Apply basic noise reduction using spectral gating
    Args:
        audio: Input audio array
        sample_rate: Sample rate
    Returns:
        Denoised audio
    """
    try:
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_duration_samples = int(0.5 * sample_rate)
        noise_frame_count = noise_duration_samples // 512  # hop_length default is 512
        
        if noise_frame_count > 0:
            noise_magnitude = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
            
            # Apply spectral gating
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor
            
            clean_magnitude = magnitude - alpha * noise_magnitude
            clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
            
            # Reconstruct signal
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
        else:
            return audio
            
    except Exception as e:
        print(f"Error applying noise reduction: {e}")
        return audio

def augment_audio(audio, sample_rate, augmentation_type='pitch'):
    """
    Apply audio augmentation
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        augmentation_type: Type of augmentation ('pitch', 'speed', 'noise')
    Returns:
        Augmented audio
    """
    try:
        if augmentation_type == 'pitch':
            # Pitch shifting
            n_steps = np.random.randint(-4, 5)  # Random pitch shift between -4 and +4 semitones
            augmented_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        
        elif augmentation_type == 'speed':
            # Time stretching (speed change)
            rate = np.random.uniform(0.8, 1.2)  # Random speed between 0.8x and 1.2x
            augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
        
        elif augmentation_type == 'noise':
            # Add white noise
            noise_factor = 0.005
            noise = np.random.randn(len(audio))
            augmented_audio = audio + noise_factor * noise
        
        else:
            augmented_audio = audio
            
        return augmented_audio
        
    except Exception as e:
        print(f"Error applying augmentation: {e}")
        return audio

def batch_preprocess(file_paths, target_sample_rate=16000):
    """
    Preprocess multiple audio files
    Args:
        file_paths: List of file paths
        target_sample_rate: Target sample rate
    Returns:
        List of preprocessed features and audio data
    """
    results = []
    
    for file_path in file_paths:
        features, audio = preprocess_audio(file_path, target_sample_rate)
        if features is not None and audio is not None:
            results.append({
                'file_path': file_path,
                'features': features,
                'audio': audio
            })
        else:
            print(f"Failed to preprocess {file_path}")
    
    return results