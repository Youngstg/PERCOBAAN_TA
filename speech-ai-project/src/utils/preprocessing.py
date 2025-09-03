def normalize_audio(audio):
    # Normalize audio data to have a mean of 0 and a standard deviation of 1
    return (audio - np.mean(audio)) / np.std(audio)

def extract_features(audio):
    # Extract features from audio data (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return mfccs

def preprocess_audio(file_path):
    # Load audio file and preprocess it
    audio, sample_rate = librosa.load(file_path, sr=None)
    normalized_audio = normalize_audio(audio)
    features = extract_features(normalized_audio)
    return features