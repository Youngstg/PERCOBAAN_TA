import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import warnings
warnings.filterwarnings("ignore")

class SpeechToTextModel:
    def __init__(self, model_path=None):
        """
        Initialize Speech-to-Text model
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.tokenizer = None
        self.sample_rate = 16000
        
        if model_path:
            self.load_model(model_path)
        else:
            # Use default pre-trained model
            self.load_default_model()

    def load_default_model(self):
        """Load default pre-trained Wav2Vec2 model"""
        try:
            model_name = "facebook/wav2vec2-base-960h"
            self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.model.eval()
            print("Default Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"Error loading default model: {e}")

    def load_model(self, model_path):
        """
        Load pre-trained model from specified path
        Args:
            model_path: Path to the model file
        """
        try:
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
            else:
                # Try loading as HuggingFace model
                self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_path)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
                self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.load_default_model()

    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file for model input
        Args:
            audio_path: Path to audio file
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)
            
            return audio_tensor
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def process_audio(self, audio_path):
        """
        Convert audio file to text
        Args:
            audio_path: Path to audio file
        Returns:
            Transcribed text
        """
        if self.model is None:
            return "Model not loaded"
        
        try:
            # Preprocess audio
            audio_input = self.preprocess_audio(audio_path)
            if audio_input is None:
                return "Error processing audio file"
            
            # Make prediction
            with torch.no_grad():
                if hasattr(self.model, 'logits'):
                    # For Wav2Vec2 models
                    logits = self.model(audio_input.unsqueeze(0)).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.tokenizer.batch_decode(predicted_ids)[0]
                else:
                    # For custom models
                    output = self.model(audio_input.unsqueeze(0))
                    transcription = output  # Adjust based on your model output format
            
            return transcription.strip()
        except Exception as e:
            print(f"Error during transcription: {e}")
            return f"Error: {str(e)}"

    def transcribe_text(self, text):
        """
        Alternative method name for compatibility
        Args:
            text: Audio file path
        Returns:
            Transcribed text
        """
        return self.process_audio(text)