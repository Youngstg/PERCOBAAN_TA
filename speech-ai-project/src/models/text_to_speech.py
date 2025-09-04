import torch
import torchaudio
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import warnings
warnings.filterwarnings("ignore")

class TextToSpeechModel:
    def __init__(self, model_path=None):
        """
        Initialize Text-to-Speech model
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.processor = None
        self.vocoder = None
        self.sample_rate = 16000
        
        if model_path:
            self.load_model(model_path)
        else:
            self.load_default_model()

    def load_default_model(self):
        """Load default pre-trained SpeechT5 model"""
        try:
            model_name = "microsoft/speecht5_tts"
            vocoder_name = "microsoft/speecht5_hifigan"
            
            self.processor = SpeechT5Processor.from_pretrained(model_name)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
            
            self.model.eval()
            self.vocoder.eval()
            print("Default SpeechT5 model loaded successfully")
        except Exception as e:
            print(f"Error loading default model: {e}")
            self.load_fallback_model()

    def load_fallback_model(self):
        """Load a simpler fallback TTS solution"""
        try:
            # Using torch's built-in text-to-speech if available
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.use_pyttsx3 = True
            print("Fallback pyttsx3 TTS engine initialized")
        except ImportError:
            print("pyttsx3 not available. Please install it: pip install pyttsx3")
            self.use_pyttsx3 = False

    def load_model(self, model_path):
        """
        Load pre-trained model from specified path
        Args:
            model_path: Path to the model
        """
        try:
            if isinstance(model_path, str) and (model_path.endswith('.pt') or model_path.endswith('.pth')):
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
            else:
                # Try loading as HuggingFace model
                self.processor = SpeechT5Processor.from_pretrained(model_path)
                self.model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
                self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.load_default_model()

    def text_to_audio(self, text):
        """
        Convert text to audio waveform
        Args:
            text: Input text to convert
        Returns:
            Tuple of (waveform, sample_rate)
        """
        if hasattr(self, 'use_pyttsx3') and self.use_pyttsx3:
            return self._pyttsx3_synthesis(text)
        
        if self.model is None or self.processor is None:
            raise ValueError("Model not properly loaded")
        
        try:
            # Tokenize text
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Generate speaker embeddings (using default)
            # In a real implementation, you might want to use specific speaker embeddings
            speaker_embeddings = torch.zeros((1, 512))  # Default embedding
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embeddings, 
                    vocoder=self.vocoder
                )
            
            return speech.cpu().numpy(), self.sample_rate
        except Exception as e:
            print(f"Error during text-to-speech conversion: {e}")
            return None, None

    def _pyttsx3_synthesis(self, text):
        """
        Fallback synthesis using pyttsx3
        Args:
            text: Text to synthesize
        Returns:
            Tuple of (None, sample_rate) - pyttsx3 saves directly to file
        """
        try:
            temp_file = "temp_tts_output.wav"
            self.tts_engine.save_to_file(text, temp_file)
            self.tts_engine.runAndWait()
            
            # Load the generated audio
            import librosa
            audio, sr = librosa.load(temp_file, sr=self.sample_rate)
            
            # Clean up temp file
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return audio, sr
        except Exception as e:
            print(f"Error with pyttsx3 synthesis: {e}")
            return None, None

    def save_audio(self, waveform, sample_rate, output_path):
        """
        Save audio waveform to file
        Args:
            waveform: Audio waveform data
            sample_rate: Sample rate of audio
            output_path: Output file path
        """
        try:
            if waveform is None:
                print("No waveform data to save")
                return False
            
            # Ensure waveform is in the right format
            if isinstance(waveform, np.ndarray):
                waveform_tensor = torch.FloatTensor(waveform)
            else:
                waveform_tensor = waveform
            
            # Add batch dimension if needed
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            
            # Save audio file
            torchaudio.save(output_path, waveform_tensor, sample_rate)
            print(f"Audio saved successfully to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False

    def convert_text_to_audio(self, text, output_path):
        """
        Convert text to audio and save to file
        Args:
            text: Input text
            output_path: Output audio file path
        """
        waveform, sample_rate = self.text_to_audio(text)
        if waveform is not None:
            self.save_audio(waveform, sample_rate, output_path)
        else:
            print("Failed to generate audio from text")