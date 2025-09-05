import torch
import torchaudio
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import warnings
warnings.filterwarnings("ignore")

class TextToSpeechModel:
    def __init__(self, model_path=None, device: str | None = None):
        """
        Initialize Text-to-Speech model
        Args:
            model_path: Path to pre-trained model (optional)
            device: 'cuda', 'cpu', 'mps', or None for auto
        """
        self.model = None
        self.processor = None
        self.vocoder = None
        self.sample_rate = 16000
        self.device = self._select_device(device)
        
        if model_path:
            self.load_model(model_path)
        else:
            self.load_default_model()

    def _select_device(self, device: str | None) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_default_model(self):
        """Load default pre-trained SpeechT5 model"""
        try:
            model_name = "microsoft/speecht5_tts"
            vocoder_name = "microsoft/speecht5_hifigan"
            
            self.processor = SpeechT5Processor.from_pretrained(model_name)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
            
            self.model.to(self.device).eval()
            self.vocoder.to(self.device).eval()
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
                self.model.to(self.device).eval()
                # Attempt to also load default processor and vocoder
                if self.processor is None:
                    self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                if self.vocoder is None:
                    self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device).eval()
            else:
                # Try loading as HuggingFace model
                self.processor = SpeechT5Processor.from_pretrained(model_path)
                self.model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
                # Always ensure a vocoder is available
                if self.vocoder is None:
                    self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                self.model.to(self.device).eval()
                self.vocoder.to(self.device).eval()
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
            speaker_embeddings = torch.zeros((1, 512), device=self.device)  # Default embedding
            
            # Generate speech
            with torch.no_grad():
                input_ids = inputs["input_ids"].to(self.device)
                # Optional mixed precision on CUDA
                use_amp = self.device.type == "cuda"
                if use_amp:
                    with torch.cuda.amp.autocast():
                        speech = self.model.generate_speech(
                            input_ids,
                            speaker_embeddings,
                            vocoder=self.vocoder
                        )
                else:
                    speech = self.model.generate_speech(
                        input_ids,
                        speaker_embeddings,
                        vocoder=self.vocoder
                    )
            
            return speech.detach().cpu().numpy(), self.sample_rate
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
            # Generate speech to a temporary file
            self.tts_engine.save_to_file(text, temp_file)
            self.tts_engine.runAndWait()
            # Ensure engine queue is flushed to avoid stalling on next call
            try:
                self.tts_engine.stop()
            except Exception:
                pass
            
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
            torchaudio.save(output_path, waveform_tensor.detach().cpu(), sample_rate)
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
        # Fast path: when using pyttsx3 fallback, save directly to target file
        if hasattr(self, 'use_pyttsx3') and self.use_pyttsx3:
            try:
                self.tts_engine.save_to_file(text, output_path)
                self.tts_engine.runAndWait()
                # Ensure queue is cleared to prevent subsequent stalls
                try:
                    self.tts_engine.stop()
                except Exception:
                    pass
                # Best-effort wait until the file is fully written (Windows file locking)
                import os, time
                last_size = -1
                stable_count = 0
                for _ in range(20):  # up to ~2s
                    if os.path.exists(output_path):
                        size = os.path.getsize(output_path)
                        if size == last_size and size > 0:
                            stable_count += 1
                            if stable_count >= 2:
                                break
                        else:
                            stable_count = 0
                            last_size = size
                    time.sleep(0.1)
                print(f"Audio saved successfully to {output_path}")
                return True
            except Exception as e:
                print(f"pyttsx3 direct save failed: {e}")
                # Fallback to waveform path below

        # Default path: generate waveform then save via torchaudio
        waveform, sample_rate = self.text_to_audio(text)
        if waveform is not None:
            self.save_audio(waveform, sample_rate, output_path)
            return True
        else:
            print("Failed to generate audio from text")
            return False
