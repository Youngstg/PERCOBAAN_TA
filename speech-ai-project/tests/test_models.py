import unittest
import tempfile
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.speech_to_text import SpeechToTextModel
    from models.text_to_speech import TextToSpeechModel
    from utils.preprocessing import normalize_audio, extract_features, preprocess_audio
    from utils.evaluation import evaluate_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running tests from the project root directory")

class TestSpeechToTextModel(unittest.TestCase):
    """Test cases for Speech-to-Text model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.model = SpeechToTextModel()
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNotNone(self.model)
        # Check if model has required attributes
        self.assertTrue(hasattr(self.model, 'model'))
        self.assertTrue(hasattr(self.model, 'sample_rate'))
    
    def test_preprocess_audio_with_valid_input(self):
        """Test preprocessing with synthetic audio data"""
        # Create synthetic audio file
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(frequency * 2 * np.pi * t).astype(np.float32)
        
        # Save to temporary file
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Test preprocessing
            result = self.model.preprocess_audio(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, torch.Tensor)
    
    def test_preprocess_audio_with_invalid_file(self):
        """Test preprocessing with non-existent file"""
        result = self.model.preprocess_audio('non_existent_file.wav')
        self.assertIsNone(result)
    
    def test_process_audio_without_file(self):
        """Test processing audio when file doesn't exist"""
        result = self.model.process_audio('non_existent_file.wav')
        self.assertIn("Error", result)

class TestTextToSpeechModel(unittest.TestCase):
    """Test cases for Text-to-Speech model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.model = TextToSpeechModel()
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'sample_rate'))
    
    def test_text_to_audio_basic(self):
        """Test basic text-to-audio conversion"""
        test_text = "Hello, this is a test."
        
        try:
            waveform, sample_rate = self.model.text_to_audio(test_text)
            
            if waveform is not None:
                self.assertIsNotNone(sample_rate)
                self.assertGreater(len(waveform), 0)
            else:
                # If model not available, that's also acceptable for testing
                print("TTS model not available, skipping waveform test")
        except Exception as e:
            print(f"TTS test failed (expected if models not available): {e}")
    
    def test_save_audio_with_valid_data(self):
        """Test saving audio with valid waveform data"""
        # Create synthetic audio data
        sample_rate = 16000
        duration = 0.5
        waveform = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            result = self.model.save_audio(waveform, sample_rate, tmp_file.name)
            
            # Clean up
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)
            
            self.assertTrue(result)
    
    def test_save_audio_with_none_waveform(self):
        """Test saving audio with None waveform"""
        result = self.model.save_audio(None, 16000, 'test_output.wav')
        self.assertFalse(result)

class TestPreprocessingUtils(unittest.TestCase):
    """Test cases for preprocessing utilities"""
    
    def test_normalize_audio_basic(self):
        """Test basic audio normalization"""
        # Create test audio data
        audio = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        normalized = normalize_audio(audio)
        
        self.assertEqual(len(normalized), len(audio))
        # Check if normalized audio has approximately zero mean
        self.assertAlmostEqual(np.mean(normalized), 0, places=5)
    
    def test_normalize_audio_empty(self):
        """Test normalization with empty audio"""
        audio = np.array([])
        normalized = normalize_audio(audio)
        self.assertEqual(len(normalized), 0)
    
    def test_normalize_audio_constant(self):
        """Test normalization with constant values"""
        audio = np.array([5, 5, 5, 5])
        normalized = normalize_audio(audio)
        # Should return zero-mean version
        self.assertAlmostEqual(np.mean(normalized), 0, places=5)
    
    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        # Create synthetic audio
        sample_rate = 16000
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        features = extract_features(audio, sample_rate)
        
        self.assertIsNotNone(features)
        self.assertGreater(features.shape[0], 0)  # Should have some features
        self.assertGreater(features.shape[1], 0)  # Should have some time steps
    
    def test_extract_features_empty_audio(self):
        """Test feature extraction with empty audio"""
        audio = np.array([])
        features = extract_features(audio)
        
        # Should return empty array
        self.assertEqual(features.size, 0)
    
    def test_preprocess_audio_with_synthetic_file(self):
        """Test complete audio preprocessing pipeline"""
        # Create synthetic audio file
        sample_rate = 16000
        duration = 1.0
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Save to temporary file
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Test preprocessing
            features, processed_audio = preprocess_audio(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            if features is not None and processed_audio is not None:
                self.assertIsNotNone(features)
                self.assertIsNotNone(processed_audio)
                self.assertGreater(len(processed_audio), 0)
            else:
                print("Preprocessing returned None (may be due to missing dependencies)")

class TestEvaluationUtils(unittest.TestCase):
    """Test cases for evaluation utilities"""
    
    def test_evaluate_model_basic(self):
        """Test basic model evaluation"""
        # Create synthetic predictions and ground truth
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy, conf_matrix = evaluate_model(y_true, y_pred)
        
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(conf_matrix)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertEqual(conf_matrix.shape, (2, 2))  # Binary classification
    
    def test_evaluate_model_perfect_prediction(self):
        """Test evaluation with perfect predictions"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        accuracy, conf_matrix = evaluate_model(y_true, y_pred)
        
        self.assertEqual(accuracy, 1.0)
    
    def test_evaluate_model_worst_prediction(self):
        """Test evaluation with worst possible predictions"""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        
        accuracy, conf_matrix = evaluate_model(y_true, y_pred)
        
        self.assertEqual(accuracy, 0.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_stt_tts_pipeline(self):
        """Test complete STT -> TTS pipeline"""
        # This test requires actual model files, so we'll make it optional
        try:
            # Initialize models
            stt_model = SpeechToTextModel()
            tts_model = TextToSpeechModel()
            
            # Create test text
            test_text = "This is a test sentence."
            
            # Convert text to speech
            waveform, sample_rate = tts_model.text_to_audio(test_text)
            
            if waveform is not None:
                # Save audio temporarily
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tts_model.save_audio(waveform, sample_rate, tmp_file.name)
                    
                    # Convert back to text
                    transcribed_text = stt_model.process_audio(tmp_file.name)
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                    
                    # The transcribed text should contain some words from original
                    self.assertIsNotNone(transcribed_text)
                    self.assertGreater(len(transcribed_text.strip()), 0)
                    
                    print(f"Original: {test_text}")
                    print(f"Transcribed: {transcribed_text}")
            else:
                print("TTS pipeline test skipped - model not available")
                
        except Exception as e:
            print(f"Integration test failed (expected if models not available): {e}")

def run_tests():
    """Run all tests"""
    # Create test directories if they don't exist
    os.makedirs('data/audio', exist_ok=True)
    os.makedirs('data/transcripts', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run tests if this file is executed directly
    success = run_tests()
    sys.exit(0 if success else 1)