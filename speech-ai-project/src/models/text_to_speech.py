import torch
import torchaudio

class TextToSpeechModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # Load the pre-trained model from the specified path
        model = torch.jit.load(model_path)
        model.eval()
        return model

    def text_to_audio(self, text):
        # Convert text input into audio
        waveform, sample_rate = self.model(text)
        return waveform, sample_rate

    def save_audio(self, waveform, sample_rate, output_path):
        # Save the generated audio file
        torchaudio.save(output_path, waveform.unsqueeze(0), sample_rate)