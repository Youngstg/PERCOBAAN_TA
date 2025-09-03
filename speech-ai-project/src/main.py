from models.speech_to_text import SpeechToTextModel
from models.text_to_speech import TextToSpeechModel
import os

def main():
    # Initialize models
    stt_model = SpeechToTextModel()
    tts_model = TextToSpeechModel()

    # Load pre-trained models
    stt_model.load_model('path/to/speech_to_text_model')
    tts_model.load_model('path/to/text_to_speech_model')

    # Example audio file path
    audio_file_path = os.path.join('data', 'audio', 'example.wav')

    # Process audio to text
    text_output = stt_model.process_audio(audio_file_path)
    print("Transcribed Text:", text_output)

    # Convert text back to speech
    tts_model.convert_text_to_audio(text_output, 'output_audio.wav')
    print("Audio saved as output_audio.wav")

if __name__ == "__main__":
    main()