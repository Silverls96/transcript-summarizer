import unittest
from unittest.mock import patch, MagicMock
from app import get_llm_response, transcribe, process_audio_file
import os
import whisper
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize console
console = Console()

# Load Whisper model
stt = whisper.load_model("turbo")

# Get API key and model
api_key = os.environ.get("OPENROUTER_API_KEY")
api_model = os.environ.get("LLM_MODEL")

class TestAudioProcessing(unittest.TestCase):

    def test_get_llm_response(self):
        # Mock the completion function to return a dummy response
        with patch('litellm.completion') as mock_completion:
            mock_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='Dummy response'))])
            text = "Test input text"
            response = get_llm_response(text)
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)

    def test_transcribe(self):
        # Use a sample audio file for testing
        audio_file_path = "sample/Podcast-Terpendek-di-Dunia.mp3"
        if os.path.exists(audio_file_path):
            transcription = transcribe(audio_file_path)
            self.assertIsNotNone(transcription)
            self.assertIsInstance(transcription, str)
        else:
            self.skipTest("No sample audio file found.")

    def test_process_audio_file(self):
        # Use a sample audio file for testing
        audio_file_path = "sample/Podcast-Terpendek-di-Dunia.mp3"
        if os.path.exists(audio_file_path):
            process_audio_file(audio_file_path)
            # Check if output files are created
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            transcription_path = os.path.join("transcribe", f"{base_name}_transcription.txt")
            response_path = os.path.join("response", f"{base_name}_response.txt")
            self.assertTrue(os.path.exists(transcription_path))
            self.assertTrue(os.path.exists(response_path))
        else:
            self.skipTest("No sample audio file found.")

if __name__ == '__main__':
    unittest.main()
