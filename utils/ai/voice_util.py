import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('ELEVENLABS_API_KEY')
# ElevenLabs API endpoint
URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Your ElevenLabs API key

# The ID of the voice you want to use
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"

def text_to_speech(text, output_path="media/voice_test.mp3"):
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(URL.format(voice_id=VOICE_ID), json=data, headers=headers)

    if response.status_code == 200:
        with open(output_path, "wb") as audio_file:
            audio_file.write(response.content)
        print(f"Audio saved to {output_path}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Example usage
if __name__ == "__main__":
    text = "Hello, this is a test of the ElevenLabs text-to-speech API."
    text_to_speech(text)