import subprocess
import requests
import os
import time
from pydub import AudioSegment
from langdetect import detect
from gtts import gTTS
from googletrans import Translator  # Make sure to install googletrans==4.0.0-rc1

# Initialize the translator instance
translator = Translator()

# Load your API keys securely (use environment variables ideally)
ASSEMBLY_API_KEY = "971cb36dd09f4881adfd5c9055e8ccd2"
ELEVENLABS_API_KEY = "sk_dd6328d0fc889397cf646eee26ea5cb3f5b900e4e915d3b7"

def detect_language(text):
    try:
        return detect(text)  # returns 'hi' for Hindi, 'en' for English, etc.
    except Exception as e:
        return "en"  # default to English if detection fails

def get_llama_response(user_prompt: str) -> str:
    try:
        # Explicitly use UTF-8 for encoding and handle errors by replacing undefined characters.
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=user_prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60
        )
        if result.returncode == 0:
            response_text = result.stdout.strip()
            # Translate the output to English so the final response is always in English
            translated_text = translator.translate(response_text, dest='en').text
            return translated_text
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

def transcribe_audio(audio_path):
    headers = {
        "authorization": ASSEMBLY_API_KEY,
        "content-type": "application/json"
    }

    # Upload audio
    with open(audio_path, 'rb') as f:
        upload_response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": ASSEMBLY_API_KEY},
            files={"file": f}
        )
    upload_url = upload_response.json()["upload_url"]

    # Start transcription with Hindi support
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json={"audio_url": upload_url, "language_code": "hi"},  # change to "en" for English if needed
        headers=headers
    )
    transcript_id = response.json()["id"]

    # Poll for completion
    while True:
        polling = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        status = polling.json()
        if status["status"] == "completed":
            return status["text"]
        elif status["status"] == "error":
            return "Transcription failed"
        else:
            time.sleep(2)

def generate_audio_with_elevenlabs(text, output_path="static/llama_response.mp3", voice_id="EXAVITQu4vr4xnSDxMaL"):
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    else:
        return None

def generate_audio_with_gtts(text, output_path="static/llama_response_hi.mp3", lang='hi'):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"gTTS Error: {str(e)}")
        return None

def generate_audio(text, output_path="static/llama_response.mp3"):
    language = detect_language(text)
    print(f"Detected Language: {language}")

    if language == "hi":
        return generate_audio_with_gtts(text, output_path=output_path, lang='hi')
    else:
        return generate_audio_with_elevenlabs(text, output_path=output_path)
