from openai import OpenAI


import os
API_KEY = os.getenv("API_KEY")


def speech_to_text(audio):

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio
    )

    return transcription.text


def text_to_speech(text):

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    return speech