from openai import OpenAI

client = OpenAI(api_key="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")

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