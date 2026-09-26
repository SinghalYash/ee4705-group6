"""
Captures a spoken instruction from the microphone and transcribes it to text.

Uses SpeechRecognition's built-in free Google Web Speech API (no key needed,
requires internet). Swap recognize_google for recognize_whisper_api if your
team prefers Whisper (needs OPENAI_API_KEY and generally more accurate).
"""

import speech_recognition as sr


def listen_and_transcribe(timeout: int = 5, phrase_time_limit: int = 8):
    """Listen on the default microphone and return the transcribed text, or None."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... speak your instruction now.")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("No speech detected in time.")
            return None

    print("Transcribing...")
    try:
        text = recognizer.recognize_google(audio)
        print(f'Heard: "{text}"')
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None


if __name__ == "__main__":
    result = listen_and_transcribe()
    print("Final transcription:", result)
