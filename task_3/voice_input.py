"""
Captures a spoken instruction from the microphone and transcribes it to text.

Uses SpeechRecognition's built-in free Google Web Speech API (no key needed,
requires internet). Swap recognize_google for recognize_whisper_api if your
team prefers Whisper (needs OPENAI_API_KEY and generally more accurate).

The raw transcription is then run through a light domain-vocabulary
correction pass (see correct_transcription() below) to fix the kind of
misheard words that came up in testing -- most notably "red" being
transcribed as "read", "read", or "reed" -- before the text is ever
handed to the Task 3 planner.
"""

import difflib

import speech_recognition as sr


# ==========================================================
# DOMAIN VOCABULARY CORRECTION
# ==========================================================
# The planner only ever needs to understand a small, fixed vocabulary
# of colors, object/region words, and action verbs. Free speech-to-text
# has no idea that vocabulary is constrained, so it sometimes returns a
# common English word that merely *sounds* like the domain word we
# actually needed (e.g. "read" instead of "red"). Rather than trying to
# fix this by tuning the recognizer, we nudge each transcribed word
# towards the closest domain word whenever it's a close-enough sound-
# alike, and leave everything else untouched.

DOMAIN_VOCABULARY = [
    # colors / regions
    "red", "blue", "green", "grey", "gray", "area", "zone", "region",
    "marker", "target",
    # objects
    "box", "cube", "stone", "rock", "cylinder", "sphere", "ball",
    # verbs / skills
    "find", "search", "look", "approach", "reach", "grasp", "grab",
    "pick", "place", "put", "move", "carry", "drop", "stop",
    "nearby", "near",
]

# A few common, unambiguous mishearings, applied before the general
# fuzzy pass so they're never missed. Kept short and deliberately
# excludes anything that's also a common, unrelated English word
# (e.g. "great" is not mapped to "grey" here, even though they sound
# similar, because "great" is too likely to appear for other reasons).
KNOWN_MISHEARDS = {
    "read": "red",   # the main mishearing reported in testing
    "reed": "red",
    "bread": "red",
    "led": "red",
    "blew": "blue",
}


def correct_word(word: str) -> str:
    lower = word.lower()

    if lower in KNOWN_MISHEARDS:
        corrected = KNOWN_MISHEARDS[lower]

    elif lower in DOMAIN_VOCABULARY:
        # Already a known word -- leave it alone.
        return word

    else:
        matches = difflib.get_close_matches(
            lower, DOMAIN_VOCABULARY, n=1, cutoff=0.75
        )
        if not matches:
            return word
        corrected = matches[0]

    # Preserve capitalisation of the first letter, matching the input.
    if word[:1].isupper():
        corrected = corrected.capitalize()

    return corrected


def correct_transcription(text: str) -> str:
    """
    Nudge each word of a transcription towards the closest word in
    DOMAIN_VOCABULARY when it's a close sound-alike (e.g. "read" ->
    "red"). Words that are not close to anything in the domain
    vocabulary, including ordinary sentence words like "the" or "and",
    are left untouched.
    """

    words = text.split()
    corrected_words = [correct_word(w) for w in words]
    return " ".join(corrected_words)


# ==========================================================
# SPEECH CAPTURE
# ==========================================================

def listen_and_transcribe(
    timeout: int = 6,
    phrase_time_limit: int = 12,
    apply_correction: bool = True,
):
    """
    Listen on the default microphone and return the transcribed text,
    or None.

    timeout: seconds to wait for speech to start before giving up.
    phrase_time_limit: max seconds of speech to record once started.
        Raised from the original 8s -> 12s, since longer instructions
        ("please pick up the nearby stone and place it on the red
        marker") were sometimes getting cut off mid-sentence.
    apply_correction: if True (default), run the domain-vocabulary
        correction pass on the result before returning it.
    """

    recognizer = sr.Recognizer()

    # Give the speaker a bit more natural pause time before the
    # recognizer decides the phrase has ended.
    recognizer.pause_threshold = 1.0

    with sr.Microphone() as source:
        print("Listening... speak your instruction now.")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
        except sr.WaitTimeoutError:
            print("No speech detected in time.")
            return None

    print("Transcribing...")
    try:
        text = recognizer.recognize_google(audio)
        print(f'Heard: "{text}"')

        if apply_correction:
            corrected = correct_transcription(text)
            if corrected != text:
                print(f'Corrected to: "{corrected}"')
            return corrected

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