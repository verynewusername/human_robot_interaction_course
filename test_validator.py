"""
Standalone test script for the Gemini sentence validator.
Run this to test validation without needing the robot connection.

Model: gemma-4-27b-it  (free via Google AI Studio)
  - Free tier: 15 requests/min, 1,500 requests/day
  - Get your key at: https://aistudio.google.com/apikey
"""

import os
import json
import re
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

chatbot = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Model to use ──────────────────────────────────────────────────────────────
# gemma-4-27b-it  → dense, instruction-tuned, free on AI Studio
# gemma-4-26b-a4b-it → MoE variant, also free (slightly faster)
GEMMA_MODEL = "gemma-4-31b-it"
# ─────────────────────────────────────────────────────────────────────────────

VALIDATOR_PROMPT = """
You are a friendly speech and language therapist helping a child with 
Developmental Language Disorder (DLD). 

Your job is to evaluate whether the child's sentence:
1. Is grammatically valid (has at least a subject and a verb)
2. Makes logical sense
3. Correctly uses the given target word

You will receive input in this format:
Target word: <word>
Child's sentence: <sentence>

Respond ONLY in this exact JSON format (no extra text):
{
  "valid": true or false,
  "reason": "short explanation if invalid, empty string if valid",
  "encouragement": "a short, warm, child-friendly message"
}

If valid is true, reason must be an empty string.
If valid is false, reason must explain simply what is wrong 
(e.g. 'Your sentence is missing a verb.' or 
'The word dog was not used in the sentence.').
Keep encouragement warm, short and suitable for a child.
"""


def validate_sentence_local(target_word, sentence):
    """
    Simple offline rule-based validator.
    Returns (is_valid, reason, encouragement) without using the API.
    """
    sentence_lower = sentence.lower().strip()
    word_lower = target_word.lower().strip()

    if word_lower not in sentence_lower.split():
        return (False,
                f"The word '{target_word}' was not used in your sentence.",
                "Try again and make sure to include the word!")

    words = sentence_lower.split()
    if len(words) < 2:
        return (False,
                "Your sentence is too short. Try making a longer sentence!",
                "You can do it! Add more words!")

    if not sentence[0].isupper():
        return (False,
                "Remember to start your sentence with a capital letter!",
                "Almost there! Just fix the capital letter.")

    if sentence[-1] not in ".!?":
        return (False,
                "Don't forget punctuation at the end!",
                "So close! Just add a period or exclamation mark.")

    return (True,
            "",
            f"Great sentence using '{target_word}'! Well done!")


def validate_sentence(target_word, sentence, use_local=False):
    """
    Validate a sentence. Uses Gemma 4 API by default.
    Set use_local=True for offline rule-based validation.
    """
    if use_local:
        return validate_sentence_local(target_word, sentence)

    prompt = f"Target word: {target_word}\nChild's sentence: {sentence}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chatbot.models.generate_content(
                model=GEMMA_MODEL,                          # ← Gemma 4 here
                config=types.GenerateContentConfig(
                    system_instruction=VALIDATOR_PROMPT
                ),
                contents=[prompt]
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return data["valid"], data.get("reason", ""), data.get("encouragement", "Well done!")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                match = re.search(r"retry in ([\d.]+)s", error_str)
                delay = float(match.group(1)) if match else min(2 ** attempt, 30)
                print(f"  Rate limited. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise


def test_cases(use_local=False):
    """Run a set of test sentences and print results."""
    mode = "LOCAL (offline)" if use_local else f"GEMMA API [{GEMMA_MODEL}]"
    tests = [
        ("dog",   "I love my dog"),
        ("dog",   "I love to open my dog"),
        ("fish",  "I caught a fish"),
        ("fish",  "fish water"),
        ("book",  "I read a book"),
        ("book",  "book"),
        ("rain",  "The rain is falling outside"),
        ("rain",  "rain rain"),
        ("apple", "I ate a red apple for lunch"),
        ("apple", "apple red"),
    ]

    print("=" * 60)
    print(f"  Sentence Validator — Test Runs  [{mode}]")
    print("=" * 60)

    for word, sentence in tests:
        print(f"\n  Target word: '{word}'")
        print(f"  Sentence:    '{sentence}'")
        try:
            valid, reason, encouragement = validate_sentence(word, sentence, use_local=use_local)
            status = "✅ VALID" if valid else "❌ INVALID"
            print(f"  Result:      {status}")
            if reason:
                print(f"  Reason:      {reason}")
            print(f"  Encourage:   {encouragement}")
        except Exception as e:
            print(f"  ERROR:       {e}")
        print(f"  {'─' * 56}")

    print("\n  Done!")


def interactive_mode(use_local=False):
    """Test your own sentences interactively."""
    mode = "LOCAL (offline)" if use_local else f"GEMMA API [{GEMMA_MODEL}]"
    print("=" * 60)
    print(f"  Interactive Mode  [{mode}]")
    print("  Enter a target word and sentence to test.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        word = input("\n  Target word: ").strip()
        if word.lower() == "quit":
            break
        sentence = input("  Sentence:    ").strip()
        if sentence.lower() == "quit":
            break

        if not word or not sentence:
            print("  Please enter both a word and a sentence.")
            continue

        try:
            valid, reason, encouragement = validate_sentence(word, sentence, use_local=use_local)
            status = "✅ VALID" if valid else "❌ INVALID"
            print(f"  Result:      {status}")
            if reason:
                print(f"  Reason:      {reason}")
            print(f"  Encourage:   {encouragement}")
        except Exception as e:
            print(f"  ERROR:       {e}")


if __name__ == "__main__":
    import sys

    # Default is now Gemma API; pass --local to use offline rule-based validation
    use_local = "--local" in sys.argv

    if "--interactive" in sys.argv:
        interactive_mode(use_local=use_local)
    else:
        test_cases(use_local=use_local)
