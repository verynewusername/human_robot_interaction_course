from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from google import genai
from google.genai import types
import random
import os
import time
import json
import re
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# WORD SETS PER TOPIC
# ─────────────────────────────────────────────
TOPICS = {
    "animals": ["dog", "bird", "fish", "cat", "elephant", "rabbit", "horse", "frog", "lion", "turtle", "bear", "duck"],
    "food":    ["apple", "bread", "soup", "cake", "rice", "banana", "cheese", "egg", "milk", "pizza", "carrot", "cookie"],
    "school":  ["pencil", "book", "teacher", "desk", "class", "bag", "pen", "eraser", "chair", "board", "ruler", "homework"],
    "weather": ["rain", "sun", "cloud", "wind", "snow", "storm", "rainbow", "fog", "ice", "thunder", "hail", "breeze"],
}

# How long (seconds) to wait for the child to speak before giving up
LISTEN_TIMEOUT = 15

# ─────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────
chatbot = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

def validate_sentence(target_word, sentence):
    """Ask Gemini to validate the child's sentence. Returns (is_valid, reason, encouragement)."""
    prompt = f"Target word: {target_word}\nChild's sentence: {sentence}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chatbot.models.generate_content(
                model="GEMMA 3 27b",
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

# ─────────────────────────────────────────────
# STT GLOBALS
# ─────────────────────────────────────────────
finish_dialogue = False
query = ""

def asr(frames):
    global finish_dialogue, query
    if frames["data"]["body"]["final"] and not finish_dialogue:
        query = str(frames["data"]["body"]["text"]).strip()
        print("Person said:", query)
        finish_dialogue = True

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@inlineCallbacks
def say(session, text):
    """Speak and print."""
    print("Robot:", text)
    yield session.call("rie.dialogue.say_animated", text=text)

@inlineCallbacks
def listen(session, timeout=LISTEN_TIMEOUT):
    """
    Wait for the child to finish speaking.
    Returns the utterance, or empty string if timeout is reached.
    """
    global finish_dialogue, query
    finish_dialogue = False
    query = ""
    yield session.call("rie.dialogue.stt.stream")

    elapsed = 0.0
    while not finish_dialogue:
        yield sleep(0.5)
        elapsed += 0.5
        if elapsed >= timeout:
            print(f"  [listen] Timeout after {timeout}s")
            yield session.call("rie.dialogue.stt.close")
            yield sleep(0.3)
            finish_dialogue = False
            query = ""
            return ""

    yield session.call("rie.dialogue.stt.close")
    yield sleep(0.5)
    result = query
    finish_dialogue = False
    query = ""
    return result

@inlineCallbacks
def run_game(session, topic):
    """
    Run one full game for the given topic.
    Returns the score.
    """
    word_list = TOPICS[topic].copy()
    random.shuffle(word_list)
    total = len(word_list)
    score = 0

    for i, word in enumerate(word_list, start=1):
        success = yield play_word_round(session, word, i, total)
        if success:
            score += 1
        yield sleep(0.5)

    return score

@inlineCallbacks
def pick_topic(session):
    """
    Let the child choose a topic by speaking its name.
    Returns the chosen topic key (e.g. 'animals').
    """
    topic_list = ", ".join(TOPICS.keys())
    yield say(session, f"Which topic would you like? You can choose: {topic_list}.")
    yield sleep(1)

    while True:
        answer = yield listen(session)
        if not answer:
            yield say(session, f"I did not hear you. Please choose one of: {topic_list}.")
            yield sleep(0.5)
            continue
        for topic in TOPICS:
            if topic in answer.lower():
                yield say(session, f"Awesome! Let's go with {topic}!")
                return topic
        yield say(session, f"Hmm, I did not catch that. Please choose one of: {topic_list}.")
        yield sleep(0.5)

@inlineCallbacks
def play_word_round(session, word, word_number, total_words):
    """
    Play one word round: ask child to form a sentence, validate, retry up to MAX_RETRIES.
    Returns True if the child succeeded, False if they exhausted retries.
    """
    MAX_RETRIES = 4

    yield say(session,
              f"Word {word_number} of {total_words}. "
              f"Your word is: {word}. "
              f"Can you make a sentence using the word '{word}'?")
    yield sleep(1)

    for attempt in range(1, MAX_RETRIES + 1):
        sentence = yield listen(session)

        if not sentence:
            yield say(session, "I did not hear anything. Please try again.")
            yield sleep(0.5)
            continue

        print(f"  Attempt {attempt}: '{sentence}'")
        is_valid, reason, encouragement = validate_sentence(word, sentence)

        if is_valid:
            yield say(session, encouragement)
            yield sleep(1)
            return True
        else:
            if attempt < MAX_RETRIES:
                retries_left = MAX_RETRIES - attempt
                retry_word = "try" if retries_left == 1 else "tries"
                feedback = (
                    f"{reason} "
                    f"You have {retries_left} more {retry_word}. "
                    f"Remember to use the word '{word}' in your sentence. Give it another go!"
                )
            else:
                feedback = (
                    f"{reason} "
                    f"That was a tough one! A good sentence could be: "
                    f"'I like the {word}.' Let's move on!"
                )
            yield say(session, feedback)
            yield sleep(1)

    return False

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
@inlineCallbacks
def main(session, details):
    # set language to English
    yield session.call("rie.dialogue.config.language", lang="en")

    # robot stands up
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    # subscribe to STT once, globally
    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    # ── Welcome ──────────────────────────────
    yield say(session,
              "Hello! I am your language buddy. "
              "We are going to play a fun sentence game together!")
    yield sleep(1)

    # ── Game loop (supports replay) ───────────
    while True:
        chosen_topic = yield pick_topic(session)
        score = yield run_game(session, chosen_topic)
        total = len(TOPICS[chosen_topic])

        yield say(session,
                  f"Amazing work! We finished all {total} words. "
                  f"You made a correct sentence for {score} out of {total} words. "
                  f"You are doing so well! Keep practising and you will be a sentence superstar!")
        yield sleep(2)

        yield say(session, "Would you like to play again with a different topic? Say yes or no.")
        yield sleep(1)
        answer = yield listen(session)

        if "yes" in answer.lower():
            yield say(session, "Wonderful! Let's go again!")
            yield sleep(1)
        else:
            yield say(session, "Okay! Great job today. Goodbye and keep up the amazing work!")
            break

    yield sleep(1)
    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()

# ─────────────────────────────────────────────
# WAMP CONNECTION
# ─────────────────────────────────────────────
wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
        "serializers": ["msgpack"],
        "max_retries": 0
    }],
    realm=os.getenv("WAMP_REALM"),
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])