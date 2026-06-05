from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from google import genai
from google.genai import types
import random
import os
import time
import json
import re
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# FLAGS
# python main.py                → Ollama local (default)
# python main.py --gemini       → Gemini/Gemma cloud API
# python main.py --local-test   → terminal I/O, no robot
# ─────────────────────────────────────────────
LOCAL_TEST = "--local-test" in sys.argv
USE_GEMINI = "--gemini"     in sys.argv

if LOCAL_TEST:
    print("[LOCAL TEST MODE] Robot connection disabled. Using terminal I/O.")
    from twisted.internet.defer import succeed
    def sleep(seconds):
        return succeed(None)
else:
    from autobahn.twisted.util import sleep

# ─────────────────────────────────────────────
# WAMP REALM GUARD
# ─────────────────────────────────────────────
if not LOCAL_TEST:
    _realm = os.getenv("WAMP_REALM", "").strip()
    if not _realm:
        print(
            "\n[FATAL] WAMP_REALM is not set or is empty in your .env file.\n"
            "  Add a line like:  WAMP_REALM=rie.your_realm_here\n"
            "  Then restart the program.\n"
        )
        sys.exit(1)

# ─────────────────────────────────────────────
# OLLAMA CONFIG
# ─────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434"
OLLAMA_API  = f"{OLLAMA_URL}/api/chat"
OLLAMA_TAGS = f"{OLLAMA_URL}/api/tags"

def _detect_ollama_model():
    try:
        resp = requests.get(OLLAMA_TAGS, timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        raise RuntimeError(
            f"[Ollama] Cannot reach Ollama at {OLLAMA_URL}. "
            f"Is it running? (`ollama serve`)  Error: {e}"
        )
    if not models:
        raise RuntimeError(
            "[Ollama] Ollama is running but has no models pulled. "
            "Run: ollama pull gemma3"
        )
    gemma_models = [m for m in models if "gemma" in m.lower()]
    chosen = gemma_models[0] if gemma_models else models[0]
    print(f"[Ollama] Auto-detected model: {chosen}")
    return chosen

if not USE_GEMINI:
    OLLAMA_MODEL = _detect_ollama_model()

# ─────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"

if USE_GEMINI:
    chatbot = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print(f"[Gemini] Using cloud model: {GEMINI_MODEL}")

# ─────────────────────────────────────────────
# GAME SETTINGS
# ─────────────────────────────────────────────
LISTEN_TIMEOUT   = 15
STREAK_THRESHOLD = 3

# ─────────────────────────────────────────────
# WORD SETS PER TOPIC
# ─────────────────────────────────────────────
TOPICS = {
    "animals": {
        "easy": ["dog", "cat", "fish", "bird", "duck", "bear"],
        "hard": ["elephant", "rabbit", "horse", "frog", "lion", "turtle"],
    },
    "food": {
        "easy": ["apple", "egg", "milk", "cake", "rice", "bread"],
        "hard": ["banana", "cheese", "soup", "pizza", "carrot", "cookie"],
    },
    "school": {
        "easy": ["pen", "bag", "book", "desk", "chair", "class"],
        "hard": ["pencil", "eraser", "teacher", "board", "ruler", "homework"],
    },
    "weather": {
        "easy": ["sun", "rain", "wind", "snow", "fog", "ice"],
        "hard": ["cloud", "storm", "rainbow", "thunder", "hail", "breeze"],
    },
}

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
VALIDATOR_PROMPT_ONE = """
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
If valid is false, reason must explain simply what is wrong.
Keep encouragement warm, short and suitable for a child.
"""

VALIDATOR_PROMPT_TWO = """
You are a friendly speech and language therapist helping a child with
Developmental Language Disorder (DLD).

Your job is to evaluate whether the child's sentence:
1. Is grammatically valid (has at least a subject and a verb)
2. Makes logical sense
3. Correctly uses BOTH given target words in the same sentence

You will receive input in this format:
Target words: <word1>, <word2>
Child's sentence: <sentence>

Respond ONLY in this exact JSON format (no extra text):
{
  "valid": true or false,
  "reason": "short explanation if invalid, empty string if valid",
  "encouragement": "a short, warm, child-friendly message"
}

If valid is true, reason must be an empty string.
If valid is false, reason must explain simply what is wrong.
Keep encouragement warm, short and suitable for a child.
"""

# ─────────────────────────────────────────────
# VALIDATION — Ollama
# ─────────────────────────────────────────────
def _validate_ollama(system_prompt, user_prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

# ─────────────────────────────────────────────
# VALIDATION — Gemini
# ─────────────────────────────────────────────
def _validate_gemini(system_prompt, user_prompt):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chatbot.models.generate_content(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=[user_prompt]
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                match = re.search(r"retry in ([\d.]+)s", error_str)
                delay = float(match.group(1)) if match else min(2 ** attempt, 30)
                print(f"  Rate limited. Retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Gemini: max retries exceeded")

# ─────────────────────────────────────────────
# UNIFIED VALIDATE
# ─────────────────────────────────────────────
def validate_sentence(words, sentence):
    if isinstance(words, list):
        user_prompt   = f"Target words: {words[0]}, {words[1]}\nChild's sentence: {sentence}"
        system_prompt = VALIDATOR_PROMPT_TWO
    else:
        user_prompt   = f"Target word: {words}\nChild's sentence: {sentence}"
        system_prompt = VALIDATOR_PROMPT_ONE

    try:
        data = _validate_gemini(system_prompt, user_prompt) if USE_GEMINI \
               else _validate_ollama(system_prompt, user_prompt)
        return (
            data["valid"],
            data.get("reason", ""),
            data.get("encouragement", "Well done!")
        )
    except Exception as e:
        print(f"  [validate_sentence] Error: {e}")
        return False, "I had trouble checking that. Please try again.", "Give it another go!"

# ─────────────────────────────────────────────
# STT GLOBALS
# ─────────────────────────────────────────────
finish_dialogue = False
query = ""

def asr(frames):
    global finish_dialogue, query
    try:
        body = frames["data"]["body"]
        if body.get("final") and not finish_dialogue:
            query = str(body.get("text", "")).strip()
            print("Person said:", query)
            finish_dialogue = True
    except (KeyError, TypeError) as e:
        print(f"  [asr] Unexpected frame format: {e}")

# ─────────────────────────────────────────────
# GRAB FRAME — robot camera via WAMP
# Confirmed payload:
#   list → [0] → 'data' → 'body.head.eyes' → raw JPEG bytes
# ─────────────────────────────────────────────
@inlineCallbacks
def grab_frame(session):
    try:
        result = yield session.call("rom.sensor.sight.read")

        if not isinstance(result, list) or len(result) == 0:
            print(f"  [Sight] Unexpected result type: {type(result)}")
            return None

        img_bytes = result[0].get("data", {}).get("body.head.eyes")

        if isinstance(img_bytes, (bytes, bytearray)):
            return bytes(img_bytes)

        print(f"  [Sight] 'body.head.eyes' missing or wrong type: "
              f"{type(img_bytes)}")
        return None

    except Exception as e:
        print(f"  [Sight] rom.sensor.sight.read error: {e}")
        return None

# ─────────────────────────────────────────────
# QR DECODE — OpenCV (no system library needed on macOS)
# ─────────────────────────────────────────────
def decode_qr(jpeg_bytes):
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img)
        if data:
            return data.strip().lower()
        return None

    except ImportError:
        print("  [QR] opencv-python not installed. Run: pip install opencv-python")
        return None
    except Exception as e:
        print(f"  [QR] Decode error: {e}")
        return None

# ─────────────────────────────────────────────
# CELEBRATE GESTURE
# Raises both arms up, pauses, then returns to rest.
# Uses rom.actuator.motor.write — confirmed in API dump.
#
# Alpha Zero motor IDs (standard mapping):
#   RightShoulderPitch : controls right arm up/down  (positive = up)
#   LeftShoulderPitch  : controls left arm up/down   (negative = up)
#
# Values are in degrees. The robot's neutral stand pose
# has arms roughly at 0. We raise to ~120° then return.
# ─────────────────────────────────────────────
@inlineCallbacks
def celebrate(session):
    """
    Celebration gesture: both arms shoot up then return to neutral.
    Motor values in RADIANS. Confirmed range: [-2.5943, 1.5943] rad.
      right arm up → -1.57 rad  (just under the -2.59 limit, raises forward)
      left arm up  →  1.57 rad  (just under the  1.59 limit)
    """
    if LOCAL_TEST:
        print("[Gesture] celebrate (skipped in local-test mode)")
        return

    print("[Gesture] celebrate — arms up!")
    try:
        # ── Arms UP ───────────────────────────────────────────────────
        yield session.call(
            "rom.actuator.motor.write",
            frames=[{
                "time": 400,
                "data": {
                    "body.arms.right.upper.pitch": -1.57,
                    "body.arms.left.upper.pitch":   1.57,
                }
            }]
        )
        yield sleep(0.6)

        # ── Arms DOWN — back to neutral ───────────────────────────────
        yield session.call(
            "rom.actuator.motor.write",
            frames=[{
                "time": 400,
                "data": {
                    "body.arms.right.upper.pitch": 0.0,
                    "body.arms.left.upper.pitch":  0.0,
                }
            }]
        )
        yield sleep(0.5)

    except Exception as e:
        print(f"  [Gesture] Motor write failed: {e}")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@inlineCallbacks
def say(session, text):
    print(f"Robot: {text}")
    if not LOCAL_TEST:
        yield session.call("rie.dialogue.say_animated", text=text)

@inlineCallbacks
def listen(session, timeout=LISTEN_TIMEOUT):
    if LOCAL_TEST:
        utterance = input("You: ").strip()
        return utterance

    global finish_dialogue, query
    finish_dialogue = False
    query = ""

    try:
        yield session.call("rie.dialogue.stt.stream")
    except Exception as e:
        print(f"  [listen] Could not start STT stream: {e}")
        return ""

    elapsed = 0.0
    while not finish_dialogue:
        yield sleep(0.5)
        elapsed += 0.5
        if elapsed >= timeout:
            print(f"  [listen] Timeout after {timeout}s")
            try:
                yield session.call("rie.dialogue.stt.close")
            except Exception:
                pass
            yield sleep(0.3)
            finish_dialogue = False
            query = ""
            return ""

    try:
        yield session.call("rie.dialogue.stt.close")
    except Exception:
        pass
    yield sleep(0.5)
    result = query
    finish_dialogue = False
    query = ""
    return result

def pick_words(topic, stage, used_words):
    easy      = TOPICS[topic]["easy"]
    hard      = TOPICS[topic]["hard"]
    all_words = easy + hard

    if stage == 1:
        pool = [w for w in easy if w not in used_words] or easy
        word = random.choice(pool)
        used_words.add(word)
        return word
    elif stage == 2:
        pool = [w for w in hard if w not in used_words] or hard
        word = random.choice(pool)
        used_words.add(word)
        return word
    else:
        pool = [w for w in all_words if w not in used_words]
        if len(pool) < 2:
            pool = all_words
        pair = random.sample(pool, 2)
        used_words.update(pair)
        return pair

@inlineCallbacks
def play_question(session, words, q_number, total_q, streak_state):
    MAX_RETRIES = 4

    if isinstance(words, list):
        word_display = f"'{words[0]}' and '{words[1]}'"
        task = f"Can you make one sentence using both words: {word_display}?"
    else:
        word_display = f"'{words}'"
        task = f"Can you make a sentence using the word {word_display}?"

    yield say(session, f"Question {q_number} of {total_q}. {task}")
    yield sleep(1)

    for attempt in range(1, MAX_RETRIES + 1):
        sentence = yield listen(session)

        if not sentence:
            yield say(session, "I did not hear anything. Please try again.")
            yield sleep(0.5)
            continue

        print(f"  Attempt {attempt}: '{sentence}'")
        is_valid, reason, encouragement = validate_sentence(words, sentence)

        if is_valid:
            streak_state["count"] += 1

            # ── Celebrate with gesture + speech simultaneously ────────
            # celebrate() moves the arms; say() speaks — both yielded
            # sequentially so arms go up while the robot talks.
            yield celebrate(session)

            if (streak_state["count"] >= STREAK_THRESHOLD
                    and streak_state["count"] % STREAK_THRESHOLD == 0):
                yield say(session,
                          f"Wow, {streak_state['count']} in a row! "
                          f"You are on fire! {encouragement}")
            else:
                yield say(session, encouragement)

            yield sleep(1)
            return True
        else:
            streak_state["count"] = 0
            if attempt < MAX_RETRIES:
                retries_left = MAX_RETRIES - attempt
                retry_word = "try" if retries_left == 1 else "tries"
                yield say(session,
                          f"{reason} You have {retries_left} more {retry_word}. "
                          f"Remember to use {word_display}. Give it another go!")
            else:
                example = (f"'I see a {words[0]} in the {words[1]}.'"
                           if isinstance(words, list) else f"'I like the {words}.'")
                yield say(session,
                          f"{reason} That was a tough one! "
                          f"A good sentence could be: {example} Let's move on!")
            yield sleep(1)

    return False

@inlineCallbacks
def run_game(session, topic):
    streak_state = {"count": 0}
    used_words   = set()
    total_score  = 0
    stage        = 1
    TOTAL_ROUNDS = 3

    stage_labels = {1: "easy words", 2: "harder words", 3: "two words at once"}
    up_messages  = {1: "Nice work! Let's go a bit harder.",
                    2: "Amazing! Now for the ultimate challenge!"}

    for round_num in range(1, TOTAL_ROUNDS + 1):
        yield say(session,
                  f"Round {round_num} of {TOTAL_ROUNDS} — {stage_labels[stage]}!")
        yield sleep(1)

        round_score = 0
        for q in range(2):
            q_number = (round_num - 1) * 2 + q + 1
            words    = pick_words(topic, stage, used_words)
            success  = yield play_question(session, words, q_number, 6, streak_state)
            if success:
                round_score += 1
                total_score += 1
            yield sleep(0.5)

        if round_num < TOTAL_ROUNDS:
            if round_score == 2 and stage < 3:
                stage += 1
                yield say(session, up_messages.get(stage - 1, "Great job!"))
                yield sleep(1)

    return total_score, 6

# ─────────────────────────────────────────────
# TOPIC SELECTION — snapshot QR loop
# ─────────────────────────────────────────────
@inlineCallbacks
def pick_topic(session):
    topic_list = ", ".join(TOPICS.keys())

    if LOCAL_TEST:
        yield say(session,
                  f"Which topic would you like? You can choose: {topic_list}.")
        yield sleep(1)
        while True:
            answer = yield listen(session)
            if not answer:
                yield say(session,
                          f"I did not hear you. Please choose one of: {topic_list}.")
                yield sleep(0.5)
                continue
            for topic in TOPICS:
                if topic in answer.lower():
                    yield say(session, f"Awesome! Let's go with {topic}!")
                    return topic
            yield say(session,
                      f"Hmm, I did not catch that. Please choose one of: {topic_list}.")
            yield sleep(0.5)
        return

    yield say(session,
              f"Show me a topic card! "
              f"You can choose: {topic_list}.")

    try:
        yield session.call("rom.sensor.sight.stream")
        print("[Sight] Camera stream started.")
    except Exception as e:
        print(f"[Sight] Could not start stream (continuing anyway): {e}")

    yield sleep(0.5)

    elapsed_no_card = 0
    print("[Sight] Scanning for QR — hold card in front of robot camera...")

    while True:
        raw = yield grab_frame(session)

        if raw is None:
            elapsed_no_card += 1
        else:
            text = decode_qr(raw)
            if text:
                print(f"[QR] Decoded: '{text}'")
                if text in TOPICS:
                    try:
                        yield session.call("rom.sensor.sight.close")
                        print("[Sight] Camera closed.")
                    except Exception:
                        pass
                    yield say(session, f"Awesome! Let's go with {text}!")
                    return text
                else:
                    print(f"  [QR] '{text}' not a valid topic, ignoring.")
                    elapsed_no_card = 0
            else:
                elapsed_no_card += 1

        if elapsed_no_card >= 30:
            yield say(session,
                      f"I cannot see a card yet. "
                      f"Please hold one closer to my eyes: {topic_list}.")
            elapsed_no_card = 0

        yield sleep(1)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
@inlineCallbacks
def main(session, details):
    if not LOCAL_TEST:
        yield session.call("rie.dialogue.config.language", lang="en")
        yield session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield session.subscribe(asr, "rie.dialogue.stt.stream")
        print("[WAMP] STT subscription registered.")
        yield sleep(0.5)

    yield say(session,
              "Hello! I am your language buddy. "
              "We are going to play a sentence game. "
              "I will adjust the difficulty based on how you do. Let's go!")
    yield sleep(1)

    while True:
        chosen_topic = yield pick_topic(session)
        score, total = yield run_game(session, chosen_topic)

        yield say(session,
                  f"Amazing work! You got {score} out of {total} correct. "
                  f"You are doing so well! Keep practising!")
        yield sleep(2)

        yield say(session, "Would you like to play again? Say yes or no.")
        yield sleep(1)
        answer = yield listen(session)

        if answer and "yes" in answer.lower():
            yield say(session, "Wonderful! Let's go again!")
            yield sleep(1)
        else:
            yield say(session,
                      "Okay! Great job today. Goodbye and keep up the amazing work!")
            break

    if not LOCAL_TEST:
        yield sleep(1)
        yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
        session.leave()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if LOCAL_TEST:
    from twisted.internet import reactor
    from twisted.internet.defer import inlineCallbacks as ib

    @ib
    def _run_local():
        yield main(None, None)
        reactor.stop()

    reactor.callLater(0, _run_local)
    reactor.run()
else:
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
