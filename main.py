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
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LOCAL TEST FLAG
# Run with: python main.py --local-test
# ─────────────────────────────────────────────
LOCAL_TEST = "--local-test" in sys.argv

if LOCAL_TEST:
    print("[LOCAL TEST MODE] Robot connection disabled. Using terminal I/O.")
    from twisted.internet.defer import succeed
    def sleep(seconds):
        return succeed(None)
else:
    from autobahn.twisted.util import sleep

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
# GEMINI CLIENT
# ─────────────────────────────────────────────
chatbot = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

def validate_sentence(words, sentence):
    """
    words: str (stages 1 & 2) or list of two str (stage 3).
    Returns (is_valid, reason, encouragement).
    """
    if isinstance(words, list):
        prompt = f"Target words: {words[0]}, {words[1]}\nChild's sentence: {sentence}"
        system_prompt = VALIDATOR_PROMPT_TWO
    else:
        prompt = f"Target word: {words}\nChild's sentence: {sentence}"
        system_prompt = VALIDATOR_PROMPT_ONE

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chatbot.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=system_prompt),
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
# QR GLOBALS
# ─────────────────────────────────────────────
qr_result = None

def on_qr(frames):
    global qr_result
    value = frames["data"]["body"].get("text", "").strip().lower()
    if value and qr_result is None:
        print(f"QR detected: {value}")
        qr_result = value

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

def pick_words(topic, stage, used_words):
    """
    Pick word(s) for the current stage, avoiding repeats where possible.
    Returns a str (stages 1 & 2) or list of two str (stage 3).
    """
    easy = TOPICS[topic]["easy"]
    hard = TOPICS[topic]["hard"]
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
    else:  # stage 3
        pool = [w for w in all_words if w not in used_words]
        if len(pool) < 2:
            pool = all_words
        pair = random.sample(pool, 2)
        used_words.update(pair)
        return pair

STAGE_LABELS = {1: "easy words", 2: "harder words", 3: "two words at once"}

@inlineCallbacks
def play_question(session, words, q_number, total_q, streak_state):
    """
    Ask one question. Returns True if child succeeded.
    """
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
            if streak_state["count"] >= STREAK_THRESHOLD and streak_state["count"] % STREAK_THRESHOLD == 0:
                yield say(session, f"Wow, {streak_state['count']} in a row! You are on fire! {encouragement}")
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
                          f"{reason} That was a tough one! A good sentence could be: {example} Let's move on!")
            yield sleep(1)

    return False

@inlineCallbacks
def run_game(session, topic):
    """
    Always 6 questions across 3 rounds of 2.
    Stage goes up if 2/2 correct, stays the same otherwise.
    Max stage is 3 (two-word) — never goes higher.
    """
    streak_state = {"count": 0}
    used_words = set()
    total_score = 0
    stage = 1
    TOTAL_ROUNDS = 3

    stage_labels = {1: "easy words", 2: "harder words", 3: "two words at once"}
    up_messages  = {1: "Nice work! Let's go a bit harder.",
                    2: "Amazing! Now for the ultimate challenge!"}

    for round_num in range(1, TOTAL_ROUNDS + 1):
        yield say(session, f"Round {round_num} of {TOTAL_ROUNDS} — {stage_labels[stage]}!")
        yield sleep(1)

        round_score = 0
        for q in range(2):
            q_number = (round_num - 1) * 2 + q + 1
            words = pick_words(topic, stage, used_words)
            success = yield play_question(session, words, q_number, 6, streak_state)
            if success:
                round_score += 1
                total_score += 1
            yield sleep(0.5)

        # Adjust stage for next round (not after the last round)
        if round_num < TOTAL_ROUNDS:
            if round_score == 2 and stage < 3:
                stage += 1
                yield say(session, up_messages.get(stage - 1, "Great job!"))
                yield sleep(1)
            # else: stay at same stage, no comment needed

    return total_score, 6

@inlineCallbacks
def pick_topic(session):
    """
    Let the child choose a topic by showing a QR card (on robot)
    or by typing/speaking (local test).
    QR card should encode the topic name: 'animals', 'food', 'school', or 'weather'.
    """
    global qr_result
    topic_list = ", ".join(TOPICS.keys())

    if LOCAL_TEST:
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
    else:
        yield say(session,
                  f"Show me a topic card to get started! "
                  f"You can choose: {topic_list}.")
        yield session.call("rie.vision.qrcode.stream")

        while True:
            qr_result = None
            elapsed = 0.0
            while qr_result is None:
                yield sleep(0.5)
                elapsed += 0.5
                if elapsed > 30:
                    yield say(session, f"I did not see a card. Please show me one of: {topic_list}.")
                    elapsed = 0.0

            topic = qr_result.lower().strip()
            qr_result = None

            if topic in TOPICS:
                yield session.call("rie.vision.qrcode.close")
                yield say(session, f"Awesome! Let's go with {topic}!")
                return topic
            else:
                yield say(session, f"I do not recognise that card. Please try one of: {topic_list}.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
@inlineCallbacks
def main(session, details):
    if not LOCAL_TEST:
        yield session.call("rie.dialogue.config.language", lang="en")
        yield session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield session.subscribe(asr, "rie.dialogue.stt.stream")
        yield session.subscribe(on_qr, "rie.vision.qrcode")

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

        if "yes" in answer.lower():
            yield say(session, "Wonderful! Let's go again!")
            yield sleep(1)
        else:
            yield say(session, "Okay! Great job today. Goodbye and keep up the amazing work!")
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