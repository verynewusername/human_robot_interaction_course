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
# WORD SETS PER TOPIC  (easy → hard order)
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

# How long (seconds) to wait for the child to speak before giving up
LISTEN_TIMEOUT = 15

STREAK_THRESHOLD = 3   # consecutive correct answers to trigger celebration

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
If valid is false, reason must explain simply what is wrong
(e.g. 'You used dog but forgot to use rain.' or
'Your sentence is missing a verb.').
Keep encouragement warm, short and suitable for a child.
"""

def validate_sentence(words, sentence):
    """
    Ask Gemini to validate the child's sentence.
    words: a single word (str) or a list of two words.
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
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
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
def pick_topic(session):
    """Let the child choose a topic. Returns the chosen topic key."""
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
def play_round(session, words, round_number, total_rounds, streak_state):
    """
    Play one round. words is a str (stages 1&2) or list of two str (stage 3).
    streak_state is a dict with key 'count' so we can mutate it across rounds.
    Returns True if the child succeeded.
    """
    MAX_RETRIES = 4

    # Build the prompt text depending on stage
    if isinstance(words, list):
        word_display = f"'{words[0]}' and '{words[1]}'"
        task = f"Can you make one sentence using both words: {word_display}?"
    else:
        word_display = f"'{words}'"
        task = f"Can you make a sentence using the word {word_display}?"

    yield say(session, f"Round {round_number} of {total_rounds}. Your word is: {word_display}. {task}")
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
                feedback = (
                    f"{reason} "
                    f"You have {retries_left} more {retry_word}. "
                    f"Remember to use {word_display}. Give it another go!"
                )
            else:
                if isinstance(words, list):
                    example = f"'I see a {words[0]} in the {words[1]}.'"
                else:
                    example = f"'I like the {words}.'"
                feedback = (
                    f"{reason} "
                    f"That was a tough one! A good sentence could be: {example} Let's move on!"
                )
            yield say(session, feedback)
            yield sleep(1)

    return False

@inlineCallbacks
def run_game(session, topic):
    """
    Run one full 3-stage game for the given topic.
    Stage 1: easy words (one word per round)
    Stage 2: hard words (one word per round)
    Stage 3: two random words from the full pool per round (4 rounds)
    Returns total score.
    """
    easy_words = TOPICS[topic]["easy"].copy()
    hard_words = TOPICS[topic]["hard"].copy()
    all_words  = easy_words + hard_words

    random.shuffle(easy_words)
    random.shuffle(hard_words)

    # Stage 3: 4 pairs of random words (no pair repeats)
    stage3_pairs = []
    pool = all_words.copy()
    random.shuffle(pool)
    for i in range(0, 8, 2):          # 4 pairs
        stage3_pairs.append([pool[i], pool[i + 1]])

    rounds = (
        [(w, 1) for w in easy_words] +
        [(w, 2) for w in hard_words] +
        [(p, 3) for p in stage3_pairs]
    )
    total = len(rounds)
    score = 0
    streak_state = {"count": 0}

    stage_labels = {1: "Stage 1 — easy words", 2: "Stage 2 — harder words", 3: "Stage 3 — two words at once!"}
    current_stage = 0

    for i, (words, stage) in enumerate(rounds, start=1):
        if stage != current_stage:
            current_stage = stage
            yield say(session, f"Now starting {stage_labels[stage]}!")
            yield sleep(1)

        success = yield play_round(session, words, i, total, streak_state)
        if success:
            score += 1
        yield sleep(0.5)

    return score

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
@inlineCallbacks
def main(session, details):
    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")
    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    yield say(session,
              "Hello! I am your language buddy. "
              "We are going to play a fun sentence game with three stages. "
              "Let's start easy and get harder as we go!")
    yield sleep(1)

    while True:
        chosen_topic = yield pick_topic(session)
        score = yield run_game(session, chosen_topic)
        total = len(TOPICS[chosen_topic]["easy"]) + len(TOPICS[chosen_topic]["hard"]) + 4

        yield say(session,
                  f"Amazing work! You completed all three stages. "
                  f"You got {score} out of {total} correct. "
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