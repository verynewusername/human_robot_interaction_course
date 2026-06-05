from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

# read prompt from file
with open("1st_assignment/1stassignmentprompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

# initialize Gemini client
chatbot = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# conversation history to maintain context
conversation_history = []

def get_robot_response(user_input):
    conversation_history.append(user_input)
    response = chatbot.models.generate_content(
        model="GEMMA 3 27b",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
        contents=conversation_history
    )
    conversation_history.append(response.text)
    return response.text

# globals for STT
finish_dialogue = False
query = ""

def asr(frames):
    global finish_dialogue, query
    if frames["data"]["body"]["final"]:
        query = str(frames["data"]["body"]["text"]).strip()
        print("Person said: ", query)
        finish_dialogue = True

@inlineCallbacks
def main(session, details):
    global finish_dialogue, query

    # set language to English
    yield session.call("rie.dialogue.config.language", lang="en")

    # robot stands up
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    # opening greeting from robot
    opening = get_robot_response("Hello, please start the conversation by greeting the person warmly and asking their name.")
    print("Robot: ", opening)
    yield session.call("rie.dialogue.say_animated", text=opening)

    # sleep so robot does not hear itself
    yield sleep(2)

    # set up STT
    yield session.subscribe(asr, "rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.stt.stream")

    exit_conditions = ("quit", "exit", "stop", "goodbye")
    dialogue = True

    while dialogue:
        if finish_dialogue:
            yield session.call("rie.dialogue.stt.close")
            yield sleep(1)

            print("Processing: ", query)

            if any(word in query.lower() for word in exit_conditions):
                dialogue = False
                yield session.call("rie.dialogue.say_animated", text="Goodbye! It was lovely talking with you.")
                break

            elif query != "":
                robot_response = get_robot_response(query)
                print("Robot: ", robot_response)
                yield session.call("rie.dialogue.say_animated", text=robot_response)
                # sleep after speaking so robot does not hear itself
                yield sleep(2)

            else:
                yield session.call("rie.dialogue.say_animated", text="Sorry, I did not catch that. Could you try again?")
                yield sleep(1)

            finish_dialogue = False
            query = ""
            yield session.call("rie.dialogue.stt.stream")

        yield sleep(0.5)

    yield session.call("rie.dialogue.stt.close")
    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()

wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
        "serializers": ["msgpack"],
        "max_retries": 0
    }],
    realm="rie.69f0754d26d8af1680826f98",
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])