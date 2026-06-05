# HRI Sentence Game – DLD

A sentence-forming game for the Alpha Mini / QT robot, designed for children with Developmental Language Disorder (DLD). The robot picks a topic, gives the child a word, and asks them to form a sentence. Gemini validates the sentence and provides feedback.

## Requirements

- Python 3.9+
- A Gemini API key (free tier works)
- Access to the RIE WAMP robot infrastructure

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your values in .env
python main.py
```

## How it works

1. Robot greets the child and asks them to choose a topic (animals, food, school, weather)
2. For each word in the topic, the child must form a sentence using that word
3. Gemini validates grammar, logic, and correct use of the target word
4. The child has up to 4 attempts per word
5. At the end, the robot summarises the score and asks if they want to play again

## Configuration

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `WAMP_REALM` | Your robot's WAMP realm (e.g. `rie.xxxxxxxxxxxx`) |
| `LISTEN_TIMEOUT` | Seconds to wait for speech before timing out (default: 15) |