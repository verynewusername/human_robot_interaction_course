# DLD Sentence Game — Language Buddy Robot

A Human-Robot Interaction (HRI) project for children with **Developmental
Language Disorder (DLD)**. A social robot (Alpha Mini / QT) guides a child
through a sentence-construction game, adapting difficulty in real time and
validating responses via a local or cloud LLM.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Project](#running-the-project)
7. [Game Flow](#game-flow)
8. [LLM Backends](#llm-backends)
9. [Recommended Models](#recommended-models)
10. [Flags Reference](#flags-reference)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

The robot presents a child with a target word (or two words at harder stages)
and asks them to form a grammatically correct sentence using it. An LLM acts
as a silent therapist, checking whether the sentence is valid and generating
warm, child-friendly feedback. Difficulty adapts automatically across three
stages:

| Stage | Task | Words used |
|-------|------|------------|
| 1 | Form a sentence with one easy word | Simple vocabulary |
| 2 | Form a sentence with one harder word | Extended vocabulary |
| 3 | Form a sentence using two words at once | Any from the topic |

The child progresses to the next stage only if they score **2 out of 2** in
the current round. Each full game is **6 questions across 3 rounds of 2**.

Topics available: **animals**, **food**, **school**, **weather**.
On the real robot the topic is selected by holding up a **QR card**.
In local test mode it is selected by typing.

---

## File Structure

```
project/
├── main.py              ← Main robot script
├── test_validator.py    ← Standalone validator test (no robot needed)
├── .env                 ← API keys and WAMP realm (not committed)
├── .env.example         ← Template for .env
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## Requirements

- Python **3.9+**
- [Ollama](https://ollama.com) installed and running locally *(default backend)*
- A supported model pulled in Ollama — see [Recommended Models](#recommended-models)
- A [Google AI Studio](https://aistudio.google.com/apikey) API key *(for
  `--gemini` flag only)*
- Access to the **RIE WAMP** robot infrastructure *(for real robot only)*

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
autobahn[twisted]
twisted
google-genai
python-dotenv
requests
```

### 4. Pull a model into Ollama

```bash
# Make sure Ollama is installed: https://ollama.com

ollama pull gemma3:4b       # recommended — 3.3 GB
# or lighter:
ollama pull qwen2.5:3b      # 1.9 GB, good fallback for low RAM Macs
```

See [Recommended Models](#recommended-models) for the full comparison.

---

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

**`.env.example`**
```
# Required for --gemini flag only
GEMINI_API_KEY=your_google_ai_studio_key_here

# Required for real robot connection only
WAMP_REALM=rie.xxxxxxxxxxxxxxxxxxxxxxxx
```

> Neither value is needed to run the game locally with Ollama and `--local-test`.

---

## Running the Project

### Quick start — local test with Ollama (no robot, no API key needed)

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — run the game
python main.py --local-test
```

### All run modes

| Command | LLM backend | Robot connected |
|---------|-------------|-----------------|
| `python main.py --local-test` | Ollama (local) | ❌ Terminal I/O |
| `python main.py` | Ollama (local) | ✅ Real robot |
| `python main.py --local-test --gemini` | Gemini cloud | ❌ Terminal I/O |
| `python main.py --gemini` | Gemini cloud | ✅ Real robot |

### Testing the validator in isolation

```bash
# Run preset test cases against Ollama (default)
python test_validator.py

# Run preset test cases against Gemini cloud
python test_validator.py --gemini

# Interactive mode — type your own words and sentences
python test_validator.py --interactive
python test_validator.py --interactive --gemini

# Offline rule-based validation (no LLM at all)
python test_validator.py --local
```

---

## Game Flow

```
Robot greets child
        │
        ▼
Child picks a topic
  ← typed (local-test) or QR card (real robot)
  animals / food / school / weather
        │
        ▼
┌──────────────────────────────────────┐
│  Round 1 — Stage 1 (easy words)      │
│  2 questions × 1 word each           │
└──────────────────────────────────────┘
        │ score 2/2 → advance to next stage
        │ score 0–1 → stay at same stage
        ▼
┌──────────────────────────────────────┐
│  Round 2 — Stage 1 or 2             │
│  2 questions                         │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Round 3 — Stage 1, 2, or 3         │
│  2 questions                         │
└──────────────────────────────────────┘
        │
        ▼
Score summary → play again?
```

**Per question:**
- Up to **4 attempts** allowed
- After each failed attempt the robot explains what was wrong and prompts a retry
- On the 4th failed attempt the robot models a correct example sentence
  *(evidence-based recasting strategy)*
- A **streak counter** gives bonus encouragement every 3 correct answers in a row

---

## LLM Backends

### Ollama — default

- Runs **fully offline** on your machine — no API key, no rate limits, no cost
- Auto-detects the first Gemma model you have pulled; falls back to any available model
- Uses Ollama's JSON schema structured output to guarantee valid JSON even on small models
- Recommended model: `gemma3:4b`

```bash
ollama serve          # start the server (port 11434 by default)
ollama list           # see what models you have pulled
ollama pull gemma3:4b # pull if not already downloaded
```

### Gemini cloud — `--gemini` flag

- Uses `gemini-1.5-flash` (fast, free tier: 15 RPM / 1,500 requests per day)
- Requires `GEMINI_API_KEY` in `.env`
- Get a free key at: https://aistudio.google.com/apikey

> **Note:** Larger models such as `gemma-4-31b-it` are available on the Gemini
> API but are heavily rate-limited on the free tier and not suitable for
> real-time robot interaction.

---

## Recommended Models

For this task — short sentence grammar checking with JSON output — the bar is
low. Any of the models below will work correctly. Choose based on your Mac's
available RAM.

| Model | Pull command | Download size | Min RAM | Notes |
|-------|-------------|---------------|---------|-------|
| `gemma3:4b` ⭐ | `ollama pull gemma3:4b` | 3.3 GB | ~5 GB | **Recommended.** Best JSON + grammar balance. Same family as Gemma 4. Fast on Apple Silicon. |
| `qwen2.5:3b` | `ollama pull qwen2.5:3b` | 1.9 GB | ~3 GB | Good fallback for Macs with 8 GB RAM. Solid instruction following and JSON output. |
| `phi4-mini` | `ollama pull phi4-mini` | 2.5 GB | ~4 GB | Microsoft model. Very strong for its size. Reliable structured output. |
| `gemma3:1b` | `ollama pull gemma3:1b` | 0.8 GB | ~2 GB | Smallest option. JSON may be inconsistent without schema enforcement. Last resort only. |

> The model is **auto-detected** at startup — no config change needed after pulling.
> The script picks the first Gemma model found, or the first available model of any kind.

### Why small models are sufficient here

The validator prompt asks for a fixed 3-field JSON object in response to a
single short sentence. This is a simple classification task — far easier than
reasoning or code generation. A 4B model handles it reliably, especially with
Ollama's JSON schema enforcement which constrains the output at the token level,
guaranteeing valid JSON regardless of model size.

---

## Flags Reference

### `main.py`

| Flag | Effect |
|------|--------|
| *(none)* | Ollama local LLM + real robot connection |
| `--local-test` | Terminal I/O instead of robot (Ollama backend) |
| `--gemini` | Use Gemini cloud API instead of Ollama |
| `--local-test --gemini` | Terminal I/O + Gemini cloud |

### `test_validator.py`

| Flag | Effect |
|------|--------|
| *(none)* | Run preset test cases via Ollama |
| `--gemini` | Run preset test cases via Gemini cloud |
| `--interactive` | Enter your own words and sentences manually |
| `--local` | Offline rule-based validation — no LLM at all |

---

## Troubleshooting

**`RuntimeError: Cannot reach Ollama at http://localhost:11434`**
```bash
# Ollama is not running — start it:
ollama serve
```

**`RuntimeError: Ollama is running but has no models pulled`**
```bash
ollama pull gemma3:4b
```

**Ollama returns invalid JSON**
- This can happen with very small models (`gemma3:1b`) that ignore the schema
- Switch to a larger model: `ollama pull gemma3:4b`
- Or use the Gemini backend: `python main.py --local-test --gemini`

**`429 RESOURCE_EXHAUSTED` when using `--gemini`**
- You have hit the free tier rate limit (15 requests/min)
- Switch to Ollama (default, no limits): remove the `--gemini` flag
- Or wait ~1 minute and retry

**`KeyError: WAMP_REALM` or robot connection fails**
- Add `WAMP_REALM=rie.xxxx...` to your `.env` file
- Only needed when running against the real robot (without `--local-test`)

**Robot does not hear the child / STT times out**
- Check that the robot microphone is active
- Increase `LISTEN_TIMEOUT` at the top of `main.py` (default: 15 seconds)

**Game is slow on Mac**
- Make sure Ollama is using the Apple Silicon GPU/Neural Engine:
  check `ollama ps` — it should show `100% GPU`
- Switch to a smaller model: `ollama pull qwen2.5:3b`

---

## AI Usage Disclosure

This project used **Raycast AI (Claude Sonnet)** as a coding assistant during
development. It was used to:

- Generate and refactor boilerplate code structure
- Debug WAMP subscription handling
- Suggest and compare LLM backend options
- Draft this README

All AI-generated output was reviewed, tested, and validated by the project
team before inclusion. The system prompt design, game logic, difficulty
adaptation strategy, and clinical justifications were authored by the team.
