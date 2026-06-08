#!/usr/bin/env python3
"""
TinyBard — Micro Interactive Text Adventure Generator
======================================================
Exposes a custom gr.Server FastAPI application with a retro CRT terminal frontend.
Exposes endpoints as Gradio APIs for MCP tool integration.

Targets: Thousand Token Wood + Tiny Titan + Llama Champion tracks.
Badges: Llama Champion, Tiny Titan, Off-Brand (gr.Server), Off the Grid, Field Notes.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tinybard")

# ---------------------------------------------------------------------------
# Config & Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "TINYBARD_MODEL_PATH",
    "/Volumes/1tb-sandisk/ml-models/huggingface/models--mradermacher--VibeThinker-1.5B-GGUF/snapshots/d0d66139a78030a92a582f966b0f7cbbb3b19406/VibeThinker-1.5B.Q8_0.gguf"
)

# ---------------------------------------------------------------------------
# Llama.cpp Inference Setup
# ---------------------------------------------------------------------------
_llm = None
_llm_failed = False

def get_llm():
    global _llm, _llm_failed
    if _llm is not None:
        return _llm
    if _llm_failed:
        return None

    if not Path(MODEL_PATH).exists():
        log.warning(f"Model file not found at {MODEL_PATH}. Fallback mode active.")
        _llm_failed = True
        return None

    try:
        from llama_cpp import Llama
        log.info(f"Loading VibeThinker-1.5B from {MODEL_PATH} ...")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=int(os.environ.get("TINYBARD_THREADS", "4")),
            verbose=False,
        )
        log.info("Model loaded successfully ✓")
        return _llm
    except Exception as e:
        log.error(f"Failed to load LLM model: {e}")
        _llm_failed = True
        return None

# ---------------------------------------------------------------------------
# Procedural Fallback Adventure Engine
# ---------------------------------------------------------------------------
GENRES = {
    "fantasy": {
        "start": "You stand before the gates of the Whisperwood. The ancient trees hum with a faint violet energy.",
        "nodes": [
            {
                "story": "A glowing sprite appears, offering a golden key or a mossy vial.",
                "choices": ["Take the golden key", "Drink the mossy vial", "Ignore the sprite and press forward"]
            },
            {
                "story": "You encounter a moss-covered stone golem blocking the path. It speaks in riddles.",
                "choices": ["Answer its riddle with a joke", "Use your golden key if you have it", "Try to climb over it"]
            },
            {
                "story": "You discover a hidden pool reflecting stars that aren't in the sky.",
                "choices": ["Drink from the star pool", "Rest by the shore", "Toss a coin into the water"]
            }
        ],
        "win": "You find the heart of the forest and unlock the ancient relic. You are victorious!",
        "lose": "The energy of the forest overwhelms you. You fade into the whispers of the wood."
    },
    "scifi": {
        "start": "The emergency lights flicker red in the derelict cargo bay of USS Horizon. Gravity is failing.",
        "nodes": [
            {
                "story": "A leaking fuel pipe blocks the corridor ahead. Sparking wires fill the air.",
                "choices": ["Siphon the fuel", "Bypass the circuits", "Wait for the cycle to clear"]
            },
            {
                "story": "An automated security drone activates, targeting you with its laser system.",
                "choices": ["Hack the drone terminal", "Throw scrap metal to distract it", "Run for the airlock"]
            },
            {
                "story": "You reach the main computer terminal. The AI core is corrupt but online.",
                "choices": ["Initiate override protocol", "Ask the AI for help", "Pull the main power breaker"]
            }
        ],
        "win": "You restore life support and secure the escape pod. You survive!",
        "lose": "The hull breaches. You are swept into the cold embrace of outer space."
    },
    "cyberpunk": {
        "start": "Acid rain beats against the neon signs of Sector 9. Your neural interface is glitching.",
        "nodes": [
            {
                "story": "A street dealer offers to patch your wetware for a few credits or a favor.",
                "choices": ["Accept the shady patch", "Decline and buy a neural booster", "Threaten him for info"]
            },
            {
                "story": "A corporate agent corner you in a wet alleyway. He demands your datapad.",
                "choices": ["Upload a virus to his cyber-eyes", "Hand over a fake datapad", "Sprint up the fire escape"]
            },
            {
                "story": "You infiltrate the mainframe room of Shinra-Tech. The security grid is active.",
                "choices": ["Jack in directly", "Use your backup deck", "Short-circuit the access node"]
            }
        ],
        "win": "You upload the corporate secrets to the net. Sector 9 is free. You win!",
        "lose": "Your brain fried due to feedback from the security grid. Game Over."
    }
}

def generate_procedural_step(genre: str, step: int, health: int, choice: str = "") -> dict:
    genre_data = GENRES.get(genre.lower(), GENRES["fantasy"])
    
    if step == 0:
        return {
            "story": genre_data["start"],
            "choices": genre_data["nodes"][0]["choices"],
            "health": health,
            "step": 1,
            "game_over": False
        }
    
    # Simple logic to resolve choice and advance
    health_delta = random.choice([-15, 0, 10])
    new_health = max(0, min(100, health + health_delta))
    
    if new_health <= 0:
        return {
            "story": f"After choosing: '{choice}'. " + genre_data["lose"],
            "choices": [],
            "health": 0,
            "step": step + 1,
            "game_over": True
        }
        
    if step >= 4:
        return {
            "story": f"After choosing: '{choice}'. " + genre_data["win"],
            "choices": [],
            "health": new_health,
            "step": step + 1,
            "game_over": True
        }
        
    node = genre_data["nodes"][step % len(genre_data["nodes"])]
    return {
        "story": f"You choose: '{choice}'.\n\n{node['story']}",
        "choices": node["choices"],
        "health": new_health,
        "step": step + 1,
        "game_over": False
    }

# ---------------------------------------------------------------------------
# LLM Generation Logic
# ---------------------------------------------------------------------------
def generate_llm_story(prompt: str, max_tokens: int = 150) -> str:
    llm = get_llm()
    if not llm:
        return ""
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["\n\n", "User:", "Narrator:"],
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        log.error(f"LLM generation error: {e}")
        return ""

def format_prompt(genre: str, history: List[Dict[str, str]], next_instruction: str) -> str:
    # A lightweight prompt template for VibeThinker/Qwen
    prompt = (
        "You are the narrator of an interactive text adventure game.\n"
        f"Genre: {genre}\n"
        "Rules:\n"
        "1. Write in the second person ('You...').\n"
        "2. Keep descriptions highly atmospheric, but short (under 3 sentences).\n"
        "3. Focus on action, mystery, and choice.\n\n"
    )
    for h in history:
        if h["role"] == "player":
            prompt += f"Player choice: {h['text']}\n"
        else:
            prompt += f"Narrator: {h['text']}\n"
            
    prompt += f"{next_instruction}\n"
    return prompt

# ---------------------------------------------------------------------------
# Web Server
# ---------------------------------------------------------------------------
app = Server()

# Serve static directory if it exists
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Main game state in memory (for simplicity and demo purposes)
# In production / Spaces multi-user, state should be client-side (passed back/forth)
# We support passing state back/forth via request/response to remain stateless!

@app.api(name="start_game")
def start_game(genre: str) -> dict:
    """
    Start a new interactive text adventure in the chosen genre (e.g. Fantasy, Sci-Fi, Cyberpunk).
    Exposed as an MCP tool!
    """
    genre = genre.lower()
    if genre not in ["fantasy", "scifi", "cyberpunk"]:
        genre = "fantasy"
        
    llm = get_llm()
    if not llm:
        # Fallback
        return generate_procedural_step(genre, 0, 100)

    # Initial prompt
    instruction = "Narrate the beginning of the adventure. What happens first? Do not offer choices yet."
    story = generate_llm_story(format_prompt(genre, [], instruction))
    if not story:
        return generate_procedural_step(genre, 0, 100)

    # Get initial choices
    history = [{"role": "narrator", "text": story}]
    choices_instruction = (
        "Provide exactly 3 short, distinct choices for the player to make next. "
        "Format: 1. [choice 1] | 2. [choice 2] | 3. [choice 3]"
    )
    choices_text = generate_llm_story(format_prompt(genre, history, choices_instruction), max_tokens=60)
    
    # Parse choices
    choices = []
    if "|" in choices_text:
        choices = [c.split(".")[-1].strip() for c in choices_text.split("|")]
    else:
        # standard list parsing
        for line in choices_text.split("\n"):
            if "." in line or "1" in line or "2" in line or "3" in line:
                parts = line.split(".", 1)
                if len(parts) > 1:
                    choices.append(parts[1].strip())
                    
    if len(choices) < 2:
        # Fallback choices
        choices = ["Explore the area", "Check your equipment", "Proceed carefully"]

    return {
        "story": story,
        "choices": choices[:3],
        "health": 100,
        "step": 1,
        "game_over": False,
        "history": history
    }

@app.api(name="make_choice")
def make_choice(choice: str, genre: str, step: int, health: int, history_json: str) -> dict:
    """
    Submit a player choice to advance the story.
    Exposed as an MCP tool!
    """
    try:
        history = json.loads(history_json)
    except Exception:
        history = []

    llm = get_llm()
    if not llm:
        return generate_procedural_step(genre, step, health, choice)

    # Add player choice
    history.append({"role": "player", "text": choice})
    
    # Calculate health impact
    health_delta = random.choice([-15, 0, 10])
    new_health = max(0, min(100, health + health_delta))

    if new_health <= 0:
        instruction = "The player has run out of health. Narrate a quick, dramatic end to their adventure. Game Over."
        story = generate_llm_story(format_prompt(genre, history, instruction))
        return {
            "story": story or "Your strength fails. The adventure ends in darkness.",
            "choices": [],
            "health": 0,
            "step": step + 1,
            "game_over": True,
            "history": history
        }

    if step >= 4:
        instruction = "Narrate the final glorious victory of the player. The adventure ends in success."
        story = generate_llm_story(format_prompt(genre, history, instruction))
        return {
            "story": story or "You have achieved your goal! You are victorious!",
            "choices": [],
            "health": new_health,
            "step": step + 1,
            "game_over": True,
            "history": history
        }

    # Narrate step
    instruction = "Narrate what happens next as a result of the player's choice."
    story = generate_llm_story(format_prompt(genre, history, instruction))
    if not story:
        return generate_procedural_step(genre, step, health, choice)

    history.append({"role": "narrator", "text": story})

    # Get next choices
    choices_instruction = (
        "Provide exactly 3 short, distinct choices for the player to make next. "
        "Format: 1. [choice 1] | 2. [choice 2] | 3. [choice 3]"
    )
    choices_text = generate_llm_story(format_prompt(genre, history, choices_instruction), max_tokens=60)
    
    choices = []
    if "|" in choices_text:
        choices = [c.split(".")[-1].strip() for c in choices_text.split("|")]
    else:
        for line in choices_text.split("\n"):
            if "." in line or "1" in line or "2" in line or "3" in line:
                parts = line.split(".", 1)
                if len(parts) > 1:
                    choices.append(parts[1].strip())
                    
    if len(choices) < 2:
        choices = ["Move forward", "Look around", "Rest a moment"]

    return {
        "story": story,
        "choices": choices[:3],
        "health": new_health,
        "step": step + 1,
        "game_over": False,
        "history": history
    }

@app.get("/api/model_status")
def model_status():
    llm = get_llm()
    return {
        "available": llm is not None,
        "model_path": MODEL_PATH,
        "fallback": _llm_failed
    }

@app.get("/", response_class=HTMLResponse)
async def homepage():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path) as f:
            return f.read()
    return HTMLResponse("<h1>TinyBard retro terminal under construction!</h1>")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    # Start the Server with MCP capability enabled by default!
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        mcp_server=True
    )
