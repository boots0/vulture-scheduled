# vulture_app/ai_client.py
import os
from datetime import datetime, timezone
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# System prompt for classifying & summarizing Reddit posts
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an expert financial summarizer. For each Reddit post, do two things:\n"
        "1) Assign exactly one Tag from {DD, News, Results, Info}:\n"
        "   • DD      = Deep-Dive or research\n"
        "   • News    = Breaking news or new information\n"
        "   • Results = Gains/Losses or bragging (will be filtered out)\n"
        "   • Info    = Requests for opinions, advice, or feedback (will be filtered out)\n"
        "2) Write a TLDR of max two sentences.\n\n"
        "Output exactly:\n"
        "Tag: <DD|News|Results|Info>\n"
        "TLDR: <summary>\n"
        "Ticker: <ticker>\n"
        "Direction: <Up|Down>\n"
        "Positions: <brief or None>\n"
        "Date: <ISO 8601 UTC datetime>\n"
        "Do not include anything else."
    )
}

def parse_gpt_summary(text: str) -> dict:
    """
    Turn the assistant’s raw response into structured fields.
    """
    sections = {"tag": "", "summary": "", "ticker": "", "direction": "", "positions": ""}
    curr = None
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        low = l.lower()
        if low.startswith("tag:"):
            curr = "tag";    sections["tag"]       = l.split(":", 1)[1].strip()
        elif low.startswith("tldr:"):
            curr = "summary";sections["summary"]   = l.split(":", 1)[1].strip()
        elif low.startswith("ticker:"):
            curr = "ticker"; sections["ticker"]    = l.split(":", 1)[1].strip()
        elif low.startswith("direction:"):
            curr = "direction"; sections["direction"] = l.split(":", 1)[1].strip()
        elif low.startswith("positions:"):
            curr = "positions";sections["positions"]  = l.split(":", 1)[1].strip()
        else:
            if curr:
                sections[curr] += " " + l
    return {k: v.strip() for k, v in sections.items()}

def generate_tldr_for_post(title: str, body: str, pid: str, cache: dict) -> str:
    """
    Ask the model for a TLDR; cache results for CACHE_EXPIRY_HOURS.
    Returns the raw assistant text.
    """
    # If already in cache, return
    if pid in cache:
        return cache[pid]["summary"]

    user_msg = {
        "role": "user",
        "content": f"Title: {title}\n\nPost: {body}"
    }

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[SYSTEM_MESSAGE, user_msg],
        temperature=0.7,
        max_tokens=200,
        timeout=30
    )
    txt = resp.choices[0].message.content.strip()

    # Update cache
    cache[pid] = {
        "summary": txt,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return txt

def generate_image_caption(post) -> str | None:
    """
    If a preview image exists, ask the vision model to caption it.
    """
    try:
        img_url = post.preview["images"][0]["source"]["url"]
    except Exception:
        return None

    vision_system = {
        "role": "system",
        "content": (
            "You are an assistant that looks at a chart or screenshot and "
            "writes a brief, factual 1–2 sentence caption of what it shows."
        )
    }
    vision_user = {
        "role": "user",
        "content": f"Please describe what this image shows: {img_url}"
    }

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[vision_system, vision_user],
        temperature=0.0,
        max_tokens=60,
        timeout=30
    )
    return resp.choices[0].message.content.strip()
