# vulture_app/pipeline.py
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from .ai_client import client

import pandas as pd

from .cache import prune_cache
from .reddit_client import fetch_posts, fetch_comments_for_post
from .ai_client import generate_tldr_for_post, generate_image_caption, parse_gpt_summary
from .scoring import (
    score_thesis_clarity,
    score_catalyst_present,
    score_formatting_quality,
    score_image_insight,
    generate_engagement_score,
    score_position_disclosure,
    calculate_confidence_score
)

def process_posts(posts) -> list[dict]:
    """
    Turn a list of PRAW submissions into scored, summarized dicts.
    """
    now = datetime.now(timezone.utc)
    results = []
    seen_titles = set()

    cache = prune_cache()

    for p in posts:
        created = datetime.fromtimestamp(p.created_utc, timezone.utc)
        # skip if older than 24h
        if created < now - timedelta(days=1):
            continue
        # skip media-only posts
        if p.url.endswith((".jpeg", ".png")) or "v.redd.it" in p.url:
            continue
        # dedupe
        if p.title in seen_titles:
            continue
        seen_titles.add(p.title)

        # 1) TLDR + parse
        summary_text = generate_tldr_for_post(p.title, p.selftext, p.id, cache)
        f = parse_gpt_summary(summary_text)
        tag = f["tag"].lower()
        if tag in ("results", "info"):
            continue

        # 2) Clean ticker
        ticker = f["ticker"].strip("<>").upper()
        if not ticker or ticker.lower() in ("nan", "n/a", "none", "not specified"):
            continue

        # 3) Date & direction
        et = created.astimezone(ZoneInfo("US/Eastern"))
        date_str = et.strftime("%Y-%m-%d %I:%M %p ET")
        direction = "Up" if "up" in f["direction"].lower() else "Down"

        # 4) Image caption
        img_caption = generate_image_caption(p) or ""

        # 5) Comments & engagement
        comments_sample, total_comments = fetch_comments_for_post(p)
        engagement = generate_engagement_score(comments_sample, p.score, total_comments)

        # 6) Sub-scores
        thesis_clarity = score_thesis_clarity(summary_text)
        catalyst_present = score_catalyst_present(summary_text)
        formatting_quality = score_formatting_quality(p.selftext)
        image_insight = score_image_insight(img_caption)
        pos_bonus = score_position_disclosure(p.selftext, comments_sample)

        # 7) Author credibility
        try:
            auth = p.author
            age_years = (now - datetime.fromtimestamp(auth.created_utc, timezone.utc)).days / 365
            acc_age = age_years
            karma = auth.link_karma + auth.comment_karma
            history_count = sum(1 for _ in auth.submissions.new(limit=None) if True)
        except Exception:
            acc_age = 0
            karma = 0
            history_count = 0

        # assemble data dict
        data = {
            "Tag":               f["tag"],
            "Title":             p.title,
            "Ticker":            ticker,
            "Direction":         direction,
            "Positions":         f["positions"] or "None listed",
            "Summary":           f["summary"],
            "Date":              date_str,
            "URL":               "https://reddit.com" + p.permalink,
            "Engagement":        round(engagement, 2),
            "thesis_clarity":    thesis_clarity,
            "catalyst_present":  catalyst_present,
            "formatting_quality":formatting_quality,
            "image_insight":     image_insight,
            "engagement_quality":engagement,
            "position_bonus":    pos_bonus,
            "account_age_years": acc_age,
            "total_karma":       karma,
            "finance_post_count":history_count
        }

        # 8) Final confidence & legendary flag
        data["ConfidenceScore"] = calculate_confidence_score(data)
        data["Legendary"]       = (thesis_clarity + catalyst_present +
                                   engagement + formatting_quality +
                                   image_insight + pos_bonus +
                                   # simplified cred sum
                                   min(acc_age/3, 1) + min(karma/1000, 1) +
                                   min(history_count/3, 1)
                                  ) > 10

        # 9) Clean up for output
        data["Confidence"] = f"{data['ConfidenceScore']}/10"
        del data["thesis_clarity"], data["catalyst_present"]
        del data["formatting_quality"], data["image_insight"]
        del data["engagement_quality"], data["position_bonus"]
        del data["account_age_years"], data["total_karma"], data["finance_post_count"]

        results.append(data)

    # Save updated cache
    # (cache is pruned & saved inside generate_tldr_for_post)
    return results

def run_scraper(subreddits: list[str]) -> str:
    """
    Run the pipeline for each subreddit, write an Excel file,
    and return the filepath.
    """
    all_data = {}
    for sub in subreddits:
        posts = []
        for t in ("top", "new"):
            posts.extend(fetch_posts(sub, post_type=t, limit=100))
        rows = process_posts(posts)
        all_data[sub] = rows

    outdir = os.path.join(os.getcwd(), "data")
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, "reddit_posts_by_subreddit.xlsx")

    with pd.ExcelWriter(output_path) as writer:
        for sub, rows in all_data.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=sub.capitalize(), index=False)

    return output_path

def run_user_analysis(usernames: list[str]) -> str:
    """
    Fetch submissions & comments for each username in the past 24h,
    call the AI to generate a market forecast, and return the text.
    """
    from prawcore import NotFound
    from .reddit_client import reddit
    now_utc       = datetime.now(timezone.utc)
    threshold_utc = now_utc - timedelta(days=1)

    collected = []
    for name in usernames:
        try:
            redditor = reddit.redditor(name)
        except NotFound:
            continue

        # fetch all recent activity
        subs  = list(redditor.submissions.new(limit=None))
        comms = list(redditor.comments.new(limit=None))
        combined = sorted(subs + comms, key=lambda i: i.created_utc, reverse=True)

        for item in combined:
            created = datetime.fromtimestamp(item.created_utc, timezone.utc)
            if created < threshold_utc:
                break
            collected.append(item)

    if not collected:
        return "No activity found in the past 24 hours."

    # Build the prompt
    blocks = []
    for item in collected:
        if hasattr(item, "selftext"):
            title = item.title
            body  = item.selftext or "[no body]"
        else:
            title = f"Comment on “{item.link_title}”"
            body  = item.body or "[no body]"
        blocks.append(f"Title: {title}\nContent: {body}")
    user_content = "\n\n".join(blocks)

    system_msg = {
        "role": "system",
        "content": (
            "You are a professional market analyst. "
            "Given these Reddit activities from the past 24 hours, "
            "craft a daily market forecast in 3–5 concise bullet points, "
            "and finish with a TL;DR of overall sentiment (Bullish, Bearish, or Neutral)."
        )
    }
    user_msg = {"role": "user", "content": user_content}

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=500,
        timeout=60
    )
    return resp.choices[0].message.content.strip()
