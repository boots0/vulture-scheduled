import os
import sys
import json
import re
import argparse
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import praw
import requests
from dotenv import load_dotenv
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------
# Setup and Configuration
# ---------------------------

load_dotenv("vulture_cred.env")

# Function to check for required environment variables
def check_environment_variables():
    """Checks for all required environment variables and raises an error if any are missing."""
    required_vars = [
        "CLIENT_ID", "CLIENT_SECRET", "USER_AGENT", "OPENAI_API_KEY",
        "DISCORD_WEBHOOK_FORUM", "DISCORD_WEBHOOK_NEWS", "DISCORD_TAG_ID_LOW",
        "DISCORD_TAG_ID_MEDIUM", "DISCORD_TAG_ID_HIGH",
        "GOOGLE_CREDENTIALS_JSON", "GOOGLE_SHEET_NAME",
        "GOOGLE_TRAINING_SHEET_NAME"
    ]
    missing_vars = []
    print("--- Checking Environment Variables ---")
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"- {var}: MISSING!")
        else:
            print(f"- {var}: Found.")
    
    if missing_vars:
        raise ValueError(
            f"The following required environment variables are missing: {', '.join(missing_vars)}. "
        )
    print("--- All required environment variables found. ---\n")

# Run the check immediately on startup
check_environment_variables()

# --- PRAW (Reddit API) Setup ---
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

# --- OpenAI Client Setup ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Discord Configuration ---
WEBHOOKS = {
    "forum": os.getenv("DISCORD_WEBHOOK_FORUM"),
    "news": os.getenv("DISCORD_WEBHOOK_NEWS"),
    "tag_id_low": os.getenv("DISCORD_TAG_ID_LOW"),
    "tag_id_medium": os.getenv("DISCORD_TAG_ID_MEDIUM"),
    "tag_id_high": os.getenv("DISCORD_TAG_ID_HIGH"),
}

# --- Constants ---
TARGET_SUBREDDITS = [
    "options", "wallstreetbets", "shortsqueeze", "DueDilligence",
    "WallStreetbetsELITE", "smallstreetbets", "valueinvesting", "technicalanalysis"
]
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROCESSED_POSTS_FILE = os.path.join(OUTPUT_DIR, "processed_posts.txt")
DAILY_SUMMARY_LOG_FILE = os.path.join(OUTPUT_DIR, "daily_summary_log.txt")


# ---------------------------
# Google Sheets Client
# ---------------------------

def get_gspread_client():
    """Initializes and returns an authenticated gspread client."""
    creds_json = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_json, scopes=scopes)
    return gspread.authorize(creds)

# ---------------------------
# Core Vulture Logic
# ---------------------------

def get_comments_for_post(post_id, limit=25):
    """Fetches top comments for a given Reddit post ID."""
    try:
        print(f"Fetching comments for post ID: {post_id}...")
        submission = reddit.submission(id=post_id)
        submission.comment_sort = "top"
        comments = [comment.body for comment in submission.comments[:limit] if not comment.stickied]
        time.sleep(1) # Be respectful to Reddit's API rate limits
        return "\n".join(comments)
    except Exception as e:
        print(f"Could not fetch comments for post {post_id}: {e}")
        return ""

def scrape_new_posts(subreddits, processed_ids):
    """Scrapes only new, unprocessed posts."""
    all_posts_data = []
    now = datetime.now(timezone.utc)
    for sub in subreddits:
        print(f"Fetching posts from r/{sub}...")
        posts = list(reddit.subreddit(sub).new(limit=100))
        for p in posts:
            if p.id in processed_ids: continue
            created = datetime.fromtimestamp(p.created_utc, timezone.utc)
            if created < now - timedelta(days=2): continue
            if p.url.endswith((".jpeg", ".png")) or "v.redd.it" in p.url: continue
            all_posts_data.append({
                "id": p.id, "subreddit": sub, "title": p.title, "selftext": p.selftext,
                "url": f"https://reddit.com{p.permalink}", "created_utc": created.isoformat(),
                "score": p.score, "num_comments": p.num_comments
            })
    return all_posts_data

def get_ai_synthesis(post_data, comments_text):
    """Sends a post and its comments to the AI for a holistic analysis."""
    print(f"Sending post '{post_data['title']}' for AI synthesis...")
    
    system_prompt = """
    You are an expert retail investor, skilled at replicating the intuitive process of a human analyst to find actionable trading ideas on Reddit.
    Your primary goal is to synthesize the original post with the community's reaction in the comments to form a holistic view.

    **Analysis Steps & Output Fields:**
    You will be given the original post and a sample of its top comments. Based on BOTH, generate:
    1.  `ticker`: The SINGLE stock ticker discussed. If multiple or none, return "N/A".
    2.  `briefing`: A synthesis of the post and comments. What is the core thesis? How did the community react? Are there strong counterarguments or validations?
    3.  `the_play`: The community-vetted actionable takeaway. What is the real play after considering the comments? If none, state "No clear play identified."
    4.  `confidence_score`: A score from 0.0 to 10.0. This score MUST reflect the combined quality of the original thesis AND the community's reception. A great idea torn apart by comments should receive a LOW score.

    **Core Rules for Scoring:**
    - If `ticker` is "N/A", the `confidence_score` MUST be 0.0.
    - **High Confidence (8.0-10.0):** A clear, well-reasoned thesis with strong, positive validation in the comments.
    - **Medium Confidence (4.0-7.9):** A decent thesis with mixed or moderate community feedback.
    - **Low Confidence (0.1-3.9):** A speculative idea, or a good idea that was heavily criticized in the comments.
    - **Zero Confidence (0.0):** No actionable play, a question, or a post that was thoroughly debunked by the community.

    Respond with ONLY a valid JSON object containing these fields.
    """
    user_prompt = f"**Original Post Title:** {post_data['title']}\n\n**Original Post Body:**\n{post_data['selftext']}\n\n**Top Comments:**\n{comments_text}"
    
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=0.5)
        analysis = json.loads(response.choices[0].message.content)
        
        required_keys = ['ticker', 'briefing', 'the_play', 'confidence_score']
        if all(key in analysis for key in required_keys):
            return analysis
        else:
            print("AI response was missing required keys.")
            return None
    except Exception as e:
        print(f"An error occurred during AI synthesis: {e}"); return None

def append_plays_to_sheet(plays_data, sheet_name, training_mode=False):
    """Appends a list of plays to the specified Google Sheet."""
    if not plays_data: return
    print(f"Appending {len(plays_data)} rows to Google Sheet: {sheet_name}...")
    try:
        gc = get_gspread_client()
        worksheet = gc.open(sheet_name).sheet1
        
        rows_to_append = []
        for play in plays_data:
            if training_mode:
                # Format for Agent Training Data sheet
                rows_to_append.append([
                    play.get('id'), play.get('title'), play.get('selftext'),
                    play.get('ticker'), play.get('briefing'), play.get('the_play'),
                    play.get('confidence_score'), datetime.now(timezone.utc).isoformat()
                ])
            else:
                # Format for main Vulture Data sheet
                rows_to_append.append([
                    play.get('id'), play.get('ticker'), play.get('briefing'),
                    play.get('the_play'), play.get('confidence_score'),
                    play.get('url'), play.get('subreddit'), play.get('created_utc'),
                    datetime.now(timezone.utc).isoformat(), "Pending Review" # Status
                ])
        
        worksheet.append_rows(rows_to_append)
        print(f"Successfully appended data to {sheet_name}.")
    except Exception as e:
        print(f"Error appending to {sheet_name}: {e}")


def update_reviewed_plays_in_sheet(updated_plays):
    """Updates existing rows in the Google Sheet with new analysis."""
    if not updated_plays: return
    print(f"Updating {len(updated_plays)} reviewed plays in Google Sheet...")
    try:
        gc = get_gspread_client()
        sheet_name = os.getenv("GOOGLE_SHEET_NAME")
        worksheet = gc.open(sheet_name).sheet1
        
        cell_updates = []
        for play in updated_plays:
            try:
                cell = worksheet.find(play['id'])
                cell_updates.append(gspread.Cell(cell.row, 3, play['briefing']))
                cell_updates.append(gspread.Cell(cell.row, 4, play['the_play']))
                cell_updates.append(gspread.Cell(cell.row, 5, play['confidence_score']))
                cell_updates.append(gspread.Cell(cell.row, 10, "Confirmed"))
            except gspread.exceptions.CellNotFound:
                print(f"Could not find post ID {play['id']} in sheet to update.")
                continue
        
        if cell_updates:
            worksheet.update_cells(cell_updates)
            print("Successfully updated reviewed plays.")
    except Exception as e:
        print(f"Error updating reviewed plays in sheet: {e}")

def post_plays_to_discord(plays_data):
    """Posts a finalized list of plays to the Discord forum."""
    if not plays_data:
        print("No plays to post to Discord.")
        return

    print(f"Posting {len(plays_data)} plays to Discord...")
    forum_webhook_url = WEBHOOKS.get("forum")
    if not forum_webhook_url:
        print("Warning: Discord forum webhook not set.")
        return

    for play in plays_data:
        score = play.get('confidence_score', 0.0)
        ticker = play.get('ticker', 'N/A')
        
        if score >= 8.0: tag_id, color, emoji = WEBHOOKS.get("tag_id_high"), 0x00C775, "🚀"
        elif score >= 4.0: tag_id, color, emoji = WEBHOOKS.get("tag_id_medium"), 0xFFFF00, "🤔"
        else: tag_id, color, emoji = WEBHOOKS.get("tag_id_low"), 0xFF0000, "⛔️"
        
        thread_name = f"{ticker} | Confidence: {score:.1f} | {emoji}"
        
        embed = {
            "title": f"Vulture Analysis: {ticker}",
            "description": play.get('briefing', 'No briefing available.'),
            "color": color,
            "fields": [
                {"name": "The Community-Vetted Play", "value": play.get('the_play', 'N/A'), "inline": False},
                {"name": "Source", "value": f"r/{play.get('subreddit', 'N/A')}", "inline": True},
                {"name": "Link", "value": f"[View Post]({play.get('url', '#')})", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Vulture Analysis"}
        }
        payload = {"thread_name": thread_name, "embeds": [embed], "applied_tags": [tag_id] if tag_id else []}
        
        try:
            requests.post(f"{forum_webhook_url}?wait=True", json=payload, timeout=15).raise_for_status()
            print(f"Successfully posted play for {ticker} to Discord.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to create Discord forum post for {ticker}: {e}")
        
        time.sleep(2)

# ---------------------------
# Main Execution Logic
# ---------------------------

def run_discovery_scan(processed_ids):
    """Finds new posts, analyzes them, and adds them to the sheet."""
    print("\n--- Running Discovery Scan ---")
    new_posts_data = scrape_new_posts(TARGET_SUBREDDITS, processed_ids)
    if not new_posts_data:
        print("No new posts found to discover."); return
    
    newly_analyzed_plays = []
    new_processed_ids = []
    for post in new_posts_data:
        comments = get_comments_for_post(post['id'])
        analysis = get_ai_synthesis(post, comments)
        if analysis and analysis.get('confidence_score', 0) > 0:
            full_play_data = {**post, **analysis}
            newly_analyzed_plays.append(full_play_data)
        new_processed_ids.append(post['id'])
    
    append_plays_to_sheet(newly_analyzed_plays, os.getenv("GOOGLE_SHEET_NAME"))
    append_plays_to_sheet(newly_analyzed_plays, os.getenv("GOOGLE_TRAINING_SHEET_NAME"), training_mode=True)
    save_processed_ids(new_processed_ids)

def run_confirmation_scan():
    """Finds 'Pending Review' posts, re-analyzes, updates sheet, and posts to Discord."""
    print("\n--- Running Confirmation Scan ---")
    try:
        gc = get_gspread_client()
        sheet_name = os.getenv("GOOGLE_SHEET_NAME")
        worksheet = gc.open(sheet_name).sheet1
        all_records = worksheet.get_all_records()
        
        pending_posts = [record for record in all_records if record.get('status') == 'Pending Review']
        if not pending_posts:
            print("No posts are pending review."); return
            
        print(f"Found {len(pending_posts)} posts pending review.")
        updated_plays = []
        confirmed_plays_for_discord = []
        for post in pending_posts:
            comments = get_comments_for_post(post['id'])
            # We need the original title and selftext for a good re-analysis.
            # This is a limitation; for now, we pass placeholder data.
            # A better implementation might store selftext in the sheet or re-fetch it.
            original_post_data = {'title': post.get('title', 'Title not available'), 'selftext': 'Body not available'}
            analysis = get_ai_synthesis(original_post_data, comments)
            if analysis:
                analysis['id'] = post['id']
                updated_plays.append(analysis)
                # Combine with original data for Discord post
                confirmed_play = {**post, **analysis}
                confirmed_plays_for_discord.append(confirmed_play)
        
        update_reviewed_plays_in_sheet(updated_plays)
        post_plays_to_discord(confirmed_plays_for_discord)

    except Exception as e:
        print(f"An error occurred during the confirmation scan: {e}")

def run_community_scan():
    """Main pipeline for the community scan feature."""
    print("--- Vulture Community Scan triggered ---")
    
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    last_run_date_file = os.path.join(OUTPUT_DIR, "last_run_date.txt")
    last_run_date = ""
    if os.path.exists(last_run_date_file):
        with open(last_run_date_file, 'r') as f:
            last_run_date = f.read().strip()
    
    is_first_run_of_day = (last_run_date != today_str)

    processed_ids = load_processed_ids()

    if not is_first_run_of_day:
        run_confirmation_scan()

    run_discovery_scan(processed_ids)
    
    if is_first_run_of_day:
        with open(last_run_date_file, 'w') as f:
            f.write(today_str)
        print("First run of the day complete. Daily summary could be generated here.")

    print("--- Vulture Community Scan Complete ---")

# ---------------------------
# Entry point for command-line execution
# ---------------------------

if __name__ == "__main__":
    run_community_scan()
