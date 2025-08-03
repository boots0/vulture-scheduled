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
import finnhub

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
        "GOOGLE_CREDENTIALS_JSON", "GOOGLE_SHEET_NAME", "GOOGLE_TRAINING_SHEET_NAME",
        "GOOGLE_NEWS_SHEET_NAME", "GOOGLE_CALENDAR_SHEET_NAME", "FINNHUB_API_KEY"
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

# --- API Client Setup ---
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))


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
    "wallstreetbets", "shortsqueeze", "WallStreetbetsELITE", "smallstreetbets"
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
# Memory/Cache Functions
# ---------------------------

def load_processed_ids():
    if not os.path.exists(PROCESSED_POSTS_FILE): return set()
    with open(PROCESSED_POSTS_FILE, 'r') as f: return {line.strip() for line in f}

def save_processed_ids(ids_to_save):
    with open(PROCESSED_POSTS_FILE, 'a') as f:
        for post_id in ids_to_save: f.write(f"{post_id}\n")

# ---------------------------
# Reddit Scan Logic
# ---------------------------

def get_comments_for_post(post_id, limit=25):
    try:
        print(f"Fetching comments for post ID: {post_id}...")
        submission = reddit.submission(id=post_id)
        submission.comment_sort = "top"
        comments = [comment.body for comment in submission.comments[:limit] if not comment.stickied]
        time.sleep(1) 
        return "\n".join(comments)
    except Exception as e:
        print(f"Could not fetch comments for post {post_id}: {e}")
        return ""

def scrape_new_posts(subreddits, processed_ids):
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
    print(f"Sending post '{post_data['title']}' for AI synthesis...")
    system_prompt = """
    You are an expert retail investor... Your primary goal is to synthesize the original post with the community's reaction...
    """
    user_prompt = f"**Original Post Title:** {post_data['title']}\n\n**Original Post Body:**\n{post_data['selftext']}\n\n**Top Comments:**\n{comments_text}"
    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=0.5)
        analysis = json.loads(response.choices[0].message.content)
        required_keys = ['ticker', 'briefing', 'the_play', 'confidence_score']
        if all(key in analysis for key in required_keys): return analysis
        else: print("AI response was missing required keys."); return None
    except Exception as e:
        print(f"An error occurred during AI synthesis: {e}"); return None

def post_plays_to_discord(plays_data):
    if not plays_data: print("No plays to post to Discord."); return
    print(f"Posting {len(plays_data)} plays to Discord...")
    forum_webhook_url = WEBHOOKS.get("forum")
    if not forum_webhook_url: print("Warning: Discord forum webhook not set."); return
    for play in plays_data:
        score = float(play.get('confidence_score', 0.0))
        ticker = play.get('ticker', 'N/A')
        if score >= 8.0: tag_id, color, emoji = WEBHOOKS.get("tag_id_high"), 0x00C775, "🚀"
        elif score >= 4.0: tag_id, color, emoji = WEBHOOKS.get("tag_id_medium"), 0xFFFF00, "🤔"
        else: tag_id, color, emoji = WEBHOOKS.get("tag_id_low"), 0xFF0000, "⛔️"
        thread_name = f"{ticker} | Confidence: {score:.1f} | {emoji}"
        embed = {"title": play.get('title', 'No Title'), "description": play.get('briefing', 'No briefing available.'), "color": color, "fields": [{"name": "The Community-Vetted Play", "value": play.get('the_play', 'N/A'), "inline": False}, {"name": "Source", "value": f"r/{play.get('subreddit', 'N/A')}", "inline": True}, {"name": "Link", "value": f"[View Post]({play.get('url', '#')})", "inline": True}], "timestamp": datetime.now(timezone.utc).isoformat(), "footer": {"text": "Vulture Analysis"}}
        payload = {"thread_name": thread_name, "embeds": [embed], "applied_tags": [tag_id] if tag_id else []}
        try:
            requests.post(f"{forum_webhook_url}?wait=True", json=payload, timeout=15).raise_for_status()
            print(f"Successfully posted play for {ticker} to Discord.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to create Discord forum post for {ticker}: {e}")
        time.sleep(2)

def find_daily_discussion_thread():
    print("Searching for the daily discussion thread on r/wallstreetbets...")
    wsb = reddit.subreddit("wallstreetbets")
    query = "What Are Your Moves Tomorrow"
    for post in wsb.search(query, sort="new", time_filter="day", limit=5):
        if query.lower() in post.title.lower():
            print(f"Found daily discussion thread: {post.title}"); return post
    print("Daily discussion thread not found."); return None

def analyze_discussion_comments(post):
    print(f"Analyzing comments for '{post.title}'...")
    post.comment_sort = "top"
    comments = [comment.body for comment in post.comments[:50] if not comment.stickied]
    comments_text = "\n".join(comments)
    system_prompt = "You are a market sentiment analyst..."
    user_prompt = f"Here are the top comments:\n\n{comments_text}"
    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during comment analysis: {e}"); return None

def post_daily_summary(sentiment_summary):
    if not sentiment_summary: return
    print("Posting daily market summary...")
    news_webhook_url = WEBHOOKS.get("news")
    if not news_webhook_url: print("Warning: News webhook not set."); return
    embed = {"title": "Vulture Daily Market Briefing", "description": sentiment_summary, "color": 0x0077be, "footer": {"text": "For informational purposes only. Not financial advice."}, "timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        requests.post(news_webhook_url, json={"embeds": [embed]}, timeout=10).raise_for_status()
        print("Successfully posted daily summary.")
    except Exception as e:
        print(f"An error occurred posting the summary: {e}")

def run_reddit_scan():
    """Main pipeline for the Reddit community scan feature."""
    print("--- Vulture Reddit Scan triggered ---")
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    last_run_date_file = os.path.join(OUTPUT_DIR, "daily_summary_log.txt")
    last_run_date = ""
    if os.path.exists(last_run_date_file):
        with open(last_run_date_file, 'r') as f: last_run_date = f.read().strip()
    is_first_run_of_day = (last_run_date != today_str)
    processed_ids = load_processed_ids()
    new_posts = scrape_new_posts(TARGET_SUBREDDITS, processed_ids)
    if not new_posts: print("Scan finished: No new posts to process."); return
    analyzed_plays, newly_processed_ids = [], []
    for post in new_posts:
        comments = get_comments_for_post(post['id'])
        analysis = get_ai_synthesis(post, comments)
        if analysis:
            full_play_data = {**post, **analysis}
            if float(full_play_data.get('confidence_score', 0)) > 0 and full_play_data.get('ticker', 'N/A').upper() not in ['N/A', 'MULTI_STOCK']:
                analyzed_plays.append(full_play_data)
        newly_processed_ids.append(post['id'])
    if not analyzed_plays: print("No actionable plays found after analysis."); return
    df = pd.DataFrame(analyzed_plays)
    df.sort_values(by=['score', 'num_comments'], ascending=False, inplace=True)
    final_plays = df.drop_duplicates(subset=['title'], keep='first').to_dict(orient='records')
    post_plays_to_discord(final_plays)
    save_processed_ids(newly_processed_ids)
    if is_first_run_of_day:
        daily_thread = find_daily_discussion_thread()
        if daily_thread:
            sentiment_summary = analyze_discussion_comments(daily_thread)
            post_daily_summary(sentiment_summary)
        with open(last_run_date_file, 'w') as f: f.write(today_str)
        print("First run of the day complete.")
    print("--- Vulture Reddit Scan Complete ---")

# ---------------------------
# NEW: News Scan Logic
# ---------------------------
def run_news_scan():
    """Fetches general market news and appends it to a Google Sheet."""
    print("--- Vulture News Scan triggered ---")
    try:
        # Fetch the latest 50 general news articles
        news = finnhub_client.general_news('general', min_id=0)
        if not news:
            print("No news found from API."); return

        print(f"Fetched {len(news)} news articles.")
        
        # Prepare data for Google Sheets
        rows_to_append = []
        for article in news[:50]: # Limit to 50 articles
            rows_to_append.append([
                article.get('id'),
                datetime.fromtimestamp(article.get('datetime')).isoformat(),
                article.get('headline'),
                article.get('summary'),
                article.get('source'),
                article.get('url')
            ])
        
        # Append to Google Sheet
        gc = get_gspread_client()
        sheet_name = os.getenv("GOOGLE_NEWS_SHEET_NAME")
        worksheet = gc.open(sheet_name).sheet1
        worksheet.append_rows(rows_to_append)
        print(f"Successfully appended {len(rows_to_append)} news articles to '{sheet_name}'.")

    except Exception as e:
        print(f"An error occurred during the news scan: {e}")

# ---------------------------
# NEW: Economic Calendar Logic
# ---------------------------
def run_calendar_scan():
    """Fetches the economic calendar for the next 7 days and updates a Google Sheet."""
    print("--- Vulture Economic Calendar Scan triggered ---")
    try:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        one_week_from_now = (datetime.now(timezone.utc) + timedelta(days=7)).strftime('%Y-%m-%d')
        
        calendar = finnhub_client.economic_calendar(from_date=today, to_date=one_week_from_now)
        events = calendar.get('economicCalendar', [])
        
        if not events:
            print("No economic events found for the upcoming week."); return

        print(f"Fetched {len(events)} economic events for the next 7 days.")
        
        # Prepare data for Google Sheets
        rows_to_append = []
        for event in events:
            rows_to_append.append([
                event.get('time'),
                event.get('country'),
                event.get('event'),
                event.get('impact'),
                event.get('estimate'),
                event.get('actual'),
                event.get('prev')
            ])
            
        # Update Google Sheet (Clear and then write)
        gc = get_gspread_client()
        sheet_name = os.getenv("GOOGLE_CALENDAR_SHEET_NAME")
        worksheet = gc.open(sheet_name).sheet1
        worksheet.clear()
        # Add a header row
        header = ["Time", "Country", "Event", "Impact", "Estimate", "Actual", "Previous"]
        worksheet.append_row(header)
        worksheet.append_rows(rows_to_append)
        print(f"Successfully updated '{sheet_name}' with {len(rows_to_append)} events.")

    except Exception as e:
        print(f"An error occurred during the calendar scan: {e}")


# ---------------------------
# Entry point for command-line execution
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vulture: A multi-source market intelligence tool.")
    parser.add_argument('scan_type', choices=['reddit', 'news', 'calendar'], help="The type of scan to run.")
    
    args = parser.parse_args()

    if args.scan_type == 'reddit':
        run_reddit_scan()
    elif args.scan_type == 'news':
        run_news_scan()
    elif args.scan_type == 'calendar':
        run_calendar_scan()
