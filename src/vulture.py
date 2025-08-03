import os
import sys
import json
import re
import argparse
import time
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import io

import pandas as pd
import praw
import requests
from dotenv import load_dotenv
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
import finnhub
from alpha_vantage.fundamentaldata import FundamentalData

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
        "GOOGLE_NEWS_SHEET_NAME", "GOOGLE_CALENDAR_SHEET_NAME", "FINNHUB_API_KEY",
        "ALPHA_VANTAGE_API_KEY"
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
alpha_vantage_client = FundamentalData(key=os.getenv("ALPHA_VANTAGE_API_KEY"))


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

def write_to_sheet(spreadsheet_name, worksheet_name, rows_to_append, clear_sheet=False, header=None):
    """A robust function to write data to a specific worksheet (tab) within a spreadsheet."""
    if not rows_to_append:
        print(f"No data to write to sheet '{spreadsheet_name} -> {worksheet_name}'.")
        return
    
    print(f"Attempting to write {len(rows_to_append)} rows to Google Sheet: {spreadsheet_name} -> {worksheet_name}...")
    try:
        gc = get_gspread_client()
        spreadsheet = gc.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet(worksheet_name)
        
        if clear_sheet:
            worksheet.clear()
            if header:
                worksheet.append_row(header)
        
        worksheet.append_rows(rows_to_append)
        print(f"Successfully wrote data to '{spreadsheet_name} -> {worksheet_name}'.")
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet '{spreadsheet_name}' not found. Please check the name and sharing permissions.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet (tab) '{worksheet_name}' not found in the spreadsheet '{spreadsheet_name}'.")
    except gspread.exceptions.APIError as e:
        print(f"ERROR: An API error occurred with Google Sheets: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing to Google Sheets: {e}")
        traceback.print_exc()


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

    Your entire response must be a single, valid JSON object. Do not include any other text, explanations, or markdown.
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
        return response.choices[0].message.content.strip(), comments
    except Exception as e:
        print(f"An error occurred during comment analysis: {e}"); return None, []

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
    
    if not analyzed_plays: 
        print("No actionable plays found after analysis."); 
    else:
        df = pd.DataFrame(analyzed_plays)
        df.sort_values(by=['score', 'num_comments'], ascending=False, inplace=True)
        final_plays = df.drop_duplicates(subset=['title'], keep='first').to_dict(orient='records')
        
        post_plays_to_discord(final_plays)
        
        rows_to_save = []
        for play in final_plays:
            rows_to_save.append([
                play.get('id'), play.get('ticker'), play.get('briefing'),
                play.get('the_play'), play.get('confidence_score'),
                play.get('url'), play.get('subreddit'), play.get('created_utc'),
                datetime.now(timezone.utc).isoformat()
            ])
        write_to_sheet(os.getenv("GOOGLE_SHEET_NAME"), "Vulture Data", rows_to_save)

    save_processed_ids(newly_processed_ids)
    
    if is_first_run_of_day:
        daily_thread = find_daily_discussion_thread()
        if daily_thread:
            sentiment_summary, raw_comments = analyze_discussion_comments(daily_thread)
            if sentiment_summary:
                post_daily_summary(sentiment_summary)
                training_rows = [[
                    daily_thread.id, daily_thread.title, "\n".join(raw_comments),
                    sentiment_summary, datetime.now(timezone.utc).isoformat()
                ]]
                write_to_sheet(os.getenv("GOOGLE_SHEET_NAME"), os.getenv("GOOGLE_TRAINING_SHEET_NAME"), training_rows)
        with open(last_run_date_file, 'w') as f: f.write(today_str)
        print("First run of the day complete.")
    print("--- Vulture Reddit Scan Complete ---")

# ---------------------------
# News Scan Logic
# ---------------------------
def run_news_scan():
    """Fetches general market news and appends it to a Google Sheet."""
    print("--- Vulture News Scan triggered ---")
    try:
        print("Fetching news from Finnhub...")
        news = finnhub_client.general_news('general', min_id=0)
        if not news or not isinstance(news, list):
            print("No news found or unexpected format from API."); return
        print(f"Fetched {len(news)} news articles.")
        
        rows_to_append = []
        for article in news[:50]:
            dt = article.get('datetime')
            iso_date = datetime.fromtimestamp(dt).isoformat() if dt else None
            rows_to_append.append([
                article.get('id'), iso_date, article.get('headline'),
                article.get('summary'), article.get('source'), article.get('url')
            ])
        
        spreadsheet_name = os.getenv("GOOGLE_SHEET_NAME")
        worksheet_name = os.getenv("GOOGLE_NEWS_SHEET_NAME")
        write_to_sheet(spreadsheet_name, worksheet_name, rows_to_append)

    except finnhub.FinnhubAPIException as e:
        print(f"Finnhub API error during news scan: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during the news scan: {e}")
        traceback.print_exc()


# ---------------------------
# Economic Calendar Logic
# ---------------------------
def post_weekly_earnings_summary(earnings_data):
    """Formats and posts the weekly earnings summary to Discord."""
    if not earnings_data:
        print("No earnings data to post.")
        return

    news_webhook_url = WEBHOOKS.get("news")
    if not news_webhook_url:
        print("Warning: News webhook not set. Skipping earnings post.")
        return

    earnings_by_day = {}
    for item in earnings_data:
        report_date = item.get('reportDate', 'Unknown Date')
        day_name = datetime.strptime(report_date, '%Y-%m-%d').strftime('%A, %B %d')
        if day_name not in earnings_by_day:
            earnings_by_day[day_name] = []
        earnings_by_day[day_name].append(item)

    description = ""
    for day, earnings in sorted(earnings_by_day.items()):
        description += f"**{day}**\n"
        for item in earnings[:5]:
            description += f"- **{item.get('symbol')}** ({item.get('hour', 'N/A').upper()})\n"
        if len(earnings) > 5:
            description += "- ...and more\n"
        description += "\n"

    embed = {
        "title": "📅 Weekly Earnings Outlook",
        "description": description.strip(),
        "color": 0x4A90E2,
        "footer": {"text": "Data from Alpha Vantage"},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        requests.post(news_webhook_url, json={"embeds": [embed]}, timeout=10).raise_for_status()
        print("Successfully posted weekly earnings summary to Discord.")
    except Exception as e:
        print(f"An error occurred posting the earnings summary: {e}")


def run_calendar_scan():
    """Fetches economic and earnings calendars and updates Google Sheets/Discord."""
    print("--- Vulture Calendar Scan triggered ---")
    
    is_monday = (datetime.now(timezone.utc).weekday() == 0)

    if is_monday:
        try:
            print("Fetching weekly earnings calendar from Alpha Vantage...")
            data, _ = alpha_vantage_client.get_earnings_calendar(horizon='7day')
            df_earnings = pd.read_csv(io.StringIO(data))

            if not df_earnings.empty:
                print(f"Fetched {len(df_earnings)} upcoming earnings reports.")
                post_weekly_earnings_summary(df_earnings.to_dict(orient='records'))
            else:
                print("No upcoming earnings found for the next 7 days.")
        except Exception as e:
            print(f"An unexpected error occurred during the earnings calendar scan: {e}")
            traceback.print_exc()

    # **FIX**: Replaced the non-functional economic calendar with a direct API call.
    try:
        print("Fetching general economic calendar from Alpha Vantage...")
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        # Alpha Vantage provides this data as a direct CSV download
        url = f'https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&horizon=3month&apikey={api_key}'
        r = requests.get(url)
        r.raise_for_status()
        
        df_econ = pd.read_csv(io.StringIO(r.text))

        if df_econ.empty:
            print("No economic events found from Alpha Vantage."); return

        print(f"Fetched {len(df_econ)} economic events.")
        
        # Filter for the next 7 days
        df_econ['releaseTime'] = pd.to_datetime(df_econ['releaseTime'])
        today = datetime.now(timezone.utc)
        one_week_from_now = today + timedelta(days=7)
        df_filtered = df_econ[
            (df_econ['releaseTime'] >= today) & 
            (df_econ['releaseTime'] <= one_week_from_now)
        ].copy()

        if df_filtered.empty:
            print("No events found for the upcoming week after filtering.")
            return

        # Prepare data for Google Sheets
        df_filtered['releaseTime'] = df_filtered['releaseTime'].astype(str)
        rows_to_append = df_filtered.values.tolist()
        header = df_filtered.columns.tolist()
        spreadsheet_name = os.getenv("GOOGLE_SHEET_NAME")
        worksheet_name = os.getenv("GOOGLE_CALENDAR_SHEET_NAME")
        write_to_sheet(spreadsheet_name, worksheet_name, rows_to_append, clear_sheet=True, header=header)

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch economic calendar from Alpha Vantage API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the economic calendar scan: {e}")
        traceback.print_exc()


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
