# vulture_app/reddit_client.py
import os
import praw
from .cache import prune_cache

# PRAW setup (env-vars loaded by scripts via python-dotenv)
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT'),
)

def fetch_posts(subreddit: str, post_type: str = 'top', limit: int = 50):
    sr = reddit.subreddit(subreddit)
    out, after = [], None
    while len(out) < limit:
        gen = (
            sr.top(limit=limit, params={'after': after})
            if post_type == 'top' else
            sr.new(limit=limit, params={'after': after})
        )
        batch = list(gen)
        if not batch:
            break
        after = batch[-1].name
        out.extend(batch)
    return out[:limit]

def fetch_comments_for_post(post):
    post.comments.replace_more(limit=0)
    flat = post.comments.list()
    bodies = [c.body for c in flat if hasattr(c, 'body')]
    total = len(bodies)
    sorted_comments = sorted(flat, key=lambda c: getattr(c, 'score', 0), reverse=True)
    half = max(1, total // 2)
    return [c.body for c in sorted_comments[:half]], total
