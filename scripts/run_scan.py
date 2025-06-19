#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from vulture_app.pipeline import run_scraper

load_dotenv()

def main():
    subs = os.getenv('SUBREDDITS', '')
    subs_list = [s.strip() for s in subs.split(',') if s.strip()]
    path = run_scraper(subs_list)
    print(f"Scan complete: {path}")

if __name__ == '__main__':
    main()
