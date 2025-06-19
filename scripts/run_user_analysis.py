#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from vulture_app.pipeline import run_user_analysis

load_dotenv()

def main():
    users = os.getenv('USERNAMES', '')
    names = [u.strip() for u in users.split(',') if u.strip()]
    result = run_user_analysis(names)
    print(result)

if __name__ == '__main__':
    main()
