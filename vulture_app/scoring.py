# vulture_app/scoring.py
import re

def score_thesis_clarity(text: str) -> float:
    """
    0–5 points:
      • 0–2 for Price Levels
      • 0–1 for Action keywords
      • 0–1 for Timeline
      • 0–1 for Justification
    """
    score = 0.0
    lower = text.lower()

    # Price Levels (0–2)
    prices = set(re.findall(r"\$\d+(?:\.\d+)?", text))
    if len(prices) >= 2 or re.search(r"\bfrom\s+\$\d+.*\sto\s+\$\d+", lower):
        score += 2.0
    elif len(prices) == 1:
        score += 1.0

    # Action keywords (0–1)
    if re.search(r"\b(buy|sell|go long|go short|call|put|position)\b", lower):
        score += 1.0

    # Timeline (0–1)
    if re.search(r"\b(by|into|before|after)\b.*\b(am|pm|day|week|month|earnings|fri|today|tomorrow)\b", lower):
        score += 1.0

    # Justification (0–1)
    if re.search(r"\b(because|due to|as a result|suggests|indicates|based on)\b", lower):
        score += 1.0

    return min(score, 5.0)


def score_catalyst_present(text: str) -> float:
    """
    0–1.5 points:
      • 1.5 if catalyst keyword plus explicit timing
      • 0.75 if just a catalyst keyword
      • 0 otherwise
    """
    keys = ["earnings", "fda", "merger", "acquisition", "guidance", "split", "dividend"]
    lower = text.lower()
    if not any(k in lower for k in keys):
        return 0.0
    # explicit timing nearby?
    if re.search(r"(earnings|fda|merger).{0,30}\d{1,2}(am|pm)?", lower):
        return 1.5
    return 0.75


def score_formatting_quality(text: str) -> float:
    """
    0–0.5 points:
      • 0.5 if any structure (double-newline, headers, or lists)
      • 0 otherwise
    """
    if "\n\n" in text or \
       re.search(r"^#{1,3}\s+", text, flags=re.MULTILINE) or \
       re.search(r"^- ", text, flags=re.MULTILINE):
        return 0.5
    return 0.0


def score_image_insight(caption: str) -> float:
    """
    0–0.5 points if the caption mentions $ amounts or position keywords
    """
    lower = caption.lower()
    if re.search(r"\$\d+|shares?|calls?|puts?|contracts?", lower):
        return 0.5
    return 0.0


def generate_engagement_score(comments: list[str], post_upvotes: int, num_comments: int) -> float:
    """
    0–1 point:
      • 1.0 if upvote/comment ratio high AND positive keywords
      • 0.5 if moderate engagement
      • 0   otherwise
    """
    ratio = post_upvotes / max(num_comments, 1)
    snippet = "\n\n".join(comments[:5])
    score = 0.0
    if ratio >= 2 and re.search(r"\b(solid dd|good bot|i'm in|great post)\b", snippet.lower()):
        score = 1.0
    elif ratio >= 1:
        score = 0.5
    return score


def score_position_disclosure(body: str, comments: list[str]) -> float:
    """
    0–0.5 bonus for first-person position disclosures in post or top comments.
    """
    patterns = [
        r"\bi['’]?m (long|short|in)\b",
        r"\bmy position\b",
        r"\bi (?:have|hold|own)\b.*\$\d+",
        r"\$\d+[kKmM]\b.*\b(my|i)\b",
    ]
    lower_body = body.lower()
    for pat in patterns:
        if re.search(pat, lower_body):
            return 0.5

    for c in comments:
        for pat in patterns:
            if re.search(pat, c.lower()):
                return 0.5

    return 0.0


def calculate_confidence_score(post_data: dict) -> float:
    """
    Aggregate sub-scores into 0–10 confidence.
    """
    # Thesis, Catalyst, Engagement, Format, Image
    thesis = min(post_data.get("thesis_clarity", 0), 3.0)
    catalyst = min(post_data.get("catalyst_present", 0), 1.5)
    engage = min(post_data.get("engagement_quality", 0), 1.0)
    fmt = min(post_data.get("formatting_quality", 0), 0.5)
    img = min(post_data.get("image_insight", 0), 0.5)

    # Author credibility
    age = post_data.get("account_age_years", 0)
    if age >= 3:
        age_score = 1.0
    elif age >= 1:
        age_score = 0.5
    else:
        age_score = 0.0

    karma = post_data.get("total_karma", 0)
    if karma >= 1000:
        karma_score = 1.0
    elif karma >= 100:
        karma_score = 0.5
    else:
        karma_score = 0.0

    history = post_data.get("finance_post_count", 0)
    if history >= 3:
        history_score = 1.0
    elif history > 0:
        history_score = 0.5
    else:
        history_score = 0.0

    cred = age_score + karma_score + history_score  # max 3

    total = thesis + catalyst + engage + fmt + img + cred
    return round(min(total, 10.0), 1)
