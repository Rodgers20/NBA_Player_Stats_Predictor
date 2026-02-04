# utils/injury_news.py
"""
Injury & News Data Module
=========================
This module fetches player injury statuses and relevant news that could
affect predictions.

WHY THIS MATTERS FOR PREDICTIONS:
---------------------------------
1. Injured players won't play (obviously) - predict 0 stats
2. Players returning from injury often have minutes restrictions
3. If a star teammate is out, other players may see more usage
4. Back injuries, leg injuries affect performance differently

DATA SOURCES (all free):
------------------------
1. NBA API - Has basic injury/inactive status for current games
2. ESPN RSS - Real-time news feed with injury reports
3. Web scraping - For comprehensive injury lists (as backup)

NOTE ON WEB SCRAPING:
Web scraping can break if the website changes its structure.
We handle errors gracefully and fall back to other sources.
"""

import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# Headers to mimic a browser - many sites block requests without proper headers
# This is a common and acceptable practice for public RSS feeds
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# For NBA API injury data
try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False


# =============================================================================
# ESPN RSS FEED - Real-time NBA news
# =============================================================================

def fetch_rss_feed(url: str, limit: int = 20) -> list[dict]:
    """
    Generic RSS feed fetcher that works with multiple sources.

    HOW RSS WORKS:
    RSS (Really Simple Syndication) is an XML format for news feeds.
    Structure: <rss><channel><item>...</item><item>...</item></channel></rss>
    Each <item> is one article with title, description, link, pubDate.

    Args:
        url: The RSS feed URL
        limit: Maximum number of articles to return

    Returns:
        List of dictionaries with keys: title, description, link, published, source
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        channel = root.find("channel")

        if channel is None:
            # Try Atom format (different XML structure)
            # Atom uses <feed><entry>...</entry></feed>
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if entries:
                news_items = []
                for entry in entries[:limit]:
                    news_items.append({
                        "title": entry.findtext("{http://www.w3.org/2005/Atom}title", ""),
                        "description": entry.findtext("{http://www.w3.org/2005/Atom}summary", ""),
                        "link": entry.find("{http://www.w3.org/2005/Atom}link").get("href", "") if entry.find("{http://www.w3.org/2005/Atom}link") is not None else "",
                        "published": entry.findtext("{http://www.w3.org/2005/Atom}updated", ""),
                        "source": url
                    })
                return news_items
            return []

        news_items = []
        for item in channel.findall("item")[:limit]:
            news_items.append({
                "title": item.findtext("title", ""),
                "description": item.findtext("description", ""),
                "link": item.findtext("link", ""),
                "published": item.findtext("pubDate", ""),
                "source": url
            })

        return news_items

    except requests.RequestException as e:
        # Silently fail - we have multiple sources
        return []
    except ET.ParseError as e:
        return []


# List of NBA news RSS feeds to try (in order of preference)
NBA_RSS_FEEDS = [
    "https://www.cbssports.com/rss/headlines/nba/",
    "https://www.espn.com/espn/rss/nba/news",
    "https://www.rotowire.com/rss/news.php?sport=NBA",
]


def get_espn_nba_news(limit: int = 20) -> list[dict]:
    """
    Fetch latest NBA news from multiple RSS sources.

    WHY MULTIPLE SOURCES:
    Not all RSS feeds are always available. By trying multiple sources,
    we increase reliability. We combine results and deduplicate.

    Args:
        limit: Maximum number of articles to return

    Returns:
        List of dictionaries with keys: title, description, link, published
    """
    all_news = []

    # Try each feed until we get enough articles
    for feed_url in NBA_RSS_FEEDS:
        news = fetch_rss_feed(feed_url, limit=limit)
        all_news.extend(news)

        # If we have enough, stop
        if len(all_news) >= limit:
            break

    # Remove duplicates based on title similarity
    seen_titles = set()
    unique_news = []
    for item in all_news:
        # Normalize title for comparison
        title_key = item["title"].lower()[:50]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(item)

    return unique_news[:limit]


def search_player_news(player_name: str, news_items: list[dict] = None) -> list[dict]:
    """
    Search news items for mentions of a specific player.

    HOW THE SEARCH WORKS:
    We look for the player's name (or parts of it) in the title and description.
    This catches articles like "LeBron James questionable for Tuesday's game"

    Args:
        player_name: Full name like "LeBron James"
        news_items: Pre-fetched news (if None, fetches fresh)

    Returns:
        List of news items mentioning the player
    """
    if news_items is None:
        news_items = get_espn_nba_news(limit=50)

    # Split name into parts for flexible matching
    # "LeBron James" -> ["LeBron", "James"]
    name_parts = player_name.lower().split()

    matching_news = []
    for item in news_items:
        text = (item["title"] + " " + item["description"]).lower()

        # Check if all parts of the name appear in the text
        # This handles cases like "James scores 30" or "LeBron questionable"
        if all(part in text for part in name_parts):
            matching_news.append(item)

    return matching_news


# =============================================================================
# INJURY STATUS DETECTION
# =============================================================================

# Keywords that indicate injury-related news
INJURY_KEYWORDS = [
    "injury", "injured", "out", "questionable", "doubtful", "probable",
    "day-to-day", "week-to-week", "sidelined", "miss", "ruled out",
    "surgery", "sprain", "strain", "fracture", "concussion", "illness",
    "rest", "load management", "dnp", "did not play", "inactive",
    "return", "returning", "back", "cleared", "healthy"
]

# Keywords indicating player will NOT play
OUT_KEYWORDS = [
    "out", "ruled out", "miss", "sidelined", "will not play",
    "surgery", "inactive", "dnp", "shut down"
]

# Keywords indicating uncertain status
QUESTIONABLE_KEYWORDS = [
    "questionable", "doubtful", "game-time decision", "uncertain",
    "day-to-day", "monitor"
]

# Keywords indicating player is healthy/returning
HEALTHY_KEYWORDS = [
    "return", "cleared", "healthy", "back in lineup", "will play",
    "probable", "available"
]


def analyze_injury_status(news_items: list[dict]) -> dict:
    """
    Analyze news items to determine a player's likely injury status.

    HOW THIS WORKS:
    We scan the news text for keywords and assign a status:
    - "OUT": Strong indicators player won't play
    - "QUESTIONABLE": Uncertain status
    - "HEALTHY": Player is available or returning
    - "UNKNOWN": No clear indicators

    WHY WE DO THIS:
    For predictions, we need to know:
    1. Will the player play? (if OUT, predict 0)
    2. Are they on a minutes restriction? (reduce predictions)
    3. Is a teammate out? (may increase player's usage)

    Args:
        news_items: List of news items about a player

    Returns:
        Dictionary with status, confidence, and reasons
    """
    if not news_items:
        return {
            "status": "UNKNOWN",
            "confidence": 0.0,
            "reasons": ["No recent news found"],
            "latest_news": None
        }

    # Analyze the most recent news first (more relevant)
    all_text = " ".join([
        (item["title"] + " " + item["description"]).lower()
        for item in news_items[:5]  # Focus on most recent
    ])

    out_count = sum(1 for kw in OUT_KEYWORDS if kw in all_text)
    questionable_count = sum(1 for kw in QUESTIONABLE_KEYWORDS if kw in all_text)
    healthy_count = sum(1 for kw in HEALTHY_KEYWORDS if kw in all_text)

    # Determine status based on keyword counts
    # More recent news about returning can override older "out" news
    reasons = []

    if out_count > healthy_count and out_count > 0:
        status = "OUT"
        confidence = min(0.5 + (out_count * 0.1), 0.95)
        reasons.append(f"Found {out_count} indicators of being out")
    elif questionable_count > 0 and healthy_count == 0:
        status = "QUESTIONABLE"
        confidence = 0.5 + (questionable_count * 0.1)
        reasons.append(f"Found {questionable_count} uncertainty indicators")
    elif healthy_count > out_count:
        status = "HEALTHY"
        confidence = min(0.5 + (healthy_count * 0.1), 0.9)
        reasons.append(f"Found {healthy_count} positive indicators")
    else:
        status = "UNKNOWN"
        confidence = 0.3
        reasons.append("Mixed or unclear signals")

    return {
        "status": status,
        "confidence": confidence,
        "reasons": reasons,
        "latest_news": news_items[0] if news_items else None
    }


def get_player_injury_status(player_name: str) -> dict:
    """
    Get the current injury status for a player.

    This is the MAIN FUNCTION to use - it combines:
    1. Fetching recent news
    2. Filtering for player mentions
    3. Analyzing injury status

    Args:
        player_name: Full name like "LeBron James"

    Returns:
        Dictionary with:
        - status: "OUT", "QUESTIONABLE", "HEALTHY", or "UNKNOWN"
        - confidence: 0.0 to 1.0 confidence score
        - reasons: List of reasons for the determination
        - news: List of relevant news items
        - checked_at: Timestamp of when we checked

    Example:
        >>> get_player_injury_status("Anthony Davis")
        {
            "status": "QUESTIONABLE",
            "confidence": 0.7,
            "reasons": ["Found 2 uncertainty indicators"],
            "news": [...],
            "checked_at": "2025-02-04T10:30:00"
        }
    """
    # Fetch fresh news
    all_news = get_espn_nba_news(limit=50)

    # Filter for this player
    player_news = search_player_news(player_name, all_news)

    # Analyze status
    analysis = analyze_injury_status(player_news)

    return {
        "player": player_name,
        "status": analysis["status"],
        "confidence": analysis["confidence"],
        "reasons": analysis["reasons"],
        "news": player_news[:5],  # Return top 5 relevant articles
        "checked_at": datetime.now().isoformat()
    }


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

def get_injury_report(player_names: list[str]) -> pd.DataFrame:
    """
    Get injury status for multiple players at once.

    WHY BATCH PROCESSING:
    When predicting for a game, we need to check ALL players at once.
    This function does that efficiently.

    Args:
        player_names: List of player names to check

    Returns:
        DataFrame with columns: player, status, confidence, latest_headline
    """
    # Fetch news once (more efficient than fetching per player)
    all_news = get_espn_nba_news(limit=100)

    results = []
    for player in player_names:
        player_news = search_player_news(player, all_news)
        analysis = analyze_injury_status(player_news)

        results.append({
            "player": player,
            "status": analysis["status"],
            "confidence": analysis["confidence"],
            "latest_headline": player_news[0]["title"] if player_news else "No recent news"
        })

    return pd.DataFrame(results)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_player_available(player_name: str) -> tuple[bool, str]:
    """
    Simple check: Is this player likely to play?

    Returns:
        Tuple of (is_available: bool, reason: str)

    Example:
        >>> is_player_available("Kevin Durant")
        (True, "No injury news found - assumed healthy")

        >>> is_player_available("Ja Morant")
        (False, "OUT - Found indicators of being out")
    """
    status = get_player_injury_status(player_name)

    if status["status"] == "OUT":
        return False, f"OUT - {status['reasons'][0]}"
    elif status["status"] == "QUESTIONABLE":
        return True, f"QUESTIONABLE - {status['reasons'][0]} (may not play)"
    else:
        return True, "No injury news found - assumed healthy"


def extract_injury_type(news_text: str) -> Optional[str]:
    """
    Try to extract the specific injury type from news text.

    WHY THIS MATTERS:
    Different injuries affect performance differently:
    - Ankle sprain: May return but with reduced mobility
    - Finger injury: May affect shooting
    - Back injury: Can affect everything

    Args:
        news_text: Text from news article

    Returns:
        Injury type if found, None otherwise
    """
    injury_patterns = [
        r"(ankle\s*(sprain|injury))",
        r"(knee\s*(injury|sprain|soreness))",
        r"(hamstring\s*(strain|injury))",
        r"(back\s*(spasms|soreness|injury))",
        r"(shoulder\s*(injury|soreness))",
        r"(concussion)",
        r"(illness)",
        r"(calf\s*(strain|injury))",
        r"(quad\s*(strain|injury))",
        r"(groin\s*(strain|injury))",
        r"(finger\s*(injury|sprain))",
        r"(wrist\s*(injury|sprain))",
        r"(foot\s*(injury|soreness))",
    ]

    text_lower = news_text.lower()
    for pattern in injury_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0)

    return None
