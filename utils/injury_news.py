# utils/injury_news.py
"""
Injury & News Data Module
=========================
This module fetches player injury statuses and relevant news that could
affect predictions.

DATA SOURCES:
-------------
1. Underdog Fantasy (primary for injury news)
2. Rotowire (comprehensive injury reports)
3. ESPN RSS - Real-time news feed
4. CBS Sports RSS

All news items include clickable source links.
"""

import re
import requests
import json
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, application/json, */*",
}

# For NBA API injury data
try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False


# =============================================================================
# NEWS SOURCES CONFIGURATION
# =============================================================================

# RSS feeds for NBA news
NBA_RSS_FEEDS = [
    {
        "url": "https://www.rotowire.com/basketball/news.php",
        "name": "Rotowire",
        "type": "scrape"
    },
    {
        "url": "https://www.cbssports.com/rss/headlines/nba/",
        "name": "CBS Sports",
        "type": "rss"
    },
    {
        "url": "https://www.espn.com/espn/rss/nba/news",
        "name": "ESPN",
        "type": "rss"
    },
]

# Underdog Fantasy Twitter/X - We'll fetch from their Nitter mirror or API
UNDERDOG_SOURCES = [
    {
        "url": "https://nitter.net/Underdog__NBA/rss",
        "name": "Underdog Fantasy",
        "type": "rss",
        "twitter_url": "https://twitter.com/Underdog__NBA"
    },
    {
        "url": "https://nitter.poast.org/Underdog__NBA/rss",
        "name": "Underdog Fantasy",
        "type": "rss",
        "twitter_url": "https://twitter.com/Underdog__NBA"
    },
]

# Other reliable injury sources
INJURY_SOURCES = [
    {
        "url": "https://www.fantasylabs.com/api/nba/injuries",
        "name": "FantasyLabs",
        "type": "api"
    },
]


# =============================================================================
# RSS FEED FETCHING
# =============================================================================

def fetch_rss_feed(url: str, limit: int = 20) -> list[dict]:
    """
    Generic RSS feed fetcher.
    Returns list of news items with title, description, link, published, source.
    """
    import xml.etree.ElementTree as ET

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        channel = root.find("channel")

        if channel is None:
            # Try Atom format
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if entries:
                news_items = []
                for entry in entries[:limit]:
                    link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                    news_items.append({
                        "title": entry.findtext("{http://www.w3.org/2005/Atom}title", ""),
                        "description": entry.findtext("{http://www.w3.org/2005/Atom}summary", ""),
                        "link": link_elem.get("href", "") if link_elem is not None else "",
                        "published": entry.findtext("{http://www.w3.org/2005/Atom}updated", ""),
                        "source": "RSS Feed",
                        "source_url": url
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
                "source": "RSS Feed",
                "source_url": url
            })

        return news_items

    except Exception:
        return []


def fetch_underdog_news(limit: int = 30) -> list[dict]:
    """
    Fetch injury news from Underdog Fantasy Twitter via Nitter RSS.
    Falls back to other sources if unavailable.
    """
    for source in UNDERDOG_SOURCES:
        try:
            items = fetch_rss_feed(source["url"], limit)
            if items:
                # Add proper source attribution
                for item in items:
                    item["source"] = "Underdog Fantasy"
                    item["source_url"] = source.get("twitter_url", source["url"])
                    # Convert Nitter links to Twitter links
                    if "nitter" in item.get("link", ""):
                        item["link"] = item["link"].replace("nitter.net", "twitter.com")
                        item["link"] = item["link"].replace("nitter.poast.org", "twitter.com")
                return items
        except Exception:
            continue
    return []


def fetch_rotowire_news(limit: int = 30) -> list[dict]:
    """
    Fetch injury news from Rotowire (known for accurate injury reports).
    """
    try:
        url = "https://www.rotowire.com/basketball/news.php"
        response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            return []

        # Parse HTML for news items
        from html.parser import HTMLParser

        news_items = []
        content = response.text

        # Simple regex extraction for news headlines
        # Rotowire has structured news items
        pattern = r'<a[^>]*href="([^"]*news[^"]*)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, content)

        for link, title in matches[:limit]:
            if any(kw in title.lower() for kw in ["injury", "out", "questionable", "doubtful", "return"]):
                full_link = f"https://www.rotowire.com{link}" if link.startswith("/") else link
                news_items.append({
                    "title": title.strip(),
                    "description": title.strip(),
                    "link": full_link,
                    "published": datetime.now().isoformat(),
                    "source": "Rotowire",
                    "source_url": "https://www.rotowire.com"
                })

        return news_items
    except Exception:
        return []


# =============================================================================
# MAIN NEWS AGGREGATION
# =============================================================================

def get_nba_injury_news(limit: int = 50) -> list[dict]:
    """
    Aggregate injury news from all sources.
    Prioritizes Underdog Fantasy, then Rotowire, then other sources.

    Returns:
        List of news items with clickable links
    """
    all_news = []

    # 1. Try Underdog Fantasy first (most reliable for props)
    underdog_news = fetch_underdog_news(limit=20)
    all_news.extend(underdog_news)

    # 2. Try Rotowire (known for accurate injury reports)
    rotowire_news = fetch_rotowire_news(limit=20)
    all_news.extend(rotowire_news)

    # 3. Fallback to RSS feeds
    for feed in NBA_RSS_FEEDS:
        if feed["type"] == "rss":
            items = fetch_rss_feed(feed["url"], limit=15)
            for item in items:
                item["source"] = feed["name"]
                item["source_url"] = feed["url"]
            all_news.extend(items)

    # Remove duplicates based on title similarity
    seen_titles = set()
    unique_news = []
    for item in all_news:
        title_key = item["title"].lower()[:50]
        if title_key not in seen_titles and item["title"]:
            seen_titles.add(title_key)
            unique_news.append(item)

    return unique_news[:limit]


def get_espn_nba_news(limit: int = 20) -> list[dict]:
    """Legacy function - now uses aggregated sources."""
    return get_nba_injury_news(limit)


# =============================================================================
# PLAYER-SPECIFIC NEWS SEARCH
# =============================================================================

def search_player_news(player_name: str, news_items: list[dict] = None) -> list[dict]:
    """
    Search news items for mentions of a specific player.
    Returns news items with clickable source links.
    """
    if news_items is None:
        news_items = get_nba_injury_news(limit=100)

    # Split name for flexible matching
    name_parts = player_name.lower().split()
    last_name = name_parts[-1] if name_parts else ""

    matching_news = []
    for item in news_items:
        text = (item.get("title", "") + " " + item.get("description", "")).lower()

        # Check for full name or last name match
        if all(part in text for part in name_parts) or (last_name and last_name in text):
            matching_news.append(item)

    return matching_news


# =============================================================================
# INJURY STATUS DETECTION
# =============================================================================

# Keywords for status classification
OUT_KEYWORDS = [
    "out", "ruled out", "will miss", "sidelined", "will not play",
    "surgery", "inactive", "dnp", "shut down", "out indefinitely",
    "season-ending", "out for"
]

DOUBTFUL_KEYWORDS = [
    "doubtful", "unlikely to play", "not expected to play"
]

QUESTIONABLE_KEYWORDS = [
    "questionable", "game-time decision", "uncertain", "day-to-day",
    "50/50", "monitor", "listed as questionable", "gtd"
]

PROBABLE_KEYWORDS = [
    "probable", "expected to play", "likely to play", "should play"
]

HEALTHY_KEYWORDS = [
    "return", "cleared", "healthy", "back in lineup", "will play",
    "available", "returning", "activated", "off injury report"
]


def analyze_injury_status(news_items: list[dict]) -> dict:
    """
    Analyze news items to determine a player's injury status.

    Returns:
        Dictionary with status, confidence, reason, and news items
    """
    if not news_items:
        return {
            "status": "ACTIVE",  # Default to active if no news
            "confidence": 0.5,
            "reason": "No injury news found",
            "injury_type": None,
            "latest_news": None
        }

    # Analyze recent news (most recent is most relevant)
    recent_text = " ".join([
        (item.get("title", "") + " " + item.get("description", "")).lower()
        for item in news_items[:5]
    ])

    # Count keyword matches
    out_count = sum(1 for kw in OUT_KEYWORDS if kw in recent_text)
    doubtful_count = sum(1 for kw in DOUBTFUL_KEYWORDS if kw in recent_text)
    questionable_count = sum(1 for kw in QUESTIONABLE_KEYWORDS if kw in recent_text)
    probable_count = sum(1 for kw in PROBABLE_KEYWORDS if kw in recent_text)
    healthy_count = sum(1 for kw in HEALTHY_KEYWORDS if kw in recent_text)

    # Extract injury type
    injury_type = extract_injury_type(recent_text)

    # Determine status with priority
    if out_count > 0 and out_count >= healthy_count:
        status = "OUT"
        confidence = min(0.7 + (out_count * 0.1), 0.95)
        reason = f"Listed as OUT - {injury_type or 'injury reported'}"
    elif doubtful_count > 0:
        status = "DOUBTFUL"
        confidence = 0.7 + (doubtful_count * 0.1)
        reason = f"Listed as DOUBTFUL - {injury_type or 'injury concern'}"
    elif questionable_count > 0 and healthy_count == 0:
        status = "QUESTIONABLE"
        confidence = 0.6 + (questionable_count * 0.1)
        reason = f"Listed as QUESTIONABLE - {injury_type or 'game-time decision'}"
    elif probable_count > 0:
        status = "PROBABLE"
        confidence = 0.7
        reason = "Expected to play"
    elif healthy_count > out_count:
        status = "ACTIVE"
        confidence = 0.8
        reason = "Cleared to play / returning from injury"
    else:
        status = "ACTIVE"
        confidence = 0.5
        reason = "No clear injury indicators"

    return {
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "injury_type": injury_type,
        "latest_news": news_items[0] if news_items else None
    }


def get_player_injury_status(player_name: str) -> dict:
    """
    Get the current injury status for a player.

    Returns:
        Dictionary with:
        - status: "OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", or "ACTIVE"
        - confidence: 0.0 to 1.0
        - reason: Explanation of the status
        - news: List of relevant news items with source links
        - checked_at: Timestamp
    """
    # Fetch fresh news from all sources
    all_news = get_nba_injury_news(limit=100)

    # Filter for this player
    player_news = search_player_news(player_name, all_news)

    # Analyze status
    analysis = analyze_injury_status(player_news)

    return {
        "player": player_name,
        "status": analysis["status"],
        "confidence": analysis["confidence"],
        "reason": analysis["reason"],
        "injury_type": analysis.get("injury_type"),
        "news": player_news[:5],  # Return top 5 with links
        "checked_at": datetime.now().isoformat()
    }


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

def get_injury_report(player_names: list[str]) -> pd.DataFrame:
    """
    Get injury status for multiple players at once.
    More efficient than checking one by one.
    """
    # Fetch news once
    all_news = get_nba_injury_news(limit=200)

    results = []
    for player in player_names:
        player_news = search_player_news(player, all_news)
        analysis = analyze_injury_status(player_news)

        latest_news = player_news[0] if player_news else None

        results.append({
            "player": player,
            "status": analysis["status"],
            "confidence": analysis["confidence"],
            "reason": analysis["reason"],
            "latest_headline": latest_news["title"] if latest_news else "No recent news",
            "source": latest_news.get("source", "") if latest_news else "",
            "source_url": latest_news.get("link", "") if latest_news else ""
        })

    return pd.DataFrame(results)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_player_available(player_name: str) -> tuple[bool, str]:
    """
    Simple check: Is this player likely to play?
    """
    status = get_player_injury_status(player_name)

    if status["status"] == "OUT":
        return False, f"OUT - {status['reason']}"
    elif status["status"] == "DOUBTFUL":
        return False, f"DOUBTFUL - {status['reason']}"
    elif status["status"] == "QUESTIONABLE":
        return True, f"QUESTIONABLE - {status['reason']}"
    elif status["status"] == "PROBABLE":
        return True, f"PROBABLE - {status['reason']}"
    else:
        return True, "ACTIVE - Available to play"


def extract_injury_type(news_text: str) -> Optional[str]:
    """Extract specific injury type from news text."""
    injury_patterns = [
        (r"ankle\s*(sprain|injury|soreness)", "ankle"),
        (r"knee\s*(injury|sprain|soreness|contusion)", "knee"),
        (r"hamstring\s*(strain|injury|tightness)", "hamstring"),
        (r"back\s*(spasms|soreness|injury|tightness)", "back"),
        (r"shoulder\s*(injury|soreness|sprain)", "shoulder"),
        (r"concussion", "concussion"),
        (r"illness|flu|sick", "illness"),
        (r"calf\s*(strain|injury|soreness)", "calf"),
        (r"quad\s*(strain|injury|contusion)", "quad"),
        (r"groin\s*(strain|injury)", "groin"),
        (r"finger\s*(injury|sprain)", "finger"),
        (r"wrist\s*(injury|sprain)", "wrist"),
        (r"foot\s*(injury|soreness|sprain)", "foot"),
        (r"hip\s*(injury|soreness)", "hip"),
        (r"rest|load management", "rest"),
        (r"personal|personal reasons", "personal"),
    ]

    text_lower = news_text.lower()
    for pattern, injury in injury_patterns:
        if re.search(pattern, text_lower):
            return injury

    return None


def format_news_for_display(news_items: list[dict]) -> list[dict]:
    """
    Format news items for display with proper links.
    Each item will have a clickable source link.
    """
    formatted = []
    for item in news_items:
        formatted.append({
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "link": item.get("link", ""),
            "source": item.get("source", "Unknown"),
            "source_url": item.get("source_url", item.get("link", "")),
            "published": item.get("published", ""),
            "time_ago": _format_time_ago(item.get("published", ""))
        })
    return formatted


def _format_time_ago(date_str: str) -> str:
    """Convert date string to 'X hours ago' format."""
    try:
        # Try various date formats
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(date_str[:19], fmt[:19] if "T" in fmt else fmt)
                break
            except ValueError:
                continue
        else:
            return ""

        now = datetime.now()
        diff = now - dt

        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except Exception:
        return ""
