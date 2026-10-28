from utils import odds_fetcher as odds
from utils.line_shopping import summarize_books

BOOKS = [
    {'book': 'FanDuel', 'line': 20.5, 'over_price': -115, 'under_price': -105},
    {'book': 'DraftKings', 'line': 20.5, 'over_price': -105, 'under_price': -125},
    {'book': 'BetMGM', 'line': 21.5, 'over_price': +100, 'under_price': -130},
]


def test_best_prices_prefer_line_then_price():
    out = summarize_books(BOOKS)
    # Over: lower line wins, then better price. Under: higher line wins.
    assert out['best_over'] == {'book': 'DraftKings', 'price': -105, 'line': 20.5}
    assert out['best_under'] == {'book': 'BetMGM', 'price': -130, 'line': 21.5}


def test_consensus_is_modal_line_with_median_prices():
    assert summarize_books(BOOKS)['consensus'] == {'line': 20.5, 'over_price': -110, 'under_price': -115}


def test_empty_or_invalid_books_give_none():
    assert summarize_books([]) == {'best_over': None, 'best_under': None, 'consensus': None}
    bad = [{'book': 'X', 'line': 1.5, 'over_price': None, 'under_price': 'n/a'}]
    assert summarize_books(bad)['best_over'] is None


def _event(books):
    return {'id': 'e1', 'commence_time': '2026-09-18T00:00:00Z',
            'home_team': 'Boston Celtics', 'away_team': 'Miami Heat',
            'bookmakers': [{'key': key, 'title': title, 'markets': [
                {'key': 'player_points', 'outcomes': [
                    {'description': 'A', 'name': 'Over', 'point': line, 'price': over},
                    {'description': 'A', 'name': 'Under', 'point': line, 'price': under}]}]}
                for key, title, line, over, under in books]}


def test_parser_keeps_every_book_and_legacy_fields():
    result = {}
    odds._parse_event_odds(_event([('draftkings', 'DraftKings', 21.5, -105, -125),
                                   ('fanduel', 'FanDuel', 20.5, -115, -105)]), result, target_date='2026-09-17')
    quote = result['A']['PTS']
    assert quote['bookmaker'] == 'FanDuel' and quote['line'] == 20.5 and quote['over_price'] == -115
    assert {b['book'] for b in quote['books']} == {'FanDuel', 'DraftKings'}
    assert {'book', 'line', 'over_price', 'under_price'} == set(quote['books'][0])


def test_parser_keeps_sportsbook_links_on_the_books_own_domain():
    event = _event([('fanduel', 'FanDuel', 20.5, -115, -105), ('draftkings', 'DraftKings', 20.5, -110, -110)])
    fanduel, draftkings = event['bookmakers']
    fanduel['link'] = 'https://sportsbook.fanduel.com/basketball/nba/miami-heat-@-boston-celtics-123'
    over, under = fanduel['markets'][0]['outcomes']
    over['link'] = 'https://sportsbook.fanduel.com/addToBetslip?marketId=42.1&selectionId=7'
    under['link'] = 'javascript:alert(1)'                       # never kept
    draftkings['link'] = 'https://evil.example.com/draftkings'  # wrong domain, never kept
    result = {}
    odds._parse_event_odds(event, result, target_date='2026-09-17')
    books = {b['book']: b for b in result['A']['PTS']['books']}
    assert books['FanDuel']['over_link'].endswith('selectionId=7')
    assert books['FanDuel']['event_link'].startswith('https://sportsbook.fanduel.com/')
    assert 'under_link' not in books['FanDuel']
    assert {'book', 'line', 'over_price', 'under_price'} == set(books['DraftKings'])
