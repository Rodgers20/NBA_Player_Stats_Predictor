"""Personal bet journal and deliberate odds refresh."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from dash import html, dcc, Input, Output, State, callback, ctx
from utils import personal_bets, odds_budget

FIELDS = ('mode','league','player','game_date','stat','side','line','price','book','stake','notes')


def field(label, component):
    return html.Div([html.Label(label, htmlFor=component.id), component], className='journal-field')


def create_page():
    return html.Main([
        html.Div([html.Div('YOUR RECORD', className='journal-eyebrow'), html.H1('My Bets'),
                  html.P('Your selections. Your actual stakes. Paper and real results kept separate.')]),
        dcc.Store(id='journal-token', data=str(uuid4())),
        dcc.Store(id='journal-version', data=0),
        html.Section([
            html.H2('Free odds desk'),
            html.P('Refresh up to two upcoming games: points, rebounds, assists. Maximum 6 credits per refresh; 12 per day shared across the app. No automatic prop polling.'),
            field('League', dcc.Dropdown(id='odds-league', options=[{'label':'NBA','value':'nba'}, {'label':'WNBA','value':'wnba'}], value='wnba', clearable=False)),
            html.Button('Refresh real prop lines', id='journal-refresh-odds', n_clicks=0),
            html.Div(id='journal-odds-status', role='status'),
            html.Div(id='journal-quotes'),
            html.Button('Refresh ESPN game lines (no odds credits)', id='journal-refresh-games', n_clicks=0),
            html.Div(id='journal-game-status', role='status'),
            html.P('No free quota left? Enter a current line and price from your sportsbook below. A manual quote is identified as user-entered; it does not pretend to be an API quote.'),
        ], className='journal-card'),
        html.Section([
            html.H2('Record a selection'),
            html.P('Evaluate a line without saving a bet, or log a wager you chose. This app never places bets.'),
            html.Div([
                field('Record type', dcc.Dropdown(id='bet-mode', options=[{'label':'Paper test','value':'paper'}, {'label':'Real wager','value':'real'}], value='paper', clearable=False)),
                field('League', dcc.Dropdown(id='bet-league', options=['nba','wnba'], value='wnba', clearable=False)),
                field('Player name', dcc.Input(id='bet-player', type='text', placeholder="A’ja Wilson")),
                field('Game date', dcc.Input(id='bet-game_date', type='text', value=date.today().isoformat(), placeholder='YYYY-MM-DD')),
                field('Stat', dcc.Dropdown(id='bet-stat', options=['PTS','REB','AST'], value='PTS', clearable=False)),
                field('Side', dcc.Dropdown(id='bet-side', options=['Over','Under'], value='Over', clearable=False)),
                field('Book line', dcc.Input(id='bet-line', type='number', min=0, step=.5, placeholder='24.5')),
                field('Accepted American odds', dcc.Input(id='bet-price', type='number', placeholder='-110')),
                field('Sportsbook', dcc.Input(id='bet-book', type='text', placeholder='Your sportsbook')),
                field('Stake ($)', dcc.Input(id='bet-stake', type='number', min=.01, step=.01, placeholder='Actual stake')),
                field('Notes / opponent / ticket reference', dcc.Input(id='bet-notes', type='text')),
            ], className='journal-grid'),
            html.Div([html.Button('Evaluate manual line', id='journal-evaluate', n_clicks=0),
                      html.Button('Save my bet', id='journal-save', n_clicks=0),
                      html.Button('Start another entry', id='journal-new', n_clicks=0)], className='journal-actions'),
            html.Div(id='journal-evaluation', role='status'),
            html.Div(id='journal-save-status', role='status'),
        ], className='journal-card'),
        html.Section([
            html.H2('Your progress'),
            dcc.RadioItems(id='journal-mode', options=[{'label':'Paper tests','value':'paper'}, {'label':'Real wagers','value':'real'}], value='paper', inline=True),
            html.Div(id='journal-summary', className='journal-summary'),
            html.Div(id='journal-table', className='journal-table-scroll'),
            html.H3('Settle or correct a result'),
            html.Div([
                field('Recorded bet', dcc.Dropdown(id='journal-bet-id', options=[], placeholder='Select a bet')),
                field('Book settlement', dcc.Dropdown(id='journal-result', options=['win','loss','push','void','pending'], value='win', clearable=False)),
            ], className='journal-grid'),
            html.Button('Save settlement', id='journal-settle', n_clicks=0),
            html.Div(id='journal-settle-status', role='status'),
            html.P('Settle against your sportsbook receipt. Void and pending stakes are excluded from settled ROI; pushes return the stake. Corrections replace the previous result.'),
        ], className='journal-card'),
    ], className='journal')


def register_callbacks(evaluate_manual, refresh_odds):
    @callback(Output('journal-save-status','children'), Output('journal-token','data'),
              Input('journal-save','n_clicks'), Input('journal-new','n_clicks'),
              *[State('bet-'+key,'value') for key in FIELDS], State('journal-token','data'), prevent_initial_call=True)
    def save(_save, _new, *values):
        if ctx.triggered_id == 'journal-new':
            return 'Ready for another entry. Update the fields, then save.', str(uuid4())
        payload = dict(zip(FIELDS, values[:-1]))
        token = values[-1]
        try:
            identifier = personal_bets.add_bet(payload, token)
            return f'Saved {payload["mode"]} entry {identifier[:8]}. Use Start another entry before adding a different bet.', token
        except (ValueError, TypeError) as exc:
            return str(exc), token

    @callback(Output('journal-settle-status','children'), Input('journal-settle','n_clicks'),
              State('journal-bet-id','value'), State('journal-result','value'), prevent_initial_call=True)
    def settle(_clicks, identifier, result):
        try:
            personal_bets.settle_bet(identifier, result)
            return f'Result saved: {result}.'
        except ValueError as exc:
            return str(exc)

    @callback(Output('journal-summary','children'), Output('journal-table','children'), Output('journal-bet-id','options'),
              Input('journal-mode','value'), Input('journal-save-status','children'), Input('journal-settle-status','children'))
    def render(mode, _saved, _settled):
        report = personal_bets.summary(mode)
        roi = '—' if report['roi'] is None else f"{report['roi']:+.1f}%"
        metrics = [html.Div([html.Span(label), html.Strong(value)]) for label,value in (
            ('Net profit', f"${report['profit']:+.2f}"), ('Settled ROI',roi),
            ('Pending stake',f"${report['pending']:.2f}"), ('Settled bets',str(report['settled'])))]
        bets = personal_bets.list_bets(mode)
        rows = [html.Tr([html.Td(b['game_date']), html.Td(b['player']),
                         html.Td(f"{b['side']} {b['line']:g} {b['stat']}"), html.Td(b['book']),
                         html.Td(f"{b['price']:+g}"), html.Td(f"${b['stake_cents']/100:.2f}"),
                         html.Td(b['result']), html.Td('—' if b['profit_cents'] is None else f"${b['profit_cents']/100:+.2f}")]) for b in bets]
        table = html.Table([html.Thead(html.Tr([html.Th(h) for h in ('Game date','Player','Selection','Book','Odds','Stake','Result','Net')])), html.Tbody(rows)]) if bets else html.P('No bets recorded. Only entries you save appear here.')
        options = [{'label':f"{b['game_date']} · {b['player']} {b['side']} {b['line']:g} {b['stat']} · ${b['stake_cents']/100:.2f} · {b['id'][:8]}", 'value':b['id']} for b in bets]
        return metrics, table, options

    @callback(Output('journal-evaluation','children'), Input('journal-evaluate','n_clicks'),
              *[State('bet-'+key,'value') for key in FIELDS], prevent_initial_call=True)
    def evaluate(_clicks, *values):
        try:
            return evaluate_manual(dict(zip(FIELDS, values)))
        except (ValueError, TypeError, KeyError) as exc:
            return f'Cannot evaluate: {exc}'

    @callback(Output('journal-odds-status','children'), Output('journal-quotes','children'),
              Input('journal-refresh-odds','n_clicks'), State('odds-league','value'))
    def odds(clicks, league):
        quotes = refresh_odds(league) if clicks else {}
        budget = odds_budget.status()
        text = f"{budget['message']} Local budget: {budget['daily']}/{budget['daily_limit']} today, {budget['monthly']}/{budget['monthly_limit']} this month. Provider credits: {budget['remaining'] if budget['remaining'] is not None else 'not checked'}."
        rows=[]
        for player, markets in quotes.items():
            for stat,q in markets.items():
                rows.append(html.Tr([html.Td(player),html.Td(stat),html.Td(q['line']),html.Td(q.get('over_price')),html.Td(q.get('under_price')),html.Td(q['bookmaker']),html.Td(q.get('updated_at') or 'Unknown')]))
        table = html.Table([html.Thead(html.Tr([html.Th(h) for h in ('Player','Stat','Line','Over','Under','Book','Book updated (UTC)')])),html.Tbody(rows)]) if rows else html.P('No fresh quotes loaded. Player Analysis projections remain available.')
        return text, html.Div(table, className='journal-table-scroll')

    @callback(Output('journal-game-status','children'), Input('journal-refresh-games','n_clicks'),
              State('odds-league','value'), prevent_initial_call=True)
    def refresh_games(_clicks, league):
        from utils.espn_game_odds import get_game_odds
        target_date = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
        games = get_game_odds(league, target_date, None, force_refresh=True)
        if games:
            return f"Loaded ESPN game lines for {len(games)} {league.upper()} matchups. Open Today’s Games to view them. No Odds API credits used."
        return f"No ESPN game lines are available for {league.upper()} on {target_date}. No Odds API credits used."
