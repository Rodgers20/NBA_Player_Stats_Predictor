"""User-entered journal backed by the existing local personal-bets store."""
from typing import Literal
import os

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from utils import personal_bets

def _journal_available():
    if os.getenv('SERVE_FRONTEND') == '1' and os.getenv('JOURNAL_PRIVATE') != '1':
        raise HTTPException(status_code=503, detail='Journal storage is available in the local app only')


router = APIRouter(dependencies=[Depends(_journal_available)])
Mode = Literal['paper', 'real']
JournalStat = Literal['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK',
                      'PTS+REB', 'PTS+AST', 'REB+AST', 'PTS+REB+AST', 'STL+BLK']


class Entry(BaseModel):
    token: str = Field(min_length=1, max_length=128)
    mode: Mode
    league: Literal['nba', 'wnba']
    player: str = Field(max_length=200)
    game_date: str
    stat: JournalStat
    side: Literal['Over', 'Under']
    line: float = Field(allow_inf_nan=False)
    price: float = Field(allow_inf_nan=False)
    book: str = Field(max_length=200)
    stake: float = Field(allow_inf_nan=False)
    notes: str = Field(default='', max_length=4000)


class Settlement(BaseModel):
    result: Literal['win', 'loss', 'push', 'void', 'pending']


@router.get('/journal')
def journal(mode: Mode = 'paper'):
    return {'mode': mode, 'bets': personal_bets.list_bets(mode),
            'summary': personal_bets.summary(mode)}


@router.post('/journal')
def save_entry(entry: Entry):
    try:
        identifier = personal_bets.add_bet(entry.model_dump(exclude={'token'}), entry.token)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'id': identifier}


@router.patch('/journal/{identifier}')
def settle_entry(identifier: str, settlement: Settlement):
    try:
        personal_bets.settle_bet(identifier, settlement.result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {'id': identifier, 'result': settlement.result}
