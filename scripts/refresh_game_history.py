#!/usr/bin/env python3
"""Explicit free game-history refresh, isolated from prop training data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from utils.game_data import fetch_official_team_games, load_cached_nba, write_history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league',choices=['nba','wnba'],required=True)
    parser.add_argument('--cached-nba',action='store_true',help='Use existing Kaggle cache without network/download')
    parser.add_argument('--seasons',nargs='+',help='Explicit seasons e.g.2025-26 or2026; max5')
    parser.add_argument('--start-date',default='')
    parser.add_argument('--end-date',default='')
    parser.add_argument('--timeout',type=int,default=20)
    parser.add_argument('--playoffs',action='store_true')
    parser.add_argument('--supplement-cached-nba',action='store_true',help='Fill official missing games and arenas from mounted cache; preserve source provenance')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--replace',action='store_true',help='Replace full history rather than merge this refresh into existing seasons')
    args=parser.parse_args()
    if args.cached_nba:
        if args.league!='nba':parser.error('--cached-nba requires NBA')
        frame=load_cached_nba(args.start_date or '2020-10-01')
    else:
        if not args.seasons or len(args.seasons)>5:parser.error('Specify1–5seasons for bounded explicit refresh')
        if not 1<=args.timeout<=60:parser.error('Timeout must be1–60seconds')
        frames=[]
        for season in args.seasons:
            for kind in (['Regular Season','Playoffs'] if args.playoffs else ['Regular Season']):
                frames.append(fetch_official_team_games(args.league,season,kind,args.timeout,args.start_date,args.end_date))
        frame=pd.concat(frames,ignore_index=True)
        if args.supplement_cached_nba:
            if args.league!='nba':parser.error('--supplement-cached-nba requires NBA')
            cached=load_cached_nba(args.start_date or '2020-10-01')
            cached=cached[cached.SEASON.isin(args.seasons)]
            arenas=cached[['GAME_ID','TEAM_ABBREVIATION','arenaName','arenaCity','arenaState']]
            frame=frame.merge(arenas,on=['GAME_ID','TEAM_ABBREVIATION'],how='left',validate='one_to_one')
            frame=pd.concat([cached,frame],ignore_index=True)
    output=args.output or Path(__file__).resolve().parents[1]/'data'/args.league/'game_team_history.csv'
    if output.exists() and not args.replace:
        if frame.empty:raise ValueError('Empty refresh; preserving existing history')
        frame=pd.concat([pd.read_csv(output,dtype={'GAME_ID':str}),frame],ignore_index=True)
    print(json.dumps(write_history(frame,output),indent=2))


if __name__=='__main__':main()
