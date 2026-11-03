"""Causal, recency-weighted attack/defence measurements; no team-name rules."""
from collections import defaultdict
from datetime import date
from itertools import groupby
import math
import polars as pl

STATS = ('attack', 'defence', 'season_attack', 'season_defence', 'clean_sheet',
         'blank', 'sot_against', 'sot_coverage', 'adjusted_attack', 'adjusted_defence')
COLUMNS = [f'{side}_style_{stat}' for side in ('home', 'away') for stat in STATS]
COLUMNS += ['style_home_matchup', 'style_away_matchup']


def attach(df: pl.DataFrame) -> pl.DataFrame:
    history = defaultdict(list)
    output = []
    for day, batch in groupby(df.sort(['date', 'match_key']).to_dicts(), key=lambda r: r['date']):
        batch = list(batch)
        ordinal = date.fromisoformat(day).toordinal()
        snapshots = {}
        def snapshot(team, season):
            key = (team, season)
            if key in snapshots:
                return snapshots[key]
            past = history[team][-40:]
            weights = [math.exp(-math.log(2) * (ordinal-r['ordinal']) / 90) for r in past]
            def mean(field, prior, strength=3):
                pairs = [(r[field], w) for r,w in zip(past,weights) if r[field] is not None]
                return (sum(v*w for v,w in pairs)+strength*prior)/(sum(w for _,w in pairs)+strength)
            attack, defence = mean('gf',1.4), mean('ga',1.4)
            current = [r for r in past if r['season']==season]
            values = dict(attack=attack, defence=defence,
                season_attack=(sum(r['gf'] for r in current)+6*attack)/(len(current)+6),
                season_defence=(sum(r['ga'] for r in current)+6*defence)/(len(current)+6),
                clean_sheet=mean('clean_sheet',.28), blank=mean('blank',.28),
                sot_against=mean('sot_against',4.5),
                sot_coverage=sum(r['sot_against'] is not None for r in past)/max(1,len(past)),
                adjusted_attack=mean('adjusted_attack',1.),
                adjusted_defence=mean('adjusted_defence',1.))
            snapshots[key] = values
            return values
        pending = []
        for row in batch:
            h,a = row['home_team'],row['away_team']
            hv,av = snapshot(h,row['season']),snapshot(a,row['season'])
            out = {'match_key':row['match_key']}
            for side, values in [('home',hv),('away',av)]:
                out.update({f'{side}_style_{k}':v for k,v in values.items()})
            out['style_home_matchup'] = hv['attack']*av['defence']/1.4
            out['style_away_matchup'] = av['attack']*hv['defence']/1.4
            output.append(out)
            hg,ag = row['home_goals'],row['away_goals']
            if hg is None or ag is None:
                continue
            for team,gf,ga,other,sot in [(h,hg,ag,av,row.get('away_sot')),(a,ag,hg,hv,row.get('home_sot'))]:
                pending.append((team, dict(ordinal=ordinal,season=row['season'],gf=gf,ga=ga,
                    clean_sheet=float(ga==0),blank=float(gf==0),sot_against=sot,
                    adjusted_attack=gf/max(.3,other['defence']),
                    adjusted_defence=ga/max(.3,other['attack']))))
        for team,record in pending:
            history[team].append(record)
    return df.join(pl.DataFrame(output),on='match_key',how='left')
