# ESPN live scores

Premier League (`E0`), La Liga (`SP1`) and Champions League (`UCL`) scores now come from ESPN's public scoreboard feed. No API key or paid subscription is needed. The adapter is `futball/live/espn.py`; `/live/scores?competition_code=SP1` returns today's UTC fixtures, current scores, provider status and freshness metadata. An ongoing match crossing midnight remains visible.

Requests are shared per competition within the API process, with a lock, in-memory caching and a persistent cache under `FUTBALL_DATA_ROOT/raw/espn-live` on the T7. Refreshes are demand-driven: 45 seconds during play or near kickoff, five minutes when idle. No traffic is generated without a requesting page or client. The visible match-centre page polls every 45 seconds; background tabs pause polling.

Live counts and table chips use confirmed provider status, including halftime. They do not infer live play from elapsed wall-clock time. Scheduled, postponed, suspended and completed matches are not counted live. Missing scores remain unavailable rather than becoming zero. Home/away identity comes from provider side labels, not array order.

Failed refreshes retain the last cached scoreboard with its original timestamp and a stale warning. Stale matches are excluded from live counts and live standings chips. Each match links to its ESPN source. The user can see when the data was last checked. This is near-live polling of a public website feed; availability and scoring latency are not contractually guaranteed.

Verified September 15, 2026 against in-play La Liga matches: browser polling refreshed after 46 seconds, scores loaded, mobile layout had no horizontal overflow, and a simulated stale response displayed the warning. Unit tests cover missing and zero scores, home/away order, halftime/full-time/postponement, midnight transitions, shared requests, stale timestamps and exclusion of stale counts.
