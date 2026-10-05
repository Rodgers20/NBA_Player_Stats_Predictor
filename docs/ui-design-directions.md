# Court Vision UI directions

Interactive comparison: `/design-lab` in the local Next.js app. The gallery contains three visual systems × five pages. Every score, line, trend, and result in it is **illustrative**. It does not fetch live markets or place bets. The existing product pages are unchanged.

## Design principles

- Keep the working Dash product's information and interactions: player photos and profiles, stat/period controls, threshold chart, model projection directly after performance, opponent and injury context, game detail, prop research, paper/real journal, and WNBA hit rates.
- Make **model estimate**, **verified sportsbook quote**, and **historical research** visibly different. A missing line is an unavailable state, never an invented pick.
- Use large numbers only for the decision at hand. Aligned rows and tabular figures serve multi-prop scanning; clear headings, text labels, and chart data alternatives serve comprehension and accessibility.
- Build on the existing Next.js/FastAPI stack. A selected direction changes presentation and interaction layout, not the data contracts.

## Three options

| Direction | Feel | Strength | Tradeoff |
| --- | --- | --- | --- |
| 01 Quant Lab | Graphite, cool mint, monospaced annotations, precise rows | Fast repeated comparison; scales to many props and journal entries | Less emotional basketball atmosphere |
| 02 Arena Pulse | Deep charcoal, signal orange, broadcast score strips, large type | Strong matchday energy; winner/spread/total are easy to scan | Must restrain motion and density on small screens |
| 03 Courtside | Warm paper, dark ink, clay accent, editorial type and hairlines | Long-session reading and distinctive identity | Needs a compact mode for very large prop boards |

## Page-by-page treatments

| Page | Quant Lab | Arena Pulse | Courtside |
| --- | --- | --- | --- |
| Player Analysis | Stable athlete rail, chart and threshold first, prominent model panel, aligned supporting metrics | Larger athlete hero and stat ribbon; form chart, projection, then context | Player profile as a feature story; chart and projection in reading order with lighter supporting data |
| Today's Games | Selectable slate rows and three decision panels for winner, ATS, total | Broadcast matchup strip and projected scoreboard; bold call tiles with evidence below | Matchup chapter with concise score forecast and separate market notes |
| Best Props | Dense game-grouped rows, filters, sticky history inspector | Ranked player cards with clear quote/research flags | Calm research list with a player file alongside it |
| My Bets | Prefilled quick entry, settlement queue, aligned paper/real ledger | Ticket-style entry and large outcome chips | Receipt-like selections and a readable personal record |
| WNBA Hit Rates | Matchup threshold matrix and compact form bars | Team-vs-team headers with prominent form indicators | Matchup chapters and clear historical caveats |

My suggested mix, if the user wants one: **Courtside** for Player Analysis, **Arena Pulse** for Today's Games, **Quant Lab** for Best Props and My Bets, and **Courtside** for WNBA Hit Rates. The gallery intentionally allows a different choice for every page.

## Inspiration and adaptation

- [Tino's basketball betting dashboard on Dribbble](https://dribbble.com/shots/25831637-Sport-betting-Dashboard) uses athlete metrics, an upcoming-game timeline, and immediate chart feedback. The gallery adapts the hierarchy and timing cues, not its artwork.
- [Tino's sports betting web concept](https://dribbble.com/shots/25820687-Sports-Betting-Web-Design) combines game tracking, player stats, and quick interaction. We retain the scan-first idea while keeping journal actions separate from any real wager placement.
- [21st.dev's dashboard guidance](https://docs.21st.dev/blog/react-dashboard-components) argues for headline → trend → breakdown hierarchy and for aligned tables when scanning many records. That informs Quant Lab's Props and My Bets layouts.
- [21st.dev's dashboard collection](https://21st.dev/community/components/explore/ui-dashboard) and [data-table collection](https://21st.dev/community/components/explore/datatable-react) were reviewed for component patterns. The gallery is custom, so it introduces no new UI library or design dependency.

## Selection prompts

Choose a direction for each page (for example, “Analysis 03, Games 02, Props 01, Bets 01, Hit Rates 03”), or choose one direction across all five. Also call out individual elements to mix, such as the Arena scoreboard in a Courtside Games page. The gallery is a concept review, not a replacement of the live app.
