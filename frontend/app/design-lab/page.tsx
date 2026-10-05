"use client";

import { useState } from "react";
import Image from "next/image";
import { ArrowUpRight, ChartNoAxesCombined, ChevronRight, CircleCheck, ClipboardList, Layers3, SlidersHorizontal } from "lucide-react";
import s from "./style.module.css";

type Direction = "quant" | "arena" | "editorial";
type Screen = "analysis" | "games" | "props" | "bets" | "hitrates";

const directions: { id: Direction; name: string; subtitle: string; fit: string }[] = [
  { id: "quant", name: "01 / Quant Lab", subtitle: "Precise, fast, data-dense", fit: "Best for daily research" },
  { id: "arena", name: "02 / Arena Pulse", subtitle: "Bold, energetic, broadcast-led", fit: "Best for Games + Props" },
  { id: "editorial", name: "03 / Courtside", subtitle: "Warm, premium, editorial", fit: "Best for player stories" },
];

const screens: { id: Screen; label: string }[] = [
  { id: "analysis", label: "Player Analysis" },
  { id: "games", label: "Today's Games" },
  { id: "props", label: "Best Props" },
  { id: "bets", label: "My Bets" },
  { id: "hitrates", label: "WNBA Hit Rates" },
];

const sampleBars = [20, 31, 27, 34, 26, 29, 24, 33, 28, 30];
const sampleProps = [
  { player: "A'ja Wilson", team: "LVA", market: "Points", selection: "Over 24.5", trend: "8 / 10", kind: "Quoted market", accent: true },
  { player: "Breanna Stewart", team: "NYL", market: "Rebounds", selection: "Over 7.5", trend: "7 / 10", kind: "Quoted market", accent: false },
  { player: "Kelsey Plum", team: "LVA", market: "Assists", selection: "5.5 reference", trend: "6 / 10", kind: "Research only", accent: false },
];

export default function DesignLab() {
  const [direction, setDirection] = useState<Direction>("quant");
  const [screen, setScreen] = useState<Screen>("analysis");
  const chosen = directions.find(item => item.id === direction)!;

  return <main className={s.gallery}>
    <header className={s.galleryHeader}>
      <div><p className={s.kicker}>NBA / WNBA · DESIGN STUDY</p><h1>Choose the next look.</h1><p>Three visual systems. Five pages each. Switch pages inside any concept, then pick the parts you want to combine.</p></div>
      <span className={s.galleryCount}>15 PAGE PREVIEWS</span>
    </header>

    <div className={s.selector} role="group" aria-label="Design direction">
      {directions.map(option => <button key={option.id} type="button" aria-pressed={direction === option.id} onClick={() => setDirection(option.id)} className={`${s.directionButton} ${direction === option.id ? s.directionSelected : ""}`}>
        <span className={s.directionName}>{option.name}</span><span className={s.directionSubtitle}>{option.subtitle}</span><span className={s.directionFit}>{option.fit}</span>
      </button>)}
    </div>

    <div className={s.screenSelector} role="group" aria-label="Preview page">
      {screens.map(option => <button key={option.id} type="button" aria-pressed={screen === option.id} onClick={() => setScreen(option.id)} className={screen === option.id ? s.screenSelected : ""}>{option.label}</button>)}
    </div>

    <div className={`${s.preview} ${s[direction]}`}>
      <div className={s.previewTop}><span>COURT VISION / {chosen.name.toUpperCase()}</span><span>DESIGN CONCEPT · ILLUSTRATIVE VALUES ONLY</span></div>
      <div className={s.appShell}>
        <nav className={s.appNav} aria-label="Concept page navigation">
          <div className={s.brand}><span className={s.brandMark}>CV</span><span>COURT<span className={s.brandAccent}>VISION</span></span></div>
          <div className={s.appNavLinks}>{screens.map(option => <button key={option.id} type="button" aria-current={screen === option.id ? "page" : undefined} onClick={() => setScreen(option.id)}>{option.label}</button>)}</div>
          <span className={s.leaguePill}>WNBA <span>⌄</span></span>
        </nav>
        <div className={s.appContent}>
          {screen === "analysis" && <Analysis direction={direction} />}
          {screen === "games" && <Games direction={direction} />}
          {screen === "props" && <Props direction={direction} />}
          {screen === "bets" && <Bets direction={direction} />}
          {screen === "hitrates" && <HitRates direction={direction} />}
        </div>
        <div className={s.mobileNav}>{screens.map(option => <button key={option.id} type="button" aria-current={screen === option.id ? "page" : undefined} onClick={() => setScreen(option.id)}>{option.label}</button>)}</div>
      </div>
    </div>

    <section className={s.rationale} aria-label="Design direction notes">
      <div><p className={s.kicker}>WHY IT WORKS</p><h2>{chosen.name}</h2></div>
      {direction === "quant" && <p>Aligned numbers and a persistent context rail make repeated comparison quick. The denser layout suits serious users scanning many props, while cyan is reserved for model decisions and amber for market status.</p>}
      {direction === "arena" && <p>A broadcast-style score strip and dramatic player framing make the slate feel alive. The hierarchy puts the winner, spread, and total calls in separate zones, then reveals evidence underneath.</p>}
      {direction === "editorial" && <p>A warm paper surface, generous type, and compact charts bring clarity to research-heavy screens. It feels more trustworthy and readable over long sessions, with less of a sportsbook look.</p>}
    </section>
  </main>;
}

function PageTitle({ eyebrow, title, note, extra }: { eyebrow: string; title: string; note: string; extra?: string }) {
  return <header className={s.pageTitle}><div><p className={s.eyebrow}>{eyebrow}</p><h2>{title}</h2><p className={s.pageNote}>{note}</p></div>{extra && <span className={s.titleExtra}>{extra}</span>}</header>;
}

function Athlete({ large = false }: { large?: boolean }) {
  const [photoFailed, setPhotoFailed] = useState(false);
  return <div className={`${s.athlete} ${large ? s.athleteLarge : ""}`}>
    <div className={s.portrait}>{photoFailed ? <span aria-label="A'ja Wilson">AW</span> : <Image src="https://cdn.wnba.com/headshots/wnba/latest/1040x760/1628932.png" alt="A'ja Wilson" width={large ? 150 : 82} height={large ? 150 : 82} unoptimized onError={() => setPhotoFailed(true)} />}</div>
    <div><span className={s.eyebrow}>LAS VEGAS ACES · FORWARD</span><h3>A&apos;ja Wilson</h3><p>Season 2026 <span className={s.dot}>•</span> Player profile</p></div>
  </div>;
}

function SampleChart() {
  return <div className={s.chart} role="img" aria-label="Illustrative last-ten-game points bars; eight of ten cross the sample 24.5 threshold">
    <div className={s.chartLine}><span>24.5</span></div>
    {sampleBars.map((value, index) => <div key={index} className={s.barColumn}><div className={`${s.bar} ${value < 25 ? s.barBelow : ""}`} style={{ height: `${value / 36 * 100}%` }}><span>{value}</span></div><small>{index + 1}</small></div>)}
  </div>;
}

function Analysis({ direction }: { direction: Direction }) {
  return <>
    <PageTitle eyebrow="01 / PLAYER INTELLIGENCE" title="Player Analysis" note="Form, model outlook, and matchup context in one reading path." extra="Search players / ⌘K" />
    <div className={`${s.analysisHero} ${direction === "arena" ? s.analysisHeroArena : ""}`}>
      <Athlete large={direction === "arena"} />
      <div className={s.heroMetrics}><div><span>PTS / GAME</span><strong>26.8</strong></div><div><span>REB / GAME</span><strong>11.9</strong></div><div><span>FG%</span><strong>51.4</strong></div></div>
    </div>
    <div className={s.statTabs}><span className={s.activeChip}>PTS</span><span>REB</span><span>AST</span><span>3PM</span><span>P+R</span><span>PRA</span><span className={s.tabEnd}>L5&nbsp; / &nbsp;<b>L10</b>&nbsp; / &nbsp;L20</span></div>
    <div className={s.analysisGrid}>
      <section className={`${s.surface} ${s.performance}`}><div className={s.surfaceHead}><div><span className={s.eyebrow}>PERFORMANCE ANALYSIS</span><h3>8 of 10 games <em>at 24.5+</em></h3></div><strong>80%</strong></div><SampleChart /><div className={s.surfaceFoot}><span>Threshold 24.5</span><span>Home / Away&nbsp; · &nbsp;H2H&nbsp; · &nbsp;Season</span></div></section>
      <aside className={`${s.surface} ${s.contextRail}`}><span className={s.eyebrow}>NEXT MATCHUP</span><h3>LVA <span>vs</span> NYL</h3><p>Matchup and injury context stays alongside the chart, with source freshness visible.</p><div className={s.contextSplit}><span>Opponent defense</span><strong>11th</strong></div><div className={s.contextSplit}><span>Health update</span><strong>Available</strong></div></aside>
    </div>
    <div className={`${s.surface} ${s.projection}`}><div><span className={s.eyebrow}>MODEL PROJECTION · SAMPLE</span><h3>Projected 27.2 points</h3><p>Model estimate from recent form. Opponent and venue are shown when available.</p></div><div className={s.projectionNumber}>27.2 <small>PTS</small></div></div>
    <div className={s.supporting}><h3>Supporting stats</h3><div>{[["PTS", "26.8"], ["REB", "11.9"], ["AST", "3.2"], ["STL", "1.8"]].map(([label, value]) => <span key={label}><small>{label}</small><strong>{value}</strong></span>)}</div></div>
  </>;
}

function Games({ direction }: { direction: Direction }) {
  return <>
    <PageTitle eyebrow="02 / SLATE ROOM" title="Today's Games" note="A selected matchup opens its full model and market breakdown." extra="2 MATCHUPS · SAMPLE" />
    <div className={s.fixtureStrip}><div className={s.fixtureActive}><small>07:00 PM · SAMPLE</small><strong>LVA <span>vs</span> NYL</strong><small>Projected 84–79</small></div><div><small>09:00 PM · SAMPLE</small><strong>PHX <span>vs</span> SEA</strong><small>Open matchup →</small></div></div>
    <div className={s.scoreboard}><div><span>LAS VEGAS</span><strong>LVA</strong><small>Sample projection · 84</small></div><span className={s.scoreboardAt}>VS</span><div><span>NEW YORK</span><strong>NYL</strong><small>Sample projection · 79</small></div></div>
    <div className={s.pickGrid}>
      <Pick title="Projected winner" value="LVA" detail="Model margin +5.0" icon={<ChartNoAxesCombined size={18} />} />
      <Pick title="Spread cover" value="LVA −3.5" detail="Only with a verified market line" icon={<SlidersHorizontal size={18} />} />
      <Pick title="Over / Under" value="Under 166.5" detail="Model total 163.0 · sample" icon={<Layers3 size={18} />} />
    </div>
    <div className={`${s.surface} ${s.gameEvidence}`}><span><CircleCheck size={16} /> Why the model leans LVA</span><p>{direction === "arena" ? "A broadcast-style take leads, with detailed evidence one level below." : "Recent form, injury context, pace, and market freshness appear here before a user tracks any pick."}</p><small>MARKET PICKS REQUIRE REAL QUOTES · SAMPLE VALUES ABOVE</small></div>
  </>;
}

function Pick({ title, value, detail, icon }: { title: string; value: string; detail: string; icon: React.ReactNode }) {
  return <section className={s.pickCard}><div className={s.pickLabel}>{icon}<span>{title}</span></div><strong>{value}</strong><small>{detail}</small></section>;
}

function Props({ direction }: { direction: Direction }) {
  return <>
    <PageTitle eyebrow="03 / PLAYER MARKETS" title="Best Props" note="A research board that separates quoted picks from historical screens." extra="STAT · SIDE · GAME · SORT" />
    <div className={s.filterBar}><span className={s.activeChip}>All stats</span><span>Over / Under</span><span>All matchups</span><span>Best edge ↓</span><span className={s.tabEnd}>Verified only ◯</span></div>
    <div className={s.propsLayout}>
      <div className={s.propList}><div className={s.listHeading}><span>LAS VEGAS vs NEW YORK</span><span>3 players</span></div>{sampleProps.map((prop, index) => <div key={prop.player} className={`${s.propRow} ${index === 0 ? s.propFeatured : ""}`}><span className={s.propRank}>0{index + 1}</span><span className={s.propIdentity}><strong>{prop.player}</strong><small>{prop.team} · {prop.market}</small></span><span className={s.propChoice}><strong>{prop.selection}</strong><small>{prop.kind}</small></span><span className={s.propTrend}><strong>{prop.trend}</strong><small>last 10</small></span><span className={s.propArrow}><ChevronRight size={18} /></span></div>)}</div>
      <aside className={`${s.surface} ${s.propInspector}`}><span className={s.eyebrow}>{direction === "editorial" ? "THE PLAYER FILE" : "PLAYER INSPECTOR"}</span><Athlete /><div className={s.inspectorScore}><strong>8 / 10</strong><span>historical hits at sample line</span></div><div className={s.tinyBars}>{sampleBars.map((value, index) => <span key={index} style={{ height: `${value / 36 * 100}%` }} />)}</div><p>Open the full player profile or prefill My Bets from a verified quote.</p><span className={s.inlineAction}>View full analysis <ArrowUpRight size={15} /></span></aside>
    </div>
    <p className={s.disclaimer}>Research-only rows show history, never an invented sportsbook price or expected value.</p>
  </>;
}

function Bets({ direction }: { direction: Direction }) {
  return <>
    <PageTitle eyebrow="04 / PERSONAL JOURNAL" title="My Bets" note="A light-touch path from tracked pick to settled record." extra="PAPER / REAL" />
    <div className={s.betMetrics}><div><small>NET RESULT · SAMPLE</small><strong>+$84.50</strong></div><div><small>SETTLED ROI</small><strong>+12.4%</strong></div><div><small>PENDING</small><strong>02</strong></div></div>
    <div className={s.betsGrid}>
      <section className={`${s.surface} ${s.quickSlip}`}><span className={s.eyebrow}>{direction === "arena" ? "YOUR TICKET" : "QUICK ENTRY"}</span><h3>Track a selection in seconds.</h3><p>Player, stat, side and line arrive from Best Props. You only confirm the accepted odds, source and stake.</p><div className={s.slipSelection}><span>A&apos;ja Wilson · Points</span><strong>Over 24.5</strong></div><div className={s.slipFields}><span>Accepted odds <strong>−110</strong></span><span>Stake <strong>$10.00</strong></span></div><span className={s.primaryAction}>Save paper test <ArrowUpRight size={16} /></span></section>
      <section className={`${s.surface} ${s.ledger}`}><div className={s.surfaceHead}><div><span className={s.eyebrow}>SETTLEMENT QUEUE</span><h3>Your tracked picks</h3></div><ClipboardList size={20} /></div><div className={s.ledgerRow}><span>A&apos;ja Wilson<small>Over 24.5 PTS · LVA vs NYL</small></span><strong>Pending</strong></div><div className={s.ledgerRow}><span>Breanna Stewart<small>Over 7.5 REB · NYL vs LVA</small></span><strong>Won</strong></div><p>Receipt-based results and corrections remain one tap away.</p></section>
    </div>
  </>;
}

function HitRates({ direction }: { direction: Direction }) {
  return <>
    <PageTitle eyebrow="05 / WNBA RESEARCH" title="Hit Rates" note="Historical thresholds grouped by matchup, with a direct path to each player profile." extra="LAST 10 GAMES" />
    <div className={s.hitHeader}><span>LAS VEGAS ACES</span><b>vs</b><span>NEW YORK LIBERTY</span></div>
    <div className={s.hitList}>
      {[["A'ja Wilson", "24+ points", 9], ["Breanna Stewart", "7+ rebounds", 8], ["Jackie Young", "4+ assists", 8]].map(([player, stat, hits]) => <div key={player} className={s.hitRow}><span className={s.hitPlayer}><strong>{player}</strong><small>{stat}</small></span><div className={s.hitTrack}><span style={{ width: `${Number(hits) * 10}%` }} /></div><strong>{hits}/10</strong><ChevronRight size={16} /></div>)}
    </div>
    <div className={`${s.surface} ${s.hitNote}`}><strong>{direction === "quant" ? "Threshold matrix" : direction === "arena" ? "Form watch" : "The form book"}</strong><p>These are observed game logs. A 9/10 streak is not a 90% forecast or a priced betting recommendation.</p></div>
  </>;
}
