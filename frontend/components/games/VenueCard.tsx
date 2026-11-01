export interface VenueBar { win: boolean; tip: string }
export interface VenueCardProps {
  title: string;
  /** Games in the sample; 0 shows the empty note. */
  games: number;
  wins?: number;
  losses?: number;
  scores?: string | number | null;
  allows?: string | number | null;
  margin?: number | null;
  /** Oldest to newest. */
  bars: VenueBar[];
  emptyText?: string;
}

const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}`;

/** Last-N record, scoring and the row of win (teal) / loss (rose) bars. Shared by the NBA matchup tab and the CBB game page. */
export function VenueCard({ title, games, wins, losses, scores, allows, margin, bars, emptyText = "No games at this venue in the data." }: VenueCardProps) {
  if (!games) return <div className="ql-panel p-4"><h4 className="text-sm font-bold text-text-pri">{title}</h4><p className="mt-1 text-xs text-text-sec">{emptyText}</p></div>;
  const m = margin ?? 0;
  return <div className="ql-panel p-4">
    <h4 className="text-sm font-bold text-text-pri">{title}</h4>
    <p className="mt-1 text-2xl font-extrabold tabular-nums text-text-pri">{wins}–{losses}<span className="ml-2 text-xs font-medium text-text-sec">last {games}</span></p>
    <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
      <div><dt className="ql-data-label">SCORES</dt><dd className="font-bold tabular-nums text-text-pri">{scores}</dd></div>
      <div><dt className="ql-data-label">ALLOWS</dt><dd className="font-bold tabular-nums text-text-pri">{allows}</dd></div>
      <div><dt className="ql-data-label">MARGIN</dt><dd className={`font-bold tabular-nums ${m >= 0 ? "text-teal-300" : "text-rose-300"}`}>{signed(m)}</dd></div>
    </dl>
    <div className="mt-3 flex gap-1" role="img" aria-label={`Last ${games} games, oldest to newest: ${bars.map(bar => bar.win ? "win" : "loss").join(", ")}`}>
      {bars.map((bar, index) => <span key={index} title={bar.tip} className={`h-4 flex-1 rounded-sm ${bar.win ? "bg-teal-400" : "bg-rose-400/70"}`} />)}
    </div>
  </div>;
}
