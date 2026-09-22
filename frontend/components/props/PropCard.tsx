"use client";

import { cn, statColor } from "@/lib/utils";
import type { Prop } from "@/lib/types";

interface Props {
  prop: Prop;
}

export function PropCard({ prop }: Props) {
  const color = statColor(prop.stat);
  const hitPct = prop.hit_rate;

  return (
    <article
      className={cn(
        "glass p-4 flex flex-col gap-3 relative overflow-hidden",
        prop.is_lock && "ring-1 ring-[#f59e0b]/40"
      )}
    >
      {/* Lock badge */}
      {prop.is_lock && (
        <span className="absolute top-2 right-2 text-[10px] font-bold uppercase tracking-widest text-[#f59e0b] bg-[#f59e0b]/10 border border-[#f59e0b]/30 px-2 py-0.5 rounded-full">
          🔒 Lock
        </span>
      )}

      {/* Player & matchup */}
      <div className="pr-14">
        <p className="text-[13px] font-bold text-[#f0f4ff] leading-tight">{prop.player}</p>
        <p className="text-[11px] text-[#8ca0c0] mt-0.5">{prop.team} vs {prop.opponent} · {prop.game_matchup}</p>
      </div>

      {/* Stat line */}
      <div className="flex items-center gap-3">
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded-full border"
          style={{ color, borderColor: `${color}40`, background: `${color}15` }}
        >
          {prop.stat_label || prop.stat}
        </span>
        <span className="text-2xl font-extrabold" style={{ color }}>
          {prop.direction === "Over" ? "O" : "U"} {prop.line ?? "—"}
        </span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <Metric label="Hit Rate" value={`${hitPct}%`} highlight={hitPct >= 70} />
        <Metric label="Avg" value={prop.avg != null ? String(prop.avg) : "—"} />
        <Metric label="EV" value={prop.ev > 0 ? `+${prop.ev}` : String(prop.ev)} highlight={prop.ev > 5} />
      </div>

      {/* Insight */}
      {prop.insight && (
        <p className="text-[11px] text-[#8ca0c0] leading-snug border-t border-white/5 pt-2">
          {prop.insight}
        </p>
      )}

      {/* Blowout risk */}
      {prop.blowout_risk && (
        <p className="text-[10px] font-semibold text-[#f87171]">⚠ Blowout risk</p>
      )}
    </article>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className={cn("text-sm font-bold", highlight ? "text-[#2dd4bf]" : "text-[#f0f4ff]")}>
        {value}
      </span>
      <span className="text-[10px] text-[#8ca0c0] uppercase tracking-wide">{label}</span>
    </div>
  );
}
