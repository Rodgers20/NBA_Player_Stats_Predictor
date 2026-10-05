"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, Check, Users } from "lucide-react";
import { api } from "@/lib/api";

export function GameDayPlayerPicker({ league, date, value, onChange }: {
  league: string; date: string; value: string; onChange: (value: string) => void;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ["scheduled-players", league, date],
    queryFn: () => api.scheduledPlayers(league, date),
    enabled: Boolean(date), retry: false,
  });
  const players = data?.game_date === date ? data.players : [];
  const matches = players.filter(player => player.name.toLocaleLowerCase().includes(value.trim().toLocaleLowerCase())).slice(0, 30);
  const selected = players.find(player => player.name === value);
  const status = !date ? "Choose a game date to see available players."
    : isPending ? "Loading players for this date…"
    : isError ? "The schedule could not be loaded. Suggestions are unavailable."
    : !players.length ? data?.message || "No scheduled players available for this date."
    : `${players.length} players on scheduled teams · ${league.toUpperCase()}`;

  return <div className="relative space-y-2 sm:col-span-2 lg:col-span-3" onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget)) { setOpen(false); setActive(-1); }
  }}>
    <label htmlFor={id} className="text-xs font-semibold">Player</label>
    <div className="relative">
      <Search aria-hidden="true" size={17} className="pointer-events-none absolute left-3 top-3.5 text-text-sec" />
      <input id={id} name="player" role="combobox" aria-autocomplete="list" aria-expanded={open && matches.length > 0}
        aria-controls={`${id}-options`} aria-describedby={`${id}-status`} aria-activedescendant={open && active >= 0 && matches[active] ? `${id}-option-${active}` : undefined}
        required maxLength={200} autoComplete="off" value={value} placeholder="Search a player on this day's schedule"
        className="ql-control w-full py-3 pl-10 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400"
        onFocus={() => setOpen(true)} onChange={event => { onChange(event.target.value); setOpen(true); setActive(-1); }}
        onKeyDown={event => {
          if (event.key === "ArrowDown") { event.preventDefault(); setOpen(true); setActive(index => Math.min(index + 1, matches.length - 1)); }
          if (event.key === "ArrowUp") { event.preventDefault(); setActive(index => Math.max(index - 1, 0)); }
          if (event.key === "Escape") { setOpen(false); setActive(-1); }
          if (event.key === "Enter" && open && active >= 0 && matches[active]) {
            event.preventDefault(); onChange(matches[active].name); setOpen(false); setActive(-1);
          }
        }} />
      {selected && <Check aria-hidden="true" size={18} className="absolute right-3 top-3.5 text-teal-400" />}
    </div>
    {open && matches.length > 0 && <ul id={`${id}-options`} role="listbox" aria-label="Players scheduled for the selected date"
      className="absolute inset-x-0 top-[76px] z-30 max-h-64 overflow-y-auto rounded-lg border border-border-card bg-[#152128] p-1 shadow-2xl">
      {matches.map((player, index) => <li key={`${player.name}-${player.team}`} id={`${id}-option-${index}`} role="option" aria-selected={value === player.name}
        onMouseDown={event => event.preventDefault()} onClick={() => { onChange(player.name); setOpen(false); setActive(-1); }}
        className={`flex cursor-pointer items-center justify-between gap-3 rounded px-3 py-3 text-sm hover:bg-[#263d43] ${active === index ? "bg-[#263d43]" : ""}`}>
        <span className="font-medium">{player.name}</span><span className="text-xs text-text-sec">{player.team}{player.opponent ? ` · vs ${player.opponent}` : ""}</span>
      </li>)}
    </ul>}
    <p id={`${id}-status`} role="status" className="flex items-start gap-2 text-xs text-text-sec"><Users aria-hidden="true" size={14} className="mt-0.5 shrink-0" />
      <span>{status}{open && value && players.length > 0 && matches.length === 0 && " No matching player on this schedule."}</span>
    </p>
    {isError && <button type="button" onClick={() => void refetch()} className="text-xs font-semibold text-teal-400">Retry player suggestions</button>}
  </div>;
}
