"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

interface Props {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  /** Show a search box; worth it for long lists such as players. */
  searchable?: boolean;
  /** Most items that can be picked at once. */
  max?: number;
  /** Shown when there is nothing to pick yet. */
  emptyText?: string;
}

/** A dropdown of checkboxes. Nothing picked means "any", so the button shows only a count. */
export function MultiPick({ label, options, selected, onChange, searchable = false, max, emptyText = "Nothing to pick yet." }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", escape); };
  }, [open]);

  const needle = query.trim().toLowerCase();
  const shown = options.filter(option => option.toLowerCase().includes(needle));
  const full = max !== undefined && selected.length >= max;
  const toggle = (option: string) => onChange(selected.includes(option) ? selected.filter(item => item !== option) : [...selected, option]);

  return (
    <div ref={root} className="relative">
      <button type="button" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(value => !value)}
        className="ql-control flex min-h-8 items-center gap-2 px-3 py-1.5 text-xs font-semibold text-[#eaf8f6] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">
        {label}
        {selected.length > 0 && <span className="rounded-full bg-[#59e0c8] px-1.5 text-[10px] font-bold text-[#0d151b]" aria-label={`${selected.length} selected`}>{selected.length}</span>}
        <ChevronDown size={14} aria-hidden="true" className={open ? "rotate-180 transition-transform" : "transition-transform"} />
      </button>
      {open && (
        <div id={panelId} role="group" aria-label={label} className="ql-panel absolute left-0 top-full z-30 mt-2 w-64 p-2 shadow-xl">
          {searchable && options.length > 0 && (
            <input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder={`Search ${label.toLowerCase()}`} aria-label={`Search ${label}`}
              className="ql-control mb-2 min-h-8 w-full px-2.5 py-1.5 text-xs text-[#eaf8f6] placeholder:text-[#a9bec0]" />
          )}
          {options.length === 0 && <p className="px-2 py-3 text-xs text-[#a9bec0]">{emptyText}</p>}
          <ul className="max-h-64 overflow-auto">
            {shown.map(option => {
              const checked = selected.includes(option);
              return (
                <li key={option}>
                  <label className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-xs text-[#eaf8f6] hover:bg-white/5">
                    <input type="checkbox" checked={checked} disabled={!checked && full} onChange={() => toggle(option)} className="accent-[#59e0c8]" />
                    <span className="truncate">{option}</span>
                  </label>
                </li>
              );
            })}
          </ul>
          <div className="mt-2 flex items-center justify-between border-t border-white/10 px-2 pt-2 text-[11px] text-[#a9bec0]">
            <span>{full ? `Up to ${max} at a time` : selected.length ? `${selected.length} selected` : "Any"}</span>
            <button type="button" onClick={() => onChange([])} disabled={!selected.length} className="font-semibold text-[#59e0c8] disabled:opacity-40">Clear</button>
          </div>
        </div>
      )}
    </div>
  );
}
