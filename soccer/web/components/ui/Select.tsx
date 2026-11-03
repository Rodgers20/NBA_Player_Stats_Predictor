"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/format";

export interface SelectOption {
  value: string;
  /** Plain text: used for typeahead, the tooltip, and as the fallback render. */
  label: string;
  /**
   * Richer rendering of the same label, when the text alone loses something —
   * a fixture needs each club's crest beside its own name, not a pair of
   * badges bunched at the left where they belong to neither.
   */
  labelNode?: React.ReactNode;
  /** Optional second line, e.g. a kick-off time or match count. */
  hint?: string;
  /** Optional leading visual, e.g. a competition crest. */
  icon?: React.ReactNode;
  group?: string;
}

/**
 * Accessible listbox replacing the native <select>.
 *
 * A native select renders an OS-drawn menu that ignores the app's typography,
 * spacing and colours entirely — on macOS it comes out oversized with a system
 * highlight, which is exactly what made these controls look unfinished.
 *
 * Keyboard behaviour matches the WAI-ARIA listbox pattern: Up/Down move the
 * active option, Home/End jump, Enter/Space commit, Escape cancels, and typing
 * jumps to the first match. The trigger keeps a 36px height so it stays a
 * comfortable pointer target without dominating a toolbar.
 */
/**
 * How long after a selection a click on the trigger is treated as the echo of
 * that same press rather than a new request to open. Short enough that a
 * deliberate reopen is never blocked; the observed echo arrives within ~70ms.
 */
const REOPEN_GUARD_MS = 300;

export function Select({
  value, options, onChange, placeholder = "Select…", className,
  align = "start", widthClass = "w-full", disabled = false, ariaLabel,
}: {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  align?: "start" | "end";
  widthClass?: string;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const queryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const committedAt = useRef(0);
  const listId = useId();

  const selected = useMemo(
    () => options.find((o) => o.value === value),
    [options, value],
  );

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
  }, []);

  // Close on outside click or scroll of an ancestor.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) close();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, close]);

  // Keep the active option in view.
  useEffect(() => {
    if (!open) return;
    const el = listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  function toggle() {
    // Choosing an option closes the list, and the browser then delivers a
    // second click for that same press to whatever the pointer now sits over —
    // which was landing on this trigger and reopening the menu, so a selection
    // appeared to do nothing. A press this soon after a commit is that echo,
    // not a person asking for the list again.
    if (!open && Date.now() - committedAt.current < REOPEN_GUARD_MS) return;
    if (!open) setActive(Math.max(options.findIndex(o => o.value === value), 0));
    setOpen(o => !o);
  }

  function commit(index: number) {
    const opt = options[index];
    committedAt.current = Date.now();
    // Close before handing over: `onChange` may start a navigation, and the
    // menu should be gone at that moment rather than hanging over the new page.
    close();
    if (opt) onChange(opt.value);
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (disabled) return;
    if (!open && ["Enter", " ", "ArrowDown", "ArrowUp"].includes(e.key)) {
      e.preventDefault();
      setActive(Math.max(options.findIndex(o => o.value === value), 0));
      setOpen(true);
      return;
    }
    if (!open) return;

    switch (e.key) {
      case "Tab": close(); break;
      case "Escape":
        e.preventDefault(); close(); break;
      case "ArrowDown":
        e.preventDefault(); setActive((i) => Math.min(i + 1, options.length - 1)); break;
      case "ArrowUp":
        e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); break;
      case "Home":
        e.preventDefault(); setActive(0); break;
      case "End":
        e.preventDefault(); setActive(options.length - 1); break;
      case "Enter":
      case " ":
        e.preventDefault(); commit(active); break;
      default: {
        if (e.key.length !== 1) return;
        const next = query + e.key.toLowerCase();
        setQuery(next);
        if (queryTimer.current) clearTimeout(queryTimer.current);
        queryTimer.current = setTimeout(() => setQuery(""), 600);
        const i = options.findIndex((o) => o.label.toLowerCase().startsWith(next));
        if (i >= 0) setActive(i);
      }
    }
  }

  return (
    <div ref={rootRef} className={cn("relative", widthClass, className)}>
      <button
        type="button"
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-haspopup="listbox"
        aria-activedescendant={open ? `${listId}-${active}` : undefined}
        aria-label={ariaLabel}
        disabled={disabled}
        onClick={() => !disabled && toggle()}
        onKeyDown={onKeyDown}
        className={cn(
          "flex min-h-11 w-full cursor-pointer items-center gap-2 rounded-lg border px-2.5",
          "text-[12.5px] transition-colors duration-150",
          "border-stroke bg-elevated text-ink",
          "hover:border-stroke-hi hover:bg-[var(--color-elevated)]",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-pitch/50",
          open && "border-pitch/50",
          disabled && "cursor-not-allowed opacity-50",
        )}
      >
        {selected?.icon}
        <span className={cn("min-w-0 flex-1 truncate text-left",
          !selected && "text-ink-muted")} title={selected?.label}>
          {selected?.labelNode ?? selected?.label ?? placeholder}
        </span>
        <ChevronDown
          className={cn("w-3.5 h-3.5 shrink-0 text-ink-muted transition-transform duration-150",
            open && "rotate-180")}
          aria-hidden
        />
      </button>

      {open && (
        <ul
          id={listId}
          ref={listRef}
          role="listbox"
          tabIndex={-1}
          aria-activedescendant={`${listId}-${active}`}
          onKeyDown={onKeyDown}
          className={cn(
            "absolute z-50 mt-1.5 max-h-[min(380px,60vh)] overflow-y-auto",
            "min-w-full rounded-xl border border-stroke-hi bg-elevated p-1",
            "shadow-2xl shadow-black/50 backdrop-blur",
            align === "end" ? "right-0" : "left-0",
          )}
        >
          {options.length === 0 && (
            <li className="px-2.5 py-2 text-[12px] text-ink-muted">No options</li>
          )}
          {options.map((o, i) => {
            const isSelected = o.value === value;
            const isActive = i === active;
            return (
              <li
                key={o.value}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={isSelected}
                data-index={i}
                onMouseEnter={() => setActive(i)}
                onClick={(e) => { e.stopPropagation(); commit(i); }}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5",
                  "text-[12.5px] transition-colors duration-100",
                  isActive ? "bg-pitch/12 text-ink" : "text-ink-secondary",
                )}
              >
                {o.icon}
                <span className="min-w-0 flex-1">
                  <span className="block truncate" title={o.label}>
                    {o.labelNode ?? o.label}
                  </span>
                  {o.hint && (
                    <span className="block truncate text-[10.5px] text-ink-muted">
                      {o.hint}
                    </span>
                  )}
                </span>
                <Check
                  className={cn("w-3.5 h-3.5 shrink-0 text-pitch transition-opacity",
                    isSelected ? "opacity-100" : "opacity-0")}
                  aria-hidden
                />
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
