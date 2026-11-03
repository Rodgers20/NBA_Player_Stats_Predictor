"use client";

import { useCallback, useSyncExternalStore } from "react";
import { cn } from "@/lib/format";

/**
 * Follow a competition, club or player.
 *
 * Local to the browser, deliberately. There is no account system here, and
 * adding one to hold a favourites list would be a far larger change than the
 * feature is worth. The trade-off is that follows do not travel between
 * devices, which is stated on the Settings page rather than left to be
 * discovered.
 *
 * Every storage access is guarded: private windows, cleared site data and
 * browsers configured to block storage all throw rather than returning empty,
 * and a thrown read must not take the page down with it.
 */
export type FollowKind = "competition" | "team" | "player";

const KEY = "futball.follows.v1";

type Follows = Record<string, string[]>;

function read(): Follows {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Follows) : {};
  } catch {
    return {};
  }
}

function write(next: Follows): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
    // Same-tab listeners; the storage event only fires in *other* tabs.
    window.dispatchEvent(new CustomEvent("futball:follows"));
  } catch {
    // Storage unavailable: the button still reflects this session's state.
  }
}

/**
 * Subscribe to the stored follows.
 *
 * `useSyncExternalStore` rather than an effect: localStorage *is* an external
 * store, and reading it in an effect body means a setState on every mount,
 * which cascades a second render for every button on the page. It also gives a
 * defined server snapshot, so the markup matches on hydration.
 */
const EMPTY: string[] = [];
let snapshot: Follows = {};
let snapshotRaw: string | null = null;

function subscribe(onChange: () => void): () => void {
  window.addEventListener("futball:follows", onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener("futball:follows", onChange);
    window.removeEventListener("storage", onChange);
  };
}

/** Cached so repeated reads return the same array identity, as the hook requires. */
function currentFollows(): Follows {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(KEY);
  } catch {
    raw = null;
  }
  if (raw !== snapshotRaw) {
    snapshotRaw = raw;
    try {
      snapshot = raw ? (JSON.parse(raw) as Follows) : {};
    } catch {
      snapshot = {};
    }
  }
  return snapshot;
}

export function useFollows(kind: FollowKind): string[] {
  const get = useCallback(() => currentFollows()[kind] ?? EMPTY, [kind]);
  return useSyncExternalStore(subscribe, get, () => EMPTY);
}

export function FollowButton({
  kind, id, label, className,
}: {
  kind: FollowKind;
  id: string;
  /** What is being followed, for the accessible name. */
  label: string;
  className?: string;
}) {
  const followed = useFollows(kind);
  const following = followed.includes(id);

  const toggle = () => {
    const all = read();
    const current = all[kind] ?? [];
    write({
      ...all,
      [kind]: current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current, id],
    });
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={following}
      aria-label={following ? `Unfollow ${label}` : `Follow ${label}`}
      className={cn(
        "inline-flex h-8 items-center gap-1.5 rounded-full px-3.5 text-[12px] font-semibold transition-colors",
        following
          ? "bg-ink text-base hover:bg-ink/90"
          : "border border-stroke-hi bg-transparent text-ink hover:bg-elevated",
        className,
      )}
    >
      <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden
           fill={following ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.6">
        <path d="M8 2.2 9.9 6l4.1.6-3 2.9.7 4.1L8 11.7 4.3 13.6l.7-4.1-3-2.9L6.1 6z"
              strokeLinejoin="round" />
      </svg>
      {following ? "Following" : "Follow"}
    </button>
  );
}
