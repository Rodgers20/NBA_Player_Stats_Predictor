"use client";
import { useState, useSyncExternalStore } from "react";
const KEY = "futball-followed-players";
function snapshot() {
  try { return localStorage.getItem(KEY) ?? "[]"; } catch { return "[]"; }
}
function subscribe(notify: () => void) {
  window.addEventListener("storage", notify);
  window.addEventListener("futball-followed-players", notify);
  return () => { window.removeEventListener("storage", notify); window.removeEventListener("futball-followed-players", notify); };
}
export function FollowPlayerButton({ id, accent }: { id: string; accent: string }) {
  const raw = useSyncExternalStore(subscribe, snapshot, () => "[]");
  const [error, setError] = useState(false);
  let ids: string[] = [];
  try { const parsed = JSON.parse(raw); if (Array.isArray(parsed)) ids = parsed.filter((v): v is string => typeof v === "string"); } catch { /* reset invalid local preferences */ }
  const followed = ids.includes(id);
  return <div className="shrink-0 text-right">
    <button type="button" aria-pressed={followed} title="Saved on this device"
      onClick={() => {
        try { localStorage.setItem(KEY, JSON.stringify(followed ? ids.filter(v => v !== id) : [...ids, id])); window.dispatchEvent(new Event(KEY)); setError(false); }
        catch { setError(true); }
      }}
      className="min-h-11 rounded-full bg-white px-5 text-xs font-bold shadow-sm transition-transform hover:scale-[1.03] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white"
      style={{ color: /^#(?:fff|ffffff)$/i.test(accent) ? "#111827" : accent }}>{followed ? "Following" : "Follow"}</button>
    {error && <p role="status" className="mt-2 text-xs">Could not save on this device.</p>}
  </div>;
}
