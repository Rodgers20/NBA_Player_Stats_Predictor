"use client";
import { useEffect, useState, useSyncExternalStore } from "react";

function subscribe(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener("futball-preferences", callback);
  return () => { window.removeEventListener("storage", callback); window.removeEventListener("futball-preferences", callback); };
}
function snapshot() { return document.documentElement.dataset.density === "compact"; }
export function Preferences() {
  const compact = useSyncExternalStore(subscribe, snapshot, () => false);
  const [saved, setSaved] = useState("");
  function update(value: boolean) {
    document.documentElement.dataset.density = value ? "compact" : "comfortable";
    window.dispatchEvent(new Event("futball-preferences"));
    try { localStorage.setItem("futball-density", value ? "compact" : "comfortable"); setSaved("Preference saved on this device."); }
    catch { setSaved("Applied for this visit. Browser storage is unavailable."); }
  }
  return <div className="space-y-3"><label className="flex min-h-11 cursor-pointer items-center gap-3"><input type="checkbox" checked={compact} onChange={e => update(e.target.checked)} className="h-5 w-5 accent-pitch" /><span>Compact table spacing</span></label><p className="text-sm text-ink-secondary">The dashboard uses a dark stadium palette. Motion follows your device’s reduced-motion preference.</p><p role="status" className="text-sm text-pitch">{saved}</p></div>;
}
export function RestorePreferences() {
  useEffect(() => { try { document.documentElement.dataset.density = localStorage.getItem("futball-density") ?? "comfortable"; } catch {} }, []);
  return null;
}
