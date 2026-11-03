import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { Preferences } from "@/components/Preferences";
export const dynamic = "force-dynamic";
export default async function Page({searchParams}: {searchParams: Promise<{competition?: string}>}) {
  const sp = await searchParams; const shell = await loadShell(sp.competition);
  const health = await api.health().catch(() => null);
  return <AppShell shell={shell} title="Settings" breadcrumb="Display preferences and data connection"><div className="grid gap-4 lg:grid-cols-2"><Panel title="Reading preferences"><Preferences /></Panel><Panel title="Data connection"><p className="mb-3 text-sm">{health?.status === "ok" ? "Connected to the football dataset" : "Data service unavailable"}</p><p className="text-sm leading-relaxed text-ink-secondary">Match schedules and historical results power the dashboard. Live score coverage depends on the configured data provider. Predictions are estimates based on the available history.</p></Panel></div></AppShell>;
}
