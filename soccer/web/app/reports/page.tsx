import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { DownloadReport } from "@/components/DownloadReport";
export const dynamic = "force-dynamic";
export default async function Page({searchParams}: {searchParams: Promise<{competition?: string}>}) {
    const sp = await searchParams; const shell = await loadShell(sp.competition);
    const matches = await api.leagueMatches(shell.competition, {limit: 500});
    const goals = matches.rows.reduce((n,r) => n+r.total_goals, 0);
    return <AppShell shell={shell} title="Season report" breadcrumb={`${matches.competition} · ${matches.season ?? "No season data"}`} action={<CompetitionSwitcher leagues={shell.leagues} selected={shell.competition} />}><Panel title="Results at a glance"><div className="space-y-5"><p className="text-ink-secondary">Export the recorded results for the displayed season. The report contains {matches.rows.length} matches and {goals} goals.</p>{matches.is_fallback && <p className="text-sm text-ink-secondary">Current-season results are unavailable; this report uses {matches.season}.</p>}<DownloadReport rows={matches.rows} name={`${shell.competition}-${matches.season ?? "results"}`} /><p className="text-xs text-ink-muted">Source: the local football results dataset. Up to 500 matches per report.</p></div></Panel></AppShell>;
}
