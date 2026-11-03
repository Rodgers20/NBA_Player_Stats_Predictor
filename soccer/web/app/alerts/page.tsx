import Link from "next/link";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { FixtureList } from "@/components/league/FixtureList";
import { withCompetition } from "@/lib/competition";
export const dynamic = "force-dynamic";
export default async function Page({searchParams}: {searchParams: Promise<{competition?: string}>}) {
    const sp = await searchParams; const shell = await loadShell(sp.competition);
    const fixtures = await api.leagueFixtures(shell.competition, 2);
    return <AppShell shell={shell} title="Match updates" breadcrumb="Your next two days of football" action={<CompetitionSwitcher leagues={shell.leagues} selected={shell.competition} />}><div className="space-y-4"><Panel title="Schedule watch"><p className="mb-4 text-sm leading-relaxed text-ink-secondary">Updates are based on scheduled kick-off times. In-play scores are not verified here. Refresh this page for the latest available schedule; background notifications are not enabled.</p><FixtureList fixtures={fixtures} limit={30} /></Panel><Link className="inline-flex min-h-11 items-center text-sm text-flood transition-colors hover:text-ink" href={withCompetition("/live", shell.competition)}>Open match centre →</Link></div></AppShell>;
}
