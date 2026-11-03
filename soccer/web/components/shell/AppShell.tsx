import { Suspense } from "react";
import { api } from "@/lib/api";
import { resolveCompetition } from "@/lib/competition";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import type { League, LiveCounts } from "@/lib/types";

export interface ShellContext {
  leagues: League[];
  competition: string;
  live: LiveCounts;
}

export async function loadShell(requested?: string): Promise<ShellContext> {
  const [leagues, live] = await Promise.all([
    api.leagues().catch((): League[] => []),
    api.liveCounts().catch((): LiveCounts => ({ total: 0, by_competition: {}, matches: [] })),
  ]);
  return { leagues, competition: resolveCompetition(requested, leagues), live };
}

export function AppShell({
  shell, title, breadcrumb, action, children,
}: {
  shell: ShellContext;
  title: string;
  breadcrumb?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <Suspense fallback={<div className="hidden lg:block w-[236px] shrink-0 border-r border-stroke" />}>
        <Sidebar
          leagues={shell.leagues}
          selected={shell.competition}
          liveTotal={shell.live.total}
          liveByCompetition={shell.live.by_competition}
        />
      </Suspense>
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar competition={shell.competition} leagues={shell.leagues} title={title}
                breadcrumb={breadcrumb} action={action} liveTotal={shell.live.total} />
        <main id="main-content" tabIndex={-1} className="min-w-0 flex-1 p-4 lg:p-6">{children}</main>
      </div>
    </div>
  );
}
