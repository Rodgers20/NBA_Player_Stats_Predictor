import { api } from "@/lib/api";
import { resolveCompetition } from "@/lib/competition";
import { BottomNav, Header, Rail } from "./Nav";
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

/**
 * Page chrome. `breadcrumb` and `action` are accepted so existing pages keep
 * compiling; both are ignored on purpose: the header's competition pills do
 * the job of the old switcher, and each page carries its own heading.
 */
export function AppShell({
  shell, title, children,
}: {
  shell: ShellContext;
  title: string;
  breadcrumb?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <Rail competition={shell.competition} />
      <Header leagues={shell.leagues} competition={shell.competition} />
      <main id="main-content" tabIndex={-1} className="app-main">
        <h1 className="sr-only">{title}</h1>
        <div className="mx-auto w-full max-w-[1500px]">{children}</div>
      </main>
      <BottomNav competition={shell.competition} />
    </>
  );
}
