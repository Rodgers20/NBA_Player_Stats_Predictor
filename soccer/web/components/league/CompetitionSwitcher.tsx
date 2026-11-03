"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Select } from "@/components/ui/Select";
import { CompetitionCrest } from "@/components/CompetitionCrest";
import type { League } from "@/lib/types";

/** Scopes the whole app to one competition, via the URL so it is shareable. */
export function CompetitionSwitcher({
  leagues, selected,
}: { leagues: League[]; selected: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  function change(code: string) {
    const next = new URLSearchParams(params.toString());
    next.set("competition", code);
    next.delete("season_year"); // a season from another league is meaningless
    router.push(`${pathname}?${next.toString()}`);
  }

  return (
    <Select
      ariaLabel="Competition"
      value={selected}
      onChange={change}
      widthClass="w-[210px]"
      options={leagues.map((l) => ({
        value: l.code,
        label: l.name,
        hint: l.country,
        icon: <CompetitionCrest code={l.code} size={16} />,
      }))}
    />
  );
}
