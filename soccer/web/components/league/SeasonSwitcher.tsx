"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Select } from "@/components/ui/Select";
import { seasonLabel } from "@/lib/competition";
import type { LeagueSeason } from "@/lib/types";

export function SeasonSwitcher({
  seasons, selected, code,
}: { seasons: LeagueSeason[]; selected: number | null; code: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  if (seasons.length === 0) return null;

  function change(year: string) {
    const next = new URLSearchParams(params.toString());
    next.set("season_year", year);
    router.push(`${pathname}?${next.toString()}`);
  }

  return (
    <Select
      ariaLabel="Season"
      value={String(selected ?? seasons[0].season_year)}
      onChange={change}
      widthClass="w-[150px]"
      align="end"
      options={seasons.map((s) => ({
        value: String(s.season_year),
        label: seasonLabel(s.season_year, code),
        hint: `${s.matches} matches`,
      }))}
    />
  );
}
