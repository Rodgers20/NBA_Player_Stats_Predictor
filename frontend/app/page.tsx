"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "motion";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import type { PropsResponse } from "@/lib/types";
import { PropCard }    from "@/components/props/PropCard";
import { PropFilters } from "@/components/props/PropFilters";

const PAGE = 30;

export default function PropsPage() {
  const { defaultStat, defaultSort } = usePrefs();
  const [filters, setFilters] = useState({
    stat:      defaultStat,
    direction: "over",
    sort:      defaultSort,
    game:      "",
    locksOnly: false,
  });
  const [page, setPage] = useState(1);

  const params = {
    direction: filters.direction,
    sort:      filters.sort,
    limit:     PAGE * page,
    ...(filters.stat      && { stat: filters.stat }),
    ...(filters.game      && { game: filters.game }),
    ...(filters.locksOnly && { locks_only: true }),
  };

  const { data, isPending, isError } = useQuery<PropsResponse>({
    queryKey:       ["props", params],
    queryFn:        () => api.props(params),
    refetchInterval: 120_000,
  });

  function handleChange(update: Partial<typeof filters>) {
    setFilters((f) => ({ ...f, ...update }));
    setPage(1);
  }

  return (
    <div className="px-4 pt-3 max-w-2xl mx-auto">
      <div className="mb-4">
        <h1 className="text-2xl font-black tracking-tight" style={{ color: "var(--color-text-pri)", fontFamily: "var(--font-display)" }}>
          Best Props
        </h1>
        {data?.target_date && (
          <p className="text-xs mt-0.5" style={{ color: "var(--color-text-sec)" }}>{data.target_date}</p>
        )}
      </div>

      <PropFilters
        filters={filters}
        matchups={data?.game_matchups ?? []}
        onChange={handleChange}
      />

      {data && (
        <p className="text-xs mt-3 mb-2" style={{ color: "var(--color-text-sec)" }}>
          {data.count} props · showing {data.props.length}
        </p>
      )}

      {isPending && (
        <div className="flex flex-col gap-3 mt-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="glass h-36 animate-pulse" />
          ))}
        </div>
      )}

      {isError && (
        <div className="glass p-4 mt-4 text-center text-sm" style={{ color: "var(--color-danger)" }}>
          Backend offline — run <code className="font-mono text-xs bg-white/5 px-1.5 py-0.5 rounded">uvicorn api.main:app --port 8000</code>
        </div>
      )}

      {!isPending && data?.props.length === 0 && (
        <div className="glass p-8 mt-4 text-center text-sm" style={{ color: "var(--color-text-sec)" }}>
          No props match the current filters.
        </div>
      )}

      {!isPending && data && data.props.length > 0 && (
        <>
          <AnimatePresence mode="popLayout">
            <div className="flex flex-col gap-3 mt-1">
              {data.props.map((p, i) => (
                <motion.div
                  key={`${p.player}-${p.stat}`}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(i * 0.03, 0.3), duration: 0.25 }}
                >
                  <PropCard prop={p} />
                </motion.div>
              ))}
            </div>
          </AnimatePresence>

          {data.props.length < data.count && (
            <button
              onClick={() => setPage((n) => n + 1)}
              className="w-full mt-4 py-3 rounded-xl text-sm font-semibold transition-colors"
              style={{
                color: "var(--color-teal-400)",
                border: "1px solid color-mix(in srgb, var(--color-teal-400) 30%, transparent)",
                background: "color-mix(in srgb, var(--color-teal-400) 6%, transparent)",
              }}
            >
              Load more ({data.count - data.props.length} remaining)
            </button>
          )}
        </>
      )}
    </div>
  );
}
