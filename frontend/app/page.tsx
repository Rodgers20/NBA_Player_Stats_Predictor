"use client";

import { useState, useCallback } from "react";
import useSWR from "swr";
import { fetchProps } from "@/lib/api";
import type { PropsResponse } from "@/lib/types";
import { PropCard } from "@/components/props/PropCard";
import { PropFilters } from "@/components/props/PropFilters";

const PAGE = 30;

export default function PropsPage() {
  const [filters, setFilters] = useState({
    stat:      "",
    direction: "Over",
    sort:      "ev",
    game:      "",
    locksOnly: false,
  });
  const [page, setPage] = useState(1);

  const params = {
    ...(filters.stat      && { stat: filters.stat }),
    direction:  filters.direction === "All" ? "all" : filters.direction.toLowerCase(),
    sort:       filters.sort,
    ...(filters.game      && { game: filters.game }),
    ...(filters.locksOnly && { locks_only: true }),
    limit: PAGE * page,
  };

  const { data, isLoading, error } = useSWR<PropsResponse>(
    ["props", params],
    () => fetchProps(params),
    { refreshInterval: 120_000 }
  );

  const handleChange = useCallback((update: Partial<typeof filters>) => {
    setFilters((f) => ({ ...f, ...update }));
    setPage(1);
  }, []);

  return (
    <div className="px-4 pt-3 max-w-2xl mx-auto">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-xl font-extrabold text-[#f0f4ff]">Best Props</h1>
        {data?.target_date && (
          <p className="text-xs text-[#8ca0c0] mt-0.5">{data.target_date}</p>
        )}
      </div>

      {/* Filters */}
      <PropFilters
        filters={filters}
        matchups={data?.game_matchups ?? []}
        onChange={handleChange}
      />

      {/* Count */}
      {data && (
        <p className="text-xs text-[#8ca0c0] mt-3 mb-2">
          {data.count} props · showing {Math.min(data.props.length, PAGE * page)}
        </p>
      )}

      {/* List */}
      {isLoading && (
        <div className="flex flex-col gap-3 mt-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="glass h-36 animate-pulse" />
          ))}
        </div>
      )}

      {error && (
        <div className="glass p-4 mt-4 text-center text-sm text-[#f87171]">
          Failed to load props — is the backend running?
        </div>
      )}

      {!isLoading && data && (
        <>
          <div className="flex flex-col gap-3 mt-1">
            {data.props.map((p, i) => (
              <PropCard key={`${p.player}-${p.stat}-${i}`} prop={p} />
            ))}
          </div>

          {data.props.length < data.count && (
            <button
              onClick={() => setPage((n) => n + 1)}
              className="w-full mt-4 py-3 rounded-xl text-sm font-semibold text-[#2dd4bf] border border-[#2dd4bf]/30 bg-[#2dd4bf]/5 hover:bg-[#2dd4bf]/10 transition-colors"
            >
              Load more ({data.count - data.props.length} remaining)
            </button>
          )}
        </>
      )}
    </div>
  );
}
