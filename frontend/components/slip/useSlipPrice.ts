"use client";

import { useEffect, useMemo, useState } from 'react';
import { buildPriceRequest, fetchSlipPrice, type SlipPrice } from '@/lib/slip-price.mjs';
import type { SlipLeg } from '@/lib/bet-slip.mjs';
import { STATIC } from '@/lib/journal-client';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? '').replace(/\/$/, '');
const DEBOUNCE_MS = 400;

/**
 * Correlation-adjusted pricing for a parlay. Returns null whenever it can't be had (fewer than two
 * modelled legs, static hosting, API down), so callers keep showing the independent calculation.
 */
export function useSlipPrice(legs: SlipLeg[], enabled: boolean): SlipPrice | null {
  const body = useMemo(() => (enabled && !STATIC ? buildPriceRequest(legs) : null), [legs, enabled]);
  const key = body ? JSON.stringify(body) : '';
  const [result, setResult] = useState<{ key: string; price: SlipPrice } | null>(null);

  useEffect(() => {
    if (!key) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchSlipPrice(BASE, JSON.parse(key), controller.signal)
        .then(price => setResult({ key, price }))
        .catch(() => { /* degrade silently to independent maths */ });
    }, DEBOUNCE_MS);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [key]);

  return key && result?.key === key ? result.price : null;
}
