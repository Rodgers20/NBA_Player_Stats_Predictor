"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "motion/react";
import { useSlip } from "@/store/slip";
import { usePrefs } from "@/store/prefs";
import { useEffect, useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime:   60_000,
            gcTime:      5 * 60_000,
            retry:       2,
            refetchOnWindowFocus: false,
          },
        },
      })
  );
  useEffect(() => { void usePrefs.persist.rehydrate(); void useSlip.persist.rehydrate(); }, []);
  return <QueryClientProvider client={client}><MotionConfig reducedMotion="user">{children}</MotionConfig></QueryClientProvider>;
}
