"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

/** A retired page that now lives inside another one. Static export has no server redirects, so redirect on load. */
export function MovedTo({ href, label }: { href: string; label: string }) {
  const router = useRouter();
  useEffect(() => { router.replace(href); }, [href, router]);
  return <main className="ql-page mx-auto max-w-[1100px] px-4 py-10 sm:px-8">
    <p className="text-sm text-text-sec">This page moved to <Link href={href} className="font-semibold text-teal-400 underline">{label}</Link>.</p>
  </main>;
}
