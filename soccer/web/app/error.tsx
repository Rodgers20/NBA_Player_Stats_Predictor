"use client";
import Link from "next/link";
export default function ErrorPage({reset}: {error: Error & {digest?: string}; reset: () => void}) {
  return <main className="grid min-h-screen place-items-center p-5"><div className="panel max-w-lg space-y-4 p-7"><h1 className="font-display text-xl font-semibold">We couldn’t load this view</h1><p className="text-sm leading-relaxed text-ink-secondary">The data service may be unavailable. Try again, or return to the dashboard.</p><div className="flex flex-wrap gap-4"><button onClick={reset} className="min-h-11 rounded-lg bg-pitch px-4 font-semibold text-[#04160C]">Try again</button><Link href="/" className="inline-flex min-h-11 items-center text-flood transition-colors hover:text-ink">Dashboard</Link></div></div></main>;
}
