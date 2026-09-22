"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/",          label: "Props",    icon: "🎯" },
  { href: "/games",     label: "Games",    icon: "🏟️" },
  { href: "/analysis",  label: "Analysis", icon: "📊" },
];

export function BottomNav() {
  const path = usePathname();
  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 h-16 flex items-center justify-around border-t border-white/5 bg-[#0B101A]/95 backdrop-blur-[16px] safe-area-pb">
      {NAV.map(({ href, label, icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex flex-col items-center gap-0.5 text-xs font-medium transition-colors min-w-[60px]",
              active ? "text-[#2dd4bf]" : "text-[#8ca0c0]"
            )}
          >
            <span className="text-xl leading-none">{icon}</span>
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
