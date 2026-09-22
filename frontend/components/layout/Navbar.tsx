"use client";

export function Navbar() {
  return (
    <header className="fixed top-0 inset-x-0 z-50 h-14 flex items-center px-4 border-b border-white/5 bg-[#0B101A]/90 backdrop-blur-[16px]">
      <div className="flex items-center gap-2">
        <span className="text-[#2dd4bf] text-xl font-bold tracking-tight">🏀</span>
        <span className="text-[#f0f4ff] text-base font-bold tracking-tight">NBA Props AI</span>
        <span className="ml-2 text-[10px] font-semibold uppercase tracking-widest text-[#2dd4bf] bg-[#2dd4bf]/10 border border-[#2dd4bf]/30 px-1.5 py-0.5 rounded-full">
          Beta
        </span>
      </div>
    </header>
  );
}
