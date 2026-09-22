import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "bg-root":   "#0B101A",
        "bg-card":   "rgba(15,26,46,0.85)",
        "bg-glass":  "rgba(255,255,255,0.04)",
        "teal":      { 400: "#2dd4bf", 500: "#14b8a6" },
        "text-pri":  "#f0f4ff",
        "text-sec":  "#8ca0c0",
        "border-sub":"rgba(255,255,255,0.05)",
        "lock-gold": "#f59e0b",
        "stat": {
          pts:  "#14b8a6",
          ast:  "#f97066",
          reb:  "#a78bfa",
          stl:  "#fbbf24",
          blk:  "#60a5fa",
          fg3m: "#ec4899",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      borderRadius: {
        xl2: "1rem",
        xl3: "1.25rem",
      },
      backdropBlur: {
        glass: "16px",
      },
    },
  },
  plugins: [],
};

export default config;
