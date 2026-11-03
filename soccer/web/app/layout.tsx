import type { Metadata, Viewport } from "next";
import { Barlow_Condensed, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/Providers";

// Same three faces as the main Court Vision app, so both sections read as one product.
const display = Barlow_Condensed({ subsets: ["latin"], variable: "--font-barlow", weight: ["500", "600", "700", "800", "900"] });
const body = IBM_Plex_Sans({ subsets: ["latin"], variable: "--font-plex-sans", weight: ["400", "500", "600", "700"] });
const mono = IBM_Plex_Mono({ subsets: ["latin"], variable: "--font-plex-mono", weight: ["400", "500", "600", "700"] });

export const metadata: Metadata = {
  title: "Soccer Props AI",
  description: "Soccer match predictions, player props and player analysis for the Premier League, La Liga, Champions League, Bundesliga and MLS.",
};

export const viewport: Viewport = { themeColor: "#0d151b", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
