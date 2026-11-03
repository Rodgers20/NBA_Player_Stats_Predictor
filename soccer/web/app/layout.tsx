import type { Metadata } from "next";
import { Archivo, Public_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { RestorePreferences } from "@/components/Preferences";
import { Providers } from "@/components/Providers";

// Deliberately not Inter/Roboto. Archivo's condensed weights give scorelines
// presence; Public Sans is a workhorse with a voice; Plex Mono keeps stat
// columns aligned.
const display = Archivo({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "600", "700", "800"],
});
const body = Public_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "FUTBALL — Football Intelligence",
  description:
    "Live 3D football intelligence dashboard built on StatsBomb open data.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body className="min-h-screen antialiased">
        <RestorePreferences /><Providers>{children}</Providers>
      </body>
    </html>
  );
}
