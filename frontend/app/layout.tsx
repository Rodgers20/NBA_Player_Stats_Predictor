import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Providers }  from "./providers";
import { BottomNav }  from "@/components/layout/BottomNav";
import { Navbar }     from "@/components/layout/Navbar";

export const metadata: Metadata = {
  title:       "NBA Props AI",
  description: "AI-powered NBA player props and game predictions",
  manifest:    "/manifest.json",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "NBA Props AI" },
};

export const viewport: Viewport = {
  themeColor:   "#0B101A",
  width:        "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <Navbar />
          <main className="pb-24 pt-14 min-h-dvh">{children}</main>
          <BottomNav />
        </Providers>
      </body>
    </html>
  );
}
