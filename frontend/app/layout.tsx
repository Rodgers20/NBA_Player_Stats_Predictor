import type { Metadata, Viewport } from "next";
import "./globals.css";
import "./quant-lab.css";
import { Providers }  from "./providers";
import { BottomNav }  from "@/components/layout/BottomNav";
import { Navbar }     from "@/components/layout/Navbar";

export const metadata: Metadata = {
  title:       "NBA Props AI",
  description: "NBA and WNBA player research, projections, and game insights",
  manifest:    "/manifest.json",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "NBA Props AI" },
};

export const viewport: Viewport = {
  themeColor:   "#0d151b",
  width:        "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <Navbar />
          <a href="#main-content" className="skip-link">Skip to content</a>
          <main id="main-content" className="app-main min-h-dvh">{children}</main>
          <BottomNav />
        </Providers>
      </body>
    </html>
  );
}
