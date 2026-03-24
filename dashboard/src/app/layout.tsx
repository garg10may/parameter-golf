import type { Metadata } from "next";
import Link from "next/link";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";
import { AutoRefreshToggle } from "@/components/auto-refresh-toggle";
import { SiteNav } from "@/components/site-nav";
import "./globals.css";

const sans = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  weight: ["400", "500"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Parameter Golf Dashboard",
  description: "Read-only experiment intelligence for SQLite-tracked parameter golf runs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const refreshMs = Number.parseInt(process.env.DASHBOARD_REFRESH_MS ?? "10000", 10);

  return (
    <html
      lang="en"
      className={`${sans.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-canvas text-ink">
        <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(245,158,11,0.12),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(20,184,166,0.1),transparent_30%)]">
          <div className="mx-auto grid min-h-screen max-w-[1680px] gap-6 px-4 py-4 lg:grid-cols-[250px_minmax(0,1fr)] lg:px-6 lg:py-6">
            <aside className="rounded-[32px] border border-stone-200/80 bg-white/82 p-4 shadow-[0_18px_40px_rgba(28,25,23,0.06)] backdrop-blur">
              <div className="border-b border-stone-200/80 pb-4">
                <Link href="/" className="block">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-stone-500">Parameter Golf</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-stone-950">Experiment Atlas</div>
                  <p className="mt-2 text-sm leading-6 text-stone-600">
                    Read the signals, not the raw logs.
                  </p>
                </Link>
              </div>
              <div className="mt-4 space-y-4">
                <SiteNav />
                <div className="rounded-[24px] border border-stone-200 bg-stone-50/85 p-3 text-sm text-stone-600">
                  The ranking rule always prefers trusted roundtrip/validation metrics over train-only proxy scores.
                </div>
              </div>
            </aside>

            <main className="min-w-0 space-y-6">
              <header className="flex flex-col gap-4 rounded-[32px] border border-stone-200/80 bg-white/78 px-5 py-4 shadow-[0_18px_40px_rgba(28,25,23,0.05)] backdrop-blur lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500">SQLite-backed analytics</div>
                  <h1 className="mt-1 text-2xl font-semibold tracking-tight text-stone-950">Signals, sweeps, and suspicious runs</h1>
                </div>
                <div className="flex items-center gap-3">
                  <AutoRefreshToggle intervalMs={Number.isFinite(refreshMs) ? refreshMs : 10000} />
                  <div className="rounded-full border border-stone-300 bg-white/90 px-3 py-1.5 font-mono text-[11px] text-stone-500">
                    read-only
                  </div>
                </div>
              </header>
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
