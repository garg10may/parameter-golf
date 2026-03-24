"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export function AutoRefreshToggle({ intervalMs }: { intervalMs: number }) {
  const router = useRouter();
  const [enabled, setEnabled] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return globalThis.localStorage?.getItem("dashboard:auto-refresh") === "true";
  });

  useEffect(() => {
    globalThis.localStorage?.setItem("dashboard:auto-refresh", enabled ? "true" : "false");
    if (!enabled) {
      return;
    }
    const timer = globalThis.setInterval(() => {
      router.refresh();
    }, intervalMs);
    return () => globalThis.clearInterval(timer);
  }, [enabled, intervalMs, router]);

  return (
    <button
      type="button"
      onClick={() => setEnabled((current) => !current)}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition ${
        enabled
          ? "border-amber-400/60 bg-amber-300/15 text-amber-900"
          : "border-stone-300 bg-white/80 text-stone-600 hover:border-stone-400"
      }`}
    >
      <span
        className={`h-2 w-2 rounded-full ${
          enabled ? "bg-amber-500 shadow-[0_0_0_4px_rgba(245,158,11,0.16)]" : "bg-stone-400"
        }`}
      />
      Auto refresh
      <span className="font-mono text-[11px] text-stone-500">{Math.round(intervalMs / 1000)}s</span>
    </button>
  );
}
