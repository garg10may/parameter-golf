"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/format";

const links = [
  { href: "/", label: "Signals" },
  { href: "/runs", label: "Runs" },
  { href: "/groups", label: "Sweeps" },
];

export function SiteNav() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-col gap-1">
      {links.map((link) => {
        const active = pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href));
        return (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              "rounded-2xl px-3 py-2 text-sm transition",
              active
                ? "bg-black !text-white shadow-[0_12px_24px_rgba(12,10,9,0.22)]"
                : "text-stone-600 hover:bg-stone-200/70 hover:text-stone-900",
            )}
            aria-current={active ? "page" : undefined}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
