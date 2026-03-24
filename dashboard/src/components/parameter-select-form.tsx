"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTransition } from "react";
import type { ParameterOption } from "@/lib/types";

export function ParameterSelectForm({
  options,
  selectedParameter,
}: {
  options: ParameterOption[];
  selectedParameter: string | null;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = useTransition();

  return (
    <label className="flex items-center gap-2 text-sm text-stone-700">
      <span className="sr-only">Select parameter</span>
      <select
        name="param"
        value={selectedParameter ?? ""}
        onChange={(event) => {
          const nextParams = new URLSearchParams(searchParams.toString());
          const value = event.target.value;
          if (value) {
            nextParams.set("param", value);
          } else {
            nextParams.delete("param");
          }
          startTransition(() => {
            router.replace(nextParams.size > 0 ? `${pathname}?${nextParams.toString()}` : pathname, {
              scroll: false,
            });
          });
        }}
        className="rounded-full border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800 disabled:cursor-wait disabled:opacity-70"
        disabled={isPending}
      >
        {options.map((option) => (
          <option key={option.name} value={option.name}>
            {option.name}
          </option>
        ))}
      </select>
      <span className="text-xs uppercase tracking-[0.12em] text-stone-500">{isPending ? "Refreshing" : "Live"}</span>
    </label>
  );
}
