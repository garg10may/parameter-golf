import type { ReactNode } from "react";

export function SectionFrame({
  title,
  eyebrow,
  subtitle,
  action,
  children,
  className = "",
}: {
  title: string;
  eyebrow?: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-[28px] border border-stone-200/80 bg-white/88 p-5 shadow-[0_12px_32px_rgba(28,25,23,0.05)] backdrop-blur ${className}`}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          {eyebrow ? <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500">{eyebrow}</div> : null}
          <h2 className="text-lg font-semibold tracking-tight text-stone-950">{title}</h2>
          {subtitle ? <p className="max-w-2xl text-sm text-stone-600">{subtitle}</p> : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-[24px] border border-stone-200/80 bg-stone-50/90 p-4">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">{label}</div>
      <div className="mt-2 text-2xl font-semibold tracking-tight text-stone-950">{value}</div>
      {hint ? <div className="mt-2 text-sm text-stone-600">{hint}</div> : null}
    </div>
  );
}
