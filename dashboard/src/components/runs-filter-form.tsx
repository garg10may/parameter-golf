import Form from "next/form";

export function RunsFilterForm({
  groups,
  defaults,
}: {
  groups: string[];
  defaults: Record<string, string>;
}) {
  return (
    <Form action="" className="grid gap-3 rounded-[24px] border border-stone-200/80 bg-stone-50/90 p-4 lg:grid-cols-7">
      <input
        type="search"
        name="q"
        defaultValue={defaults.q}
        placeholder="Search run ID, label, comment"
        className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800 outline-none placeholder:text-stone-400 lg:col-span-2"
      />
      <select name="backend" defaultValue={defaults.backend} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800">
        <option value="all">All backends</option>
        <option value="mlx">MLX</option>
        <option value="cuda">CUDA</option>
      </select>
      <select name="status" defaultValue={defaults.status} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800">
        <option value="all">All statuses</option>
        <option value="completed">Completed</option>
        <option value="running">Running</option>
      </select>
      <select name="group" defaultValue={defaults.group} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800">
        <option value="all">All groups</option>
        {groups.map((group) => (
          <option key={group} value={group}>
            {group}
          </option>
        ))}
      </select>
      <select name="phase" defaultValue={defaults.phase} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800">
        <option value="all">All trust levels</option>
        <option value="roundtrip">Trusted</option>
        <option value="val">Validated</option>
        <option value="train">Proxy only</option>
      </select>
      <div className="grid grid-cols-2 gap-3 lg:col-span-7 lg:grid-cols-[1fr_1fr_1fr_1fr_auto]">
        <input type="date" name="from" defaultValue={defaults.from} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800" />
        <input type="date" name="to" defaultValue={defaults.to} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800" />
        <select name="sort" defaultValue={defaults.sort} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800">
          <option value="started">Started time</option>
          <option value="best_metric">Best metric</option>
          <option value="train_loss">Train loss</option>
          <option value="duration">Duration</option>
          <option value="params">Parameter count</option>
        </select>
        <select name="dir" defaultValue={defaults.dir} className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800">
          <option value="desc">Descending</option>
          <option value="asc">Ascending</option>
        </select>
        <button
          type="submit"
          className="rounded-2xl bg-stone-950 px-4 py-2 text-sm font-medium text-stone-50 transition hover:bg-stone-800"
        >
          Apply filters
        </button>
      </div>
    </Form>
  );
}
