## Parameter Golf Dashboard

Launch-enabled Next.js dashboard for experiment runs stored in the local SQLite tracker.

### Development

From `dashboard/`:

```bash
npm run dev
```

The app reads the experiment database from `../logs/experiments.sqlite3` by default.

Override it with:

```bash
EXPERIMENT_DB_PATH=/absolute/or/relative/path/to/experiments.sqlite3 npm run dev
```

Optional refresh interval override:

```bash
DASHBOARD_REFRESH_MS=5000 npm run dev
```

### Checks

```bash
npm run check
npm run build
```

### What the dashboard shows

- `/`
  signals-first overview, best runs, sweep deltas, parameter impact, speed-vs-quality, anomaly panel
- `/runs`
  filterable run ledger
- `/runs/[runId]`
  full run detail with curves, params, events, and artifacts
- `/groups`
  sweep/group list
- `/groups/[group]`
  baseline-vs-variants comparison for a sweep
- `/launch`
  host detection, trainer selection, advanced-block configuration, and background run launch

### Ranking rule

The dashboard ranks runs by the best available metric in this order:

1. `roundtrip.val_bpb`
2. `val.val_bpb`
3. final `train.train_loss`

Train-only runs are marked as `Proxy only` so they are not confused with validation-backed winners.
