import "server-only";

import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";

declare global {
  var __experimentDashboardDb: Database.Database | undefined;
}

export function getDatabasePath() {
  const configuredPath = process.env.EXPERIMENT_DB_PATH ?? "../logs/experiments.sqlite3";
  return path.resolve(/* turbopackIgnore: true */ process.cwd(), configuredPath);
}

export function databaseExists() {
  return fs.existsSync(getDatabasePath());
}

export function getDb() {
  if (!databaseExists()) {
    return null;
  }
  if (!global.__experimentDashboardDb) {
    const db = new Database(getDatabasePath(), {
      readonly: true,
      fileMustExist: true,
    });
    db.pragma("query_only = true");
    global.__experimentDashboardDb = db;
  }
  return global.__experimentDashboardDb;
}
