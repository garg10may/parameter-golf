"use client";

import type { EChartsOption } from "echarts";
import { EChart } from "@/components/echart";
import { formatMetricLabel, formatNumber } from "@/lib/format";
import type { ParameterImpactPoint, RunSummary } from "@/lib/types";

export function SweepDeltaChart({
  items,
}: {
  items: Array<{
    group: string;
    label: string;
    metricName: string;
    delta: number;
    baselineValue: number;
    metricValue: number;
    proxyOnly: boolean;
  }>;
}) {
  const option: EChartsOption = {
    animationDuration: 350,
    grid: { top: 16, left: 88, right: 20, bottom: 20 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const first = Array.isArray(params) ? (params[0] as { data: typeof items[number] }) : null;
        if (!first) return "";
        const item = first.data;
        return [
          `<strong>${item.group} / ${item.label}</strong>`,
          `${formatMetricLabel(item.metricName as never)}: ${formatNumber(item.metricValue)}`,
          `Baseline: ${formatNumber(item.baselineValue)}`,
          `Delta: ${item.delta >= 0 ? "+" : ""}${formatNumber(item.delta)}`,
          item.proxyOnly ? "Proxy only" : "Trusted",
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      axisLabel: { color: "#78716c" },
      splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.14)" } },
    },
    yAxis: {
      type: "category",
      data: items.map((item) => `${item.group} · ${item.label}`),
      axisLabel: { color: "#57534e", width: 180, overflow: "truncate" },
    },
    series: [
      {
        type: "bar",
        data: items.map((item) => ({
          value: item.delta,
          metaLabel: item.label,
          group: item.group,
          metricName: item.metricName,
          delta: item.delta,
          baselineValue: item.baselineValue,
          metricValue: item.metricValue,
          proxyOnly: item.proxyOnly,
          itemStyle: {
            color: item.delta <= 0 ? "#0f766e" : "#b45309",
            borderRadius: 999,
          },
        })),
      },
    ] as unknown as EChartsOption["series"],
  };

  return <EChart option={option} height={Math.max(280, items.length * 34)} />;
}

export function LearningCurvesChart({
  series,
  yAxisName,
}: {
  series: Array<{ label: string; points: Array<{ step: number; value: number }> }>;
  yAxisName: string;
}) {
  const denseLegend = series.length > 3;
  const option: EChartsOption = {
    animationDuration: 300,
    grid: denseLegend
      ? { top: 20, left: 52, right: 196, bottom: 36 }
      : { top: 44, left: 52, right: 20, bottom: 36 },
    tooltip: { trigger: "axis" },
    legend: {
      type: "scroll",
      orient: denseLegend ? "vertical" : "horizontal",
      top: denseLegend ? 20 : 0,
      right: denseLegend ? 0 : undefined,
      left: denseLegend ? undefined : 0,
      bottom: denseLegend ? 18 : undefined,
      width: denseLegend ? 160 : undefined,
      textStyle: { color: "#57534e", fontSize: 11 },
      pageTextStyle: { color: "#57534e", fontSize: 11 },
      formatter: (value: string) => (value.length > 22 ? `${value.slice(0, 22)}...` : value),
    },
    xAxis: {
      type: "value",
      name: "Step",
      nameLocation: "middle",
      nameGap: 28,
      axisLabel: { color: "#78716c" },
    },
    yAxis: {
      type: "value",
      name: yAxisName,
      axisLabel: { color: "#78716c" },
      splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.14)" } },
    },
    series: series.map((item, index) => ({
      type: "line",
      name: item.label,
      data: item.points.map((point) => [point.step, point.value]),
      symbol: "none",
      lineStyle: {
        width: index === 0 ? 3 : 2,
        opacity: index < 4 ? 1 : 0.5,
      },
      emphasis: {
        focus: "series",
      },
    })),
  };

  return <EChart option={option} height={360} />;
}

export function ParameterImpactChart({
  points,
  parameterName,
}: {
  points: ParameterImpactPoint[];
  parameterName: string;
}) {
  const numeric = points.every((point) => point.numericValue != null);
  let option: EChartsOption;

  if (numeric) {
    const trend = buildTrendLine(points);
    option = {
      animationDuration: 300,
      grid: { top: 18, left: 54, right: 20, bottom: 38 },
      tooltip: {
        trigger: "item",
        formatter: (param: unknown) => {
          const item = param as { data: { metaLabel: string; value: [number, number]; sampleCount: number; proxyOnly: boolean } };
          return [
            `<strong>${item.data.metaLabel}</strong>`,
            `${parameterName}: ${item.data.value[0]}`,
            `Metric: ${formatNumber(item.data.value[1])}`,
            `Sample count: ${item.data.sampleCount}`,
            item.data.proxyOnly ? "Proxy only" : "Trusted",
          ].join("<br/>");
        },
      },
      xAxis: {
        type: "value",
        name: parameterName,
        axisLabel: { color: "#78716c" },
      },
      yAxis: {
        type: "value",
        name: formatMetricLabel(points[0]?.metricName ?? null),
        axisLabel: { color: "#78716c" },
        splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.14)" } },
      },
      series: [
        {
          type: "scatter",
          symbolSize: 10,
          data: points.map((point) => ({
            value: [point.numericValue!, point.metricValue],
            metaLabel: point.label,
            sampleCount: point.sampleCount,
            proxyOnly: point.proxyOnly,
            itemStyle: {
              color: point.proxyOnly ? "#d97706" : "#0f766e",
            },
          })),
        },
        {
          type: "line",
          data: trend,
          symbol: "none",
          lineStyle: { color: "#44403c", width: 2, type: "dashed" },
        },
      ] as unknown as EChartsOption["series"],
    };
  } else {
    const grouped = new Map<string, ParameterImpactPoint[]>();
    for (const point of points) {
      const key = String(point.value);
      const list = grouped.get(key) ?? [];
      list.push(point);
      grouped.set(key, list);
    }
    const rows = [...grouped.entries()].map(([value, values]) => ({
      value,
      avg: values.reduce((sum, item) => sum + item.metricValue, 0) / values.length,
      count: values.length,
    }));
    option = {
      animationDuration: 300,
      grid: { top: 18, left: 54, right: 20, bottom: 60 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
      },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.value),
        axisLabel: { color: "#78716c", rotate: 20 },
      },
      yAxis: {
        type: "value",
        name: formatMetricLabel(points[0]?.metricName ?? null),
        axisLabel: { color: "#78716c" },
        splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.14)" } },
      },
      series: [
        {
          type: "bar",
          data: rows.map((row) => ({
            value: row.avg,
            itemStyle: { color: "#1f7668", borderRadius: [8, 8, 0, 0] },
          })),
        },
      ],
    };
  }

  return <EChart option={option} height={320} />;
}

export function SpeedQualityChart({ runs }: { runs: RunSummary[] }) {
  const option: EChartsOption = {
    animationDuration: 300,
    grid: { top: 18, left: 54, right: 24, bottom: 38 },
    tooltip: {
      trigger: "item",
      formatter: (param: unknown) => {
        const item = param as {
          data: {
            label: string;
            value: [number, number];
            metricName: string;
            proxyOnly: boolean;
          };
        };
        return [
          `<strong>${item.data.label}</strong>`,
          `Throughput: ${Math.round(item.data.value[0]).toLocaleString()} tok/s`,
          `${formatMetricLabel(item.data.metricName as never)}: ${formatNumber(item.data.value[1])}`,
          item.data.proxyOnly ? "Proxy only" : "Trusted",
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      name: "Throughput",
      axisLabel: { color: "#78716c" },
    },
    yAxis: {
      type: "value",
      name: "Best available metric",
      axisLabel: { color: "#78716c" },
      splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.14)" } },
    },
    series: [
      {
        type: "scatter",
        symbolSize: 12,
        data: runs.map((run) => ({
          value: [run.finalTokS ?? run.avgTokS ?? 0, run.bestAvailableMetricValue ?? 0],
          metaLabel: run.experimentLabel ?? run.runId,
          metricName: run.bestAvailableMetricName,
          proxyOnly: run.proxyOnly,
          itemStyle: {
            color: run.proxyOnly ? "#d97706" : "#0f766e",
          },
        })),
      },
    ] as unknown as EChartsOption["series"],
  };
  return <EChart option={option} height={320} />;
}

function buildTrendLine(points: ParameterImpactPoint[]) {
  const numericPoints = points
    .filter((point) => point.numericValue != null)
    .map((point) => [point.numericValue!, point.metricValue] as const)
    .sort((a, b) => a[0] - b[0]);
  if (numericPoints.length < 2) {
    return numericPoints;
  }
  const n = numericPoints.length;
  const sumX = numericPoints.reduce((sum, [x]) => sum + x, 0);
  const sumY = numericPoints.reduce((sum, [, y]) => sum + y, 0);
  const sumXY = numericPoints.reduce((sum, [x, y]) => sum + x * y, 0);
  const sumXX = numericPoints.reduce((sum, [x]) => sum + x * x, 0);
  const slope = (n * sumXY - sumX * sumY) / Math.max(n * sumXX - sumX * sumX, 1e-9);
  const intercept = sumY / n - slope * (sumX / n);
  const minX = numericPoints[0][0];
  const maxX = numericPoints[numericPoints.length - 1][0];
  return [
    [minX, slope * minX + intercept],
    [maxX, slope * maxX + intercept],
  ];
}
