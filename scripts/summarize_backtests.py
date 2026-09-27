"""Backtesting.py HTML 결과를 포트폴리오용 표와 SVG로 요약한다."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np


HTML_PATTERN = "test_df_backtest_results_fold_*.html"
CSV_FIELDS = [
    "ticker",
    "fold",
    "test_start",
    "test_end",
    "bars",
    "model_return_pct",
    "buy_hold_return_pct",
    "excess_return_pct",
    "max_drawdown_pct",
    "trades",
    "wins",
    "losses",
    "win_rate_pct",
    "best_trade_pct",
    "worst_trade_pct",
]


def decode_array(value):
    if not isinstance(value, dict) or value.get("type") != "ndarray":
        return value

    array = value.get("array")
    if isinstance(array, list):
        return np.asarray(array)
    if not isinstance(array, dict) or array.get("type") != "bytes":
        raise ValueError("지원하지 않는 Bokeh ndarray 형식입니다.")

    raw = gzip.decompress(base64.b64decode(array["data"]))
    dtype = np.dtype(value["dtype"]).newbyteorder("<")
    return np.frombuffer(raw, dtype=dtype).reshape(value["shape"])


def collect_data_sources(value, output):
    if isinstance(value, dict):
        if value.get("name") == "ColumnDataSource":
            data = value.get("attributes", {}).get("data", {})
            entries = data.get("entries", []) if isinstance(data, dict) else []
            output.append({key: decode_array(item) for key, item in entries})
        for child in value.values():
            collect_data_sources(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_data_sources(child, output)


def parse_document(path: Path):
    source = path.read_text(encoding="utf-8")
    blocks = re.findall(
        r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
        source,
        re.DOTALL,
    )
    if not blocks:
        raise ValueError(f"Bokeh JSON을 찾을 수 없습니다: {path}")

    document = json.loads(html.unescape(blocks[0]))
    data_sources = []
    collect_data_sources(document, data_sources)
    return data_sources


def percent(value):
    return round(float(value), 4)


def extract_fold_metrics(path: Path, ticker: str):
    data_sources = parse_document(path)
    market = next(
        (
            source
            for source in data_sources
            if {"Close", "datetime"}.issubset(source)
        ),
        None,
    )
    if market is None:
        available = [sorted(source) for source in data_sources if source]
        raise ValueError(f"가격·자산 데이터가 없습니다: {path} / {available}")
    trades = next(
        (source for source in data_sources if "returns" in source),
        None,
    )

    close = np.asarray(market["Close"], dtype=float)
    timestamps = np.asarray(market["datetime"], dtype=np.int64)
    if len(close) == 0:
        raise ValueError(f"가격 데이터가 비어 있습니다: {path}")

    if "equity" in market:
        equity = np.asarray(market["equity"], dtype=float)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity / peak - 1.0) * 100.0
        model_return = (equity[-1] / equity[0] - 1.0) * 100.0
        max_drawdown = drawdown.min()
    else:
        model_return = 0.0
        max_drawdown = 0.0
    buy_hold_return = (close[-1] / close[0] - 1.0) * 100.0

    trade_returns = (
        np.asarray(trades["returns"], dtype=float) * 100.0
        if trades is not None
        else np.asarray([], dtype=float)
    )
    wins = int((trade_returns > 0).sum())
    losses = int((trade_returns <= 0).sum())

    fold_match = re.search(r"fold_(\d+)", path.stem)
    if not fold_match:
        raise ValueError(f"파일명에서 폴드 번호를 찾을 수 없습니다: {path.name}")

    to_date = lambda timestamp: datetime.fromtimestamp(
        int(timestamp) / 1000, tz=timezone.utc
    ).date().isoformat()

    return {
        "ticker": ticker,
        "fold": int(fold_match.group(1)),
        "test_start": to_date(timestamps[0]),
        "test_end": to_date(timestamps[-1]),
        "bars": len(close),
        "model_return_pct": percent(model_return),
        "buy_hold_return_pct": percent(buy_hold_return),
        "excess_return_pct": percent(model_return - buy_hold_return),
        "max_drawdown_pct": percent(max_drawdown),
        "trades": len(trade_returns),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": percent(wins / len(trade_returns) * 100.0)
        if len(trade_returns)
        else 0.0,
        "best_trade_pct": percent(trade_returns.max()) if len(trade_returns) else 0.0,
        "worst_trade_pct": percent(trade_returns.min()) if len(trade_returns) else 0.0,
    }


def summarize(rows):
    model_returns = [row["model_return_pct"] for row in rows]
    buy_hold_returns = [row["buy_hold_return_pct"] for row in rows]
    drawdowns = [row["max_drawdown_pct"] for row in rows]
    excess_returns = [row["excess_return_pct"] for row in rows]

    return {
        "fold_count": len(rows),
        "mean_model_return_pct": percent(fmean(model_returns)),
        "std_model_return_pct": percent(pstdev(model_returns)),
        "mean_buy_hold_return_pct": percent(fmean(buy_hold_returns)),
        "mean_excess_return_pct": percent(fmean(excess_returns)),
        "mean_max_drawdown_pct": percent(fmean(drawdowns)),
        "positive_model_folds": sum(value > 0 for value in model_returns),
        "outperformed_buy_hold_folds": sum(value > 0 for value in excess_returns),
        "total_trades": sum(row["trades"] for row in rows),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_chart(path: Path, title: str, labels, series, colors, y_label: str):
    width, height = 1000, 520
    left, right, top, bottom = 90, 30, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for values in series.values() for value in values] + [0]
    minimum, maximum = min(all_values), max(all_values)
    padding = max((maximum - minimum) * 0.12, 1.0)
    minimum -= padding
    maximum += padding

    def y(value):
        return top + (maximum - value) / (maximum - minimum) * plot_height

    groups = len(labels)
    names = list(series)
    group_width = plot_width / groups
    bar_width = min(48, group_width * 0.7 / len(names))
    zero_y = y(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172033">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#94a3b8" stroke-width="1"/>',
        f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#475569">{html.escape(y_label)}</text>',
    ]

    for tick in np.linspace(minimum, maximum, 6):
        tick_y = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{tick_y:.2f}" x2="{width-right}" y2="{tick_y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{tick_y+5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{tick:.1f}</text>'
        )

    for group_index, label in enumerate(labels):
        center = left + group_width * (group_index + 0.5)
        start = center - bar_width * len(names) / 2
        for series_index, name in enumerate(names):
            value = series[name][group_index]
            value_y = y(value)
            rect_y = min(zero_y, value_y)
            rect_height = max(abs(zero_y - value_y), 1)
            x = start + series_index * bar_width
            parts.append(
                f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_width-4:.2f}" height="{rect_height:.2f}" rx="3" fill="{colors[name]}"/>'
            )
            text_y = value_y - 7 if value >= 0 else value_y + 16
            parts.append(
                f'<text x="{x+(bar_width-4)/2:.2f}" y="{text_y:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#334155">{value:.1f}</text>'
            )
        parts.append(
            f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="600" fill="#334155">{html.escape(label)}</text>'
        )

    legend_x = width - right - 185 * len(names)
    for index, name in enumerate(names):
        x = legend_x + index * 185
        parts.append(f'<rect x="{x}" y="{height-34}" width="14" height="14" rx="2" fill="{colors[name]}"/>')
        parts.append(f'<text x="{x+21}" y="{height-22}" font-family="Arial, sans-serif" font-size="13" fill="#334155">{html.escape(name)}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()

    all_rows = []
    ticker_summaries = {}
    for ticker_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        html_files = sorted(
            (ticker_dir / "backtests").glob(HTML_PATTERN),
            key=lambda path: int(re.search(r"fold_(\d+)", path.stem).group(1)),
        )
        if not html_files:
            continue
        rows = [extract_fold_metrics(path, ticker_dir.name) for path in html_files]
        write_csv(ticker_dir / "fold_metrics.csv", rows)
        summary = summarize(rows)
        summary["ticker"] = ticker_dir.name
        (ticker_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_rows.extend(rows)
        ticker_summaries[ticker_dir.name] = summary

    if not all_rows:
        raise SystemExit("요약할 백테스트 HTML이 없습니다.")

    write_csv(results_dir / "fold_metrics.csv", all_rows)
    (results_dir / "summary.json").write_text(
        json.dumps(ticker_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = list(ticker_summaries)
    svg_bar_chart(
        results_dir / "figures" / "return_comparison.svg",
        "Mean Return by Ticker",
        labels,
        {
            "GA-TA-ESN": [ticker_summaries[label]["mean_model_return_pct"] for label in labels],
            "Buy & Hold": [ticker_summaries[label]["mean_buy_hold_return_pct"] for label in labels],
        },
        {"GA-TA-ESN": "#16a34a", "Buy & Hold": "#2563eb"},
        "Return (%)",
    )
    svg_bar_chart(
        results_dir / "figures" / "drawdown_comparison.svg",
        "Mean Maximum Drawdown by Ticker",
        labels,
        {"MDD": [ticker_summaries[label]["mean_max_drawdown_pct"] for label in labels]},
        {"MDD": "#dc2626"},
        "Maximum Drawdown (%)",
    )

    print(json.dumps(ticker_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
