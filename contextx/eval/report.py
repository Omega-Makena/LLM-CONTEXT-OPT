"""Render eval results as an aligned table (plain text or Markdown)."""

from __future__ import annotations


def _fmt(cell: tuple[float, float, float]) -> str:
    mean, lo, hi = cell
    return f"{mean:.2f} [{lo:.2f},{hi:.2f}]"


def format_results(
    results: dict[str, dict[str, tuple[float, float, float]]],
    title: str = "RETRIEVAL EVAL",
    markdown: bool = False,
) -> str:
    configs = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    rows = [[cfg] + [_fmt(results[cfg][m]) for m in metrics] for cfg in configs]
    header = ["config"] + metrics
    widths = [
        max(len(header[c]), *(len(r[c]) for r in rows)) for c in range(len(header))
    ]

    def line(cells: list[str]) -> str:
        if markdown:
            return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out: list[str] = []
    if not markdown:
        out += ["=" * (sum(widths) + 2 * len(widths)), title,
                "=" * (sum(widths) + 2 * len(widths))]
    out.append(line(header))
    if markdown:
        out.append("|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|")
    else:
        out.append("-" * (sum(widths) + 2 * len(widths)))
    out += [line(r) for r in rows]
    out.append("(cells: mean [95% bootstrap CI])")
    return "\n".join(out)
