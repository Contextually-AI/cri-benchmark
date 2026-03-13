"""Report generation for CRI Benchmark results.

Generates benchmark reports in multiple formats:

- **Console** — Rich-formatted terminal output with color-coded scores.
- **JSON** — Machine-readable JSON for programmatic consumption.
- **Markdown** — Human-readable document with tables and breakdowns.
- **Comparison** — Side-by-side Markdown table for multiple systems.

Supports :class:`~cri.models.BenchmarkResult` and
:class:`~cri.models.CRIResult` models.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cri import __version__
from cri.models import (
    BenchmarkResult,
    CRIResult,
    Dimension,
    DimensionResult,
    PerformanceProfile,
)

logger = logging.getLogger(__name__)

# Type alias for any result the reporter can handle
AnyResult = BenchmarkResult | CRIResult

# Canonical dimension order for tables — all known dimensions in display order.
_DIMENSION_ORDER: list[Dimension] = [
    Dimension.PAS,
    Dimension.DBU,
    Dimension.MEI,
    Dimension.TC,
    Dimension.CRQ,
    Dimension.QRP,
    Dimension.SFC,
]

_DIMENSION_FULL_NAMES: dict[Dimension, str] = {
    Dimension.PAS: "Persona Accuracy Score",
    Dimension.DBU: "Dynamic Belief Updating",
    Dimension.MEI: "Memory Efficiency Index",
    Dimension.TC: "Temporal Coherence",
    Dimension.CRQ: "Conflict Resolution Quality",
    Dimension.QRP: "Query Response Precision",
    Dimension.SFC: "Selective Forgetting Capability",
}


class BenchmarkReporter:
    """Generates benchmark reports from CRI results.

    Supports console, JSON, Markdown, and comparison-table output.
    Works with ``BenchmarkResult`` and ``CRIResult`` models.

    Example::

        reporter = BenchmarkReporter()
        reporter.to_console(result)
        reporter.to_json(result, output_path=Path("report.json"))
        reporter.to_markdown(result, output_path=Path("report.md"))

        # Compare multiple systems
        md = reporter.generate_comparison_table([result_a, result_b])
    """

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def to_console(self, result: AnyResult) -> None:
        """Print a summary table to the terminal using Rich.

        The table uses the canonical format::

            System | CRI | PAS | DBU | MEI | TC | CRQ | QRP

        Scores are color-coded:

        - **≥ 9.0** bold green
        - **≥ 7.0** green
        - **≥ 5.0** yellow
        - **≥ 3.0** red
        - **< 3.0** bold red

        Dimension scores are on a 0–1 scale.

        Args:
            result: A ``BenchmarkResult`` or ``CRIResult``.
        """
        con = Console()
        info = self._extract_info(result)
        display_dims = self._display_dimensions(info)

        con.print()
        con.print("[bold cyan]═══ CRI Benchmark Report ═══[/bold cyan]")
        con.print(f"System: [bold]{info['system_name']}[/bold]")
        if info.get("dataset"):
            con.print(f"Dataset: {info['dataset']}")
        if info.get("timestamp"):
            con.print(f"Timestamp: {info['timestamp']}")
        con.print()

        table = Table(title="CRI Scores")
        table.add_column("System", style="cyan", no_wrap=True)
        table.add_column("CRI", justify="right", style="bold")
        for dim in display_dims:
            table.add_column(dim.value.upper(), justify="right")

        row: list[str | Text] = [info["system_name"]]

        # Composite CRI
        cri_val = info["composite_cri"]
        cri_style = self._score_style(cri_val)
        row.append(Text(f"{cri_val:.2f}", style=cri_style))

        # Per-dimension
        for dim in display_dims:
            score = info["dim_scores"].get(dim)
            if score is not None:
                style = self._score_style(score)
                row.append(Text(f"{score:.2f}", style=style))
            else:
                row.append(Text("—", style="dim"))

        table.add_row(*row)
        con.print(table)
        con.print()

    def to_json(
        self,
        result: AnyResult,
        output_path: Path | None = None,
    ) -> str:
        """Serialize the result to a JSON string.

        Works with ``BenchmarkResult`` and ``CRIResult``.

        Args:
            result: The benchmark result to serialize.
            output_path: If provided, write the JSON to this file.

        Returns:
            The JSON string.
        """
        data = result.model_dump(mode="json")

        json_str = json.dumps(data, indent=2, default=str)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_str, encoding="utf-8")
            logger.info("JSON report written to %s", output_path)

        return json_str

    def to_markdown(
        self,
        result: AnyResult,
        output_path: Path | None = None,
    ) -> str:
        """Generate a human-readable Markdown report.

        Includes:

        - Header with system name, dataset, and timestamp.
        - Summary score table with all dimensions and composite CRI.
        - Per-dimension breakdown with check counts and details.
        - Performance metrics section (new-model results only).
        - Score interpretation paragraph.
        - Generated-by footer.

        Args:
            result: The benchmark result.
            output_path: If provided, write the Markdown to this file.

        Returns:
            The Markdown string.
        """
        info = self._extract_info(result)
        display_dims = self._display_dimensions(info)
        lines: list[str] = []

        # Header
        lines.append("# CRI Benchmark Report")
        lines.append("")
        lines.append(f"**System:** {info['system_name']}")
        if info.get("dataset"):
            lines.append(f"**Dataset:** {info['dataset']}")
        if info.get("timestamp"):
            lines.append(f"**Timestamp:** {info['timestamp']}")
        lines.append(f"**Composite CRI Score:** {info['composite_cri']:.2f}")
        lines.append("")

        # Summary table
        lines.append("## Score Summary")
        lines.append("")
        lines.append("| Dimension | Code | Score | Checks |")
        lines.append("|-----------|------|------:|-------:|")
        for dim in display_dims:
            score = info["dim_scores"].get(dim)
            checks = info["dim_checks"].get(dim, 0)
            if score is not None:
                lines.append(
                    f"| {_DIMENSION_FULL_NAMES.get(dim, dim.name)} "
                    f"| {dim.value.upper()} "
                    f"| {score:.2f} "
                    f"| {checks} |"
                )
        lines.append("")

        # Per-dimension breakdown
        lines.append("## Dimension Breakdown")
        lines.append("")
        for dim in display_dims:
            score = info["dim_scores"].get(dim)
            if score is None:
                continue
            checks = info["dim_checks"].get(dim, 0)
            lines.append(f"### {_DIMENSION_FULL_NAMES.get(dim, dim.name)} ({dim.value.upper()})")
            lines.append("")
            lines.append(f"- **Score:** {score:.2f}")
            lines.append(f"- **Checks evaluated:** {checks}")

            # Extra detail from new model
            detail = info["dim_details"].get(dim)
            if detail and isinstance(detail, DimensionResult):
                lines.append(
                    f"- **Passed / Total:** {detail.passed_checks} / {detail.total_checks}"
                )
            lines.append("")

        # Performance section (new model only)
        perf = info.get("performance")
        if perf and isinstance(perf, PerformanceProfile):
            lines.append("## Performance Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|------:|")
            lines.append(f"| Ingest latency (avg) | {perf.ingest_latency_ms:.1f} ms |")
            lines.append(f"| Query latency (avg) | {perf.query_latency_avg_ms:.1f} ms |")
            lines.append(f"| Query latency (p95) | {perf.query_latency_p95_ms:.1f} ms |")
            lines.append(f"| Query latency (p99) | {perf.query_latency_p99_ms:.1f} ms |")
            lines.append(f"| Total facts stored | {perf.total_facts_stored} |")
            lines.append(f"| Judge API calls | {perf.judge_api_calls} |")
            if perf.judge_total_cost_estimate is not None:
                lines.append(f"| Est. judge cost | ${perf.judge_total_cost_estimate:.4f} |")
            lines.append("")

        # Interpretation
        lines.append("## Interpretation")
        lines.append("")
        lines.append(self._interpret_score(info["composite_cri"]))
        lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            f"*Generated by CRI Benchmark v{__version__} "
            f"at {datetime.now(timezone.utc).isoformat()}*"
        )

        md = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md, encoding="utf-8")
            logger.info("Markdown report written to %s", output_path)

        return md

    def generate_comparison_table(
        self,
        results: Sequence[AnyResult],
    ) -> str:
        """Generate a Markdown comparison table for multiple systems.

        Produces a table with one row per system.  Columns are determined
        by the union of all active dimensions across all results::

            | System | CRI | PAS | DBU | ... |

        Accepts any mix of ``CRIResult`` and ``BenchmarkResult`` objects.

        Args:
            results: Sequence of result objects to compare.

        Returns:
            A Markdown-formatted comparison table string.
        """
        # Determine columns: union of all active dimensions across results.
        all_infos = [self._extract_info(r) for r in results]
        display_dims = self._union_dimensions(all_infos)

        # Build header
        header_cells = ["System", "CRI"] + [d.value.upper() for d in display_dims]
        sep_cells = ["--------", "----:"] + ["----:" for _ in display_dims]

        lines: list[str] = []
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(sep_cells) + "|")

        for info in all_infos:
            name = info["system_name"]
            cri = f"{info['composite_cri']:.2f}"

            cells = [name, cri]
            for dim in display_dims:
                score = info["dim_scores"].get(dim)
                cells.append(f"{score:.2f}" if score is not None else "—")

            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_info(self, result: AnyResult) -> dict[str, Any]:
        """Normalize any result type into a uniform info dict.

        Returns a dict with keys:

        - ``system_name``: str
        - ``dataset``: str | None
        - ``timestamp``: str | None
        - ``composite_cri``: float
        - ``dim_scores``: dict[Dimension, float]
        - ``dim_checks``: dict[Dimension, int]
        - ``dim_details``: dict[Dimension, DimensionResult | None]
        - ``performance``: PerformanceProfile | None
        - ``active_dimensions``: list[Dimension] — dimensions that were evaluated
        """
        info: dict[str, Any] = {
            "system_name": "",
            "dataset": None,
            "timestamp": None,
            "composite_cri": 0.0,
            "dim_scores": {},
            "dim_checks": {},
            "dim_details": {},
            "performance": None,
            "active_dimensions": [],
        }

        # Build the field map dynamically from the Dimension enum.
        # Each Dimension's .value (e.g. "pas") is the CRIResult field name.
        dim_field_map: dict[Dimension, str] = {dim: dim.value for dim in Dimension}

        if isinstance(result, BenchmarkResult):
            cri = result.cri_result
            info["system_name"] = cri.system_name
            info["dataset"] = result.dataset_id
            info["timestamp"] = result.started_at
            info["composite_cri"] = cri.cri
            info["performance"] = result.performance_profile
            self._populate_dim_info(info, cri, dim_field_map)

        elif isinstance(result, CRIResult):
            info["system_name"] = result.system_name
            info["composite_cri"] = result.cri
            self._populate_dim_info(info, result, dim_field_map)

        return info

    @staticmethod
    def _populate_dim_info(
        info: dict[str, Any],
        cri: CRIResult,
        dim_field_map: dict[Dimension, str],
    ) -> None:
        """Fill dim_scores, dim_checks, dim_details, active_dimensions from a CRIResult."""
        active: list[Dimension] = []
        for dim, field_name in dim_field_map.items():
            # Check if this dimension was actually evaluated (present in details).
            detail = cri.details.get(field_name) or cri.details.get(dim.value.upper())
            score_val = getattr(cri, field_name, None)

            if detail:
                info["dim_details"][dim] = detail
                info["dim_checks"][dim] = detail.total_checks
                if score_val is not None:
                    info["dim_scores"][dim] = score_val
                active.append(dim)
            elif score_val is not None and score_val != 0.0:
                # Dimension has a non-zero score but no detail (e.g. legacy data).
                info["dim_scores"][dim] = score_val
                info["dim_checks"][dim] = 0
                active.append(dim)

        info["active_dimensions"] = active

    @staticmethod
    def _display_dimensions(info: dict[str, Any]) -> list[Dimension]:
        """Return the ordered list of dimensions to display for a single result.

        Uses ``active_dimensions`` from the info dict when available, otherwise
        falls back to dimensions that have scores in ``dim_scores``.  The
        returned list preserves the canonical ``_DIMENSION_ORDER``.
        """
        active: list[Dimension] = info.get("active_dimensions", [])
        if active:
            # Preserve canonical ordering.
            return [d for d in _DIMENSION_ORDER if d in active]
        # Fallback: any dimension with a score.
        return [d for d in _DIMENSION_ORDER if d in info.get("dim_scores", {})]

    @staticmethod
    def _union_dimensions(infos: list[dict[str, Any]]) -> list[Dimension]:
        """Return the union of active dimensions across multiple results, in canonical order."""
        seen: set[Dimension] = set()
        for info in infos:
            active: list[Dimension] = info.get("active_dimensions", [])
            if active:
                seen.update(active)
            else:
                seen.update(info.get("dim_scores", {}).keys())
        return [d for d in _DIMENSION_ORDER if d in seen]

    @staticmethod
    def _score_style(score: float) -> str:
        """Return a Rich style string based on the score value (0-1 scale).

        Args:
            score: The numeric score (0.0 to 1.0).

        Returns:
            A Rich style string (e.g., ``"bold green"``).
        """
        normalized = score * 10.0
        if normalized >= 9.0:
            return "bold green"
        if normalized >= 7.0:
            return "green"
        if normalized >= 5.0:
            return "yellow"
        if normalized >= 3.0:
            return "red"
        return "bold red"

    @staticmethod
    def _interpret_score(score: float) -> str:
        """Return a human-readable interpretation of the CRI score (0-1 scale).

        Args:
            score: The composite CRI score (0.0 to 1.0).

        Returns:
            A Markdown-formatted interpretation string.
        """
        normalized = score * 10.0
        if normalized >= 9.0:
            return (
                "**Exceptional** — The system demonstrates near-perfect contextual understanding."
            )
        if normalized >= 7.0:
            return "**Strong** — The system shows reliable memory with minor gaps."
        if normalized >= 5.0:
            return "**Moderate** — The system is functional but has notable weaknesses."
        if normalized >= 3.0:
            return "**Weak** — The system shows significant memory failures."
        return "**Poor** — The system demonstrates minimal or no effective memory."


__all__ = ["BenchmarkReporter"]
