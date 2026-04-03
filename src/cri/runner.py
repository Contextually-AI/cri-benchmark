"""Benchmark runner — orchestrates the full CRI evaluation pipeline.

The runner coordinates the end-to-end benchmark process:

1. Load a dataset (conversations + ground truth)
2. Initialize the memory adapter (from registry or dynamic import)
3. Wrap the adapter with performance instrumentation
4. Ingest conversation messages into the adapter
5. Run the scoring engine across all CRI dimensions
6. Collect performance metrics
7. Generate reports in the requested format

Provides both a **programmatic API** (:func:`run_benchmark`) and a
**CLI interface** (``cri`` command group) built on Click + Rich.

CLI Commands
~~~~~~~~~~~~

``cri run``
    Run the full benchmark pipeline with a given adapter and dataset.

``cri list-adapters``
    List all registered adapters in the built-in registry.

``cri list-datasets``
    List canonical datasets discovered in the datasets directory.

``cri validate-dataset``
    Validate a dataset directory's structure and content.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from cri.adapter import MemoryAdapter
from cri.datasets.loader import (
    list_canonical_datasets,
    load_dataset,
)
from cri.datasets.loader import (
    validate_dataset as validate_dataset_fn,
)
from cri.judge import BinaryJudge, LLMFactory, create_default_llm
from cri.models import (
    BenchmarkResult,
    ScoringConfig,
    ScoringProfile,
)
from cri.performance import PerformanceProfiler
from cri.reporter import BenchmarkReporter
from cri.scoring.engine import ScoringEngine

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

# Registry entry: (dotted_import_path, class_name, description, requires_extra)
_ADAPTER_ENTRIES: dict[str, tuple[str, str, str, str | None]] = {
    "no-memory": (
        "examples.adapters.no_memory_adapter",
        "NoMemoryAdapter",
        "Discards all input; returns nothing. Lower-bound baseline.",
        None,
    ),
    "full-context": (
        "examples.adapters.full_context_adapter",
        "FullContextAdapter",
        "Stores every user message; returns all on retrieve. Upper-bound recall baseline.",
        None,
    ),
    "rag": (
        "examples.adapters.rag_adapter",
        "RAGAdapter",
        "Simple ChromaDB vector-store RAG adapter. Requires 'pip install cri-benchmark[rag]'.",
        "rag",
    ),
    "upp": (
        "examples.adapters.upp_adapter",
        "UPPAdapter",
        "UPP (Universal Personalization Protocol) adapter. Bridges CRI to a UPP-compatible memory system.",
        None,
    ),
}


def _get_adapter_registry() -> dict[str, type]:
    """Lazily build the adapter registry, skipping entries with missing deps.

    Returns a dict mapping adapter name → adapter class. Entries whose
    dependencies are not installed are silently skipped.
    """
    registry: dict[str, type] = {}

    for name, (module_path, class_name, _desc, _extra) in _ADAPTER_ENTRIES.items():
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            registry[name] = cls
        except (ImportError, AttributeError) as exc:
            logger.debug(
                "Adapter '%s' not available (%s: %s) — skipping.",
                name,
                type(exc).__name__,
                exc,
            )

    return registry


def get_adapter_registry() -> dict[str, type]:
    """Return the built-in adapter registry.

    Returns:
        Dict mapping adapter name strings to adapter classes.
        Only includes adapters whose dependencies are installed.
    """
    return _get_adapter_registry()


# ---------------------------------------------------------------------------
# Dynamic adapter loading from dotted path
# ---------------------------------------------------------------------------


def load_adapter_class(dotted_path: str) -> type:
    """Import an adapter class from a dotted Python path.

    Supports two path formats:

    - ``module.path:ClassName`` — explicit class within a module.
    - ``module.path.ClassName`` — dotted attribute path (last component
      is the class name).

    Args:
        dotted_path: A string like ``"mypackage.adapters:MyAdapter"``
            or ``"mypackage.adapters.MyAdapter"``.

    Returns:
        The adapter class.

    Raises:
        ValueError: If the path format is invalid or the class cannot
            be found.
        ImportError: If the module cannot be imported.
    """
    if ":" in dotted_path:
        module_path, class_name = dotted_path.rsplit(":", 1)
    elif "." in dotted_path:
        module_path, class_name = dotted_path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid adapter path '{dotted_path}'. Expected 'module.path:ClassName' or 'module.path.ClassName'.")

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Cannot import module '{module_path}': {exc}") from exc

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ValueError(f"Module '{module_path}' has no attribute '{class_name}'.")

    if not isinstance(cls, type):
        raise ValueError(f"'{dotted_path}' resolved to {type(cls).__name__}, not a class.")

    return cls


def resolve_adapter(name_or_path: str) -> type:
    """Resolve an adapter name or dotted path to an adapter class.

    First checks the built-in registry. If not found, tries dynamic import.

    Args:
        name_or_path: A registry name like ``"no-memory"`` or a dotted
            path like ``"mypackage.adapters:MyAdapter"``.

    Returns:
        The adapter class.

    Raises:
        click.BadParameter: If the adapter cannot be resolved.
    """
    # Check registry first
    registry = get_adapter_registry()
    if name_or_path in registry:
        return registry[name_or_path]

    # Check all known adapter names (including unavailable ones)
    if name_or_path in _ADAPTER_ENTRIES:
        _mod, _cls, _desc, extra = _ADAPTER_ENTRIES[name_or_path]
        hint = ""
        if extra:
            hint = f" Install with: pip install cri-benchmark[{extra}]"
        raise click.BadParameter(f"Adapter '{name_or_path}' is registered but its dependencies are not installed.{hint}")

    # Try dynamic import
    try:
        return load_adapter_class(name_or_path)
    except (ValueError, ImportError) as exc:
        available = ", ".join(sorted(_ADAPTER_ENTRIES.keys()))
        raise click.BadParameter(
            f"Cannot resolve adapter '{name_or_path}'. Built-in adapters: {available}. Or provide a dotted path like 'module:ClassName'. Error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline execution (used by CLI)
# ---------------------------------------------------------------------------


async def run_benchmark(
    adapter_name: str,
    dataset_path: str,
    llm_factory: LLMFactory | None = None,
    judge_runs: int = 3,
    output_dir: str | None = None,
    output_format: str = "console",
    verbose: bool = False,
    profile: str | None = None,
    dimensions: list[str] | None = None,
    scale_test: bool = False,
    limit: int | None = None,
) -> BenchmarkResult:
    """Execute the full CRI benchmark pipeline.

    This is the high-level entry point that the CLI ``run`` command uses.
    It orchestrates:

    1. Dataset loading and validation
    2. Adapter instantiation
    3. Performance profiling setup
    4. Message ingestion
    5. Scoring engine evaluation
    6. (Optional) SSI scale-sensitivity test
    7. Performance profile collection
    8. Report generation

    Args:
        adapter_name: Registry name or dotted path for the adapter.
        dataset_path: Path to a dataset directory.
        llm_factory: A callable ``(temperature, max_tokens) -> BaseChatModel``
            for creating the judge LLM.  Defaults to :func:`create_default_llm`.
        judge_runs: Number of majority-vote runs per judgment.
        output_dir: Directory to write result files (optional).
        output_format: One of 'console', 'json', 'markdown'.
        verbose: Whether to show detailed progress.
        profile: Scoring profile name ('core', 'extended', 'full').
        dimensions: Explicit list of dimension codes (mutually exclusive
            with ``profile``).
        scale_test: Whether to run the SSI scale-sensitivity test.
        limit: Maximum number of messages to ingest (for quick smoke tests).

    Returns:
        The complete BenchmarkResult.
    """
    # 1. Load and validate dataset
    ds_path = Path(dataset_path)
    dataset = load_dataset(ds_path)
    if limit is not None:
        dataset.messages = dataset.messages[:limit]
    else:
        errors = validate_dataset_fn(dataset)
        if errors:
            console.print("[red]Dataset validation errors:[/red]")
            for err in errors:
                console.print(f"  [red]✗[/red] {err}")
            raise click.Abort()

    if verbose:
        console.print(f"[green]✓[/green] Dataset loaded: {len(dataset.messages)} messages, {len(dataset.ground_truth.final_profile)} profile dimensions")

    # 2. Build scoring configuration
    config = _build_scoring_config(profile=profile, dimensions=dimensions, scale_test=scale_test)
    run_scale_test = config.scale_test or scale_test

    if verbose:
        console.print(
            f"[green]✓[/green] Profile: [bold]{config.profile.value}[/bold]  "
            f"Dimensions: {', '.join(config.enabled_dimensions)}" + ("  [bold]+SSI[/bold]" if run_scale_test else "")
        )

    # 3. Resolve and instantiate adapter
    adapter_cls = resolve_adapter(adapter_name)
    adapter = adapter_cls()

    if not isinstance(adapter, MemoryAdapter):
        raise TypeError(
            f"Adapter '{adapter_name}' ({type(adapter).__name__}) does not "
            f"satisfy the MemoryAdapter protocol. "
            f"Required methods: ingest(), retrieve(), get_events()."
        )

    # 4. Create judge
    factory = llm_factory or create_default_llm
    judge = BinaryJudge(
        llm_factory=factory,
        num_runs=judge_runs,
    )

    # 5. Create scoring engine
    engine = ScoringEngine(
        ground_truth=dataset.ground_truth,
        judge=judge,
        config=config,
    )

    # 6. Wrap adapter with profiler
    profiler = PerformanceProfiler()
    instrumented = profiler.wrap_adapter(adapter)

    # 7. Ingest messages
    if verbose:
        console.print("[bold]Ingesting messages...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("Ingesting messages", total=len(dataset.messages))
        # Ingest in a single batch
        instrumented.ingest(dataset.messages)
        progress.update(task, completed=len(dataset.messages))

    if verbose:
        console.print(f"[green]✓[/green] Ingested {len(dataset.messages)} messages")

    # 8. Run scoring engine
    if verbose:
        console.print("[bold]Running evaluation across all dimensions...[/bold]")

    result = await engine.run(
        adapter=instrumented,
        system_name=adapter_name,
    )

    # 9. (Optional) Run SSI scale-sensitivity test
    if run_scale_test:
        if verbose:
            console.print("[bold]Running SSI scale-sensitivity test...[/bold]")

        from cri.scoring.ssi import compute_ssi

        ssi_result = await compute_ssi(
            adapter_factory=lambda: resolve_adapter(adapter_name)(),
            messages=dataset.messages,
            ground_truth=dataset.ground_truth,
            judge_factory=lambda: BinaryJudge(llm_factory=factory, num_runs=judge_runs),
            config=config,
        )
        # Attach SSI to the result details (reported separately from CRI composite).
        result.cri_result.details["SSI"] = ssi_result

        if verbose:
            console.print(f"[green]✓[/green] SSI score: {ssi_result.score:.4f}")

    # 10. Collect performance profile
    profiler.judge_api_calls = len(judge.get_log())
    perf_profile = profiler.get_profile()
    result.performance_profile = perf_profile
    result.dataset_id = dataset.metadata.dataset_id

    # 11. Generate report
    reporter = BenchmarkReporter()

    if output_format == "console":
        reporter.to_console(result)
    elif output_format == "json":
        json_str = reporter.to_json(result)
        if not output_dir:
            console.print(json_str)
    elif output_format == "markdown":
        md_str = reporter.to_markdown(result)
        if not output_dir:
            console.print(md_str)

    # 12. Write to output directory if specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Always write JSON result
        json_path = out_path / "result.json"
        reporter.to_json(result, output_path=json_path)

        # Write markdown if requested
        if output_format == "markdown":
            md_path = out_path / "report.md"
            reporter.to_markdown(result, output_path=md_path)

        # Write judge log
        judge.export_log(out_path / "judge_log.json")

        if verbose:
            console.print(f"[green]✓[/green] Results written to {out_path}")

    return result


def _build_scoring_config(
    profile: str | None = None,
    dimensions: list[str] | None = None,
    scale_test: bool = False,
) -> ScoringConfig:
    """Build a ScoringConfig from CLI options.

    Args:
        profile: Profile name string (e.g. 'core', 'extended', 'full').
        dimensions: Explicit list of dimension codes.
        scale_test: Whether to force-enable the SSI scale test.

    Returns:
        A configured :class:`ScoringConfig`.
    """
    if dimensions:
        config = ScoringConfig.from_dimensions(dimensions)
        if scale_test:
            config.scale_test = True
        return config

    if profile:
        scoring_profile = ScoringProfile(profile)
        config = ScoringConfig.from_profile(scoring_profile)
        if scale_test:
            config.scale_test = True
        return config

    # Default: core profile
    config = ScoringConfig.from_profile(ScoringProfile.CORE)
    if scale_test:
        config.scale_test = True
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="cri-benchmark")
def main() -> None:
    """CRI Benchmark — Contextual Resonance Index.

    An open-source benchmark for evaluating AI long-term memory systems.
    """


@main.command()
@click.option(
    "--adapter",
    required=True,
    help="Adapter name from registry (e.g., 'no-memory') or dotted path (e.g., 'module:Class').",
)
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to a dataset directory containing conversations.jsonl and ground_truth.json.",
)
@click.option(
    "--judge-runs",
    default=3,
    show_default=True,
    type=int,
    help="Number of judge invocations per evaluation check (majority vote).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for result files.",
)
@click.option(
    "--format",
    "fmt",
    default="console",
    type=click.Choice(["console", "json", "markdown"]),
    show_default=True,
    help="Output format for the benchmark report.",
)
@click.option("--verbose", is_flag=True, help="Show detailed progress.")
@click.option(
    "--profile",
    "profile",
    default=None,
    type=click.Choice(["core", "extended", "full"]),
    help=("Scoring profile: 'core' (PAS, DBU, MEI, TC, CRQ, QRP), 'extended' (legacy alias for core), or 'full' (core + SSI scale test)."),
)
@click.option(
    "--dimensions",
    "dimensions_str",
    default=None,
    type=str,
    help=("Comma-separated list of dimension codes to evaluate (e.g., 'PAS,DBU,MEI'). Mutually exclusive with --profile."),
)
@click.option(
    "--scale-test",
    is_flag=True,
    default=False,
    help="Run the SSI scale-sensitivity test in addition to dimension scoring.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Maximum number of messages to ingest (useful for quick smoke tests).",
)
def run(
    adapter: str,
    dataset: str,
    judge_runs: int,
    output: str | None,
    fmt: str,
    verbose: bool,
    profile: str | None,
    dimensions_str: str | None,
    scale_test: bool,
    limit: int | None,
) -> None:
    """Run the CRI benchmark with a given adapter and dataset."""
    # Validate mutual exclusivity of --profile and --dimensions.
    if profile and dimensions_str:
        console.print("[red]Error:[/red] --profile and --dimensions are mutually exclusive.")
        raise SystemExit(1)

    dimensions = [d.strip().upper() for d in dimensions_str.split(",")] if dimensions_str else None

    console.print()
    console.print("[bold cyan]═══ CRI Benchmark ═══[/bold cyan]")
    console.print(f"  Adapter:     [bold]{adapter}[/bold]")
    console.print(f"  Dataset:     {dataset}")
    console.print(f"  Judge runs:  {judge_runs}")
    console.print(f"  Format:      {fmt}")
    if profile:
        console.print(f"  Profile:     {profile}")
    if dimensions:
        console.print(f"  Dimensions:  {', '.join(dimensions)}")
    if scale_test:
        console.print("  Scale test:  enabled")
    if limit:
        console.print(f"  Limit:       {limit} messages")
    if output:
        console.print(f"  Output dir:  {output}")
    console.print()

    try:
        asyncio.run(
            run_benchmark(
                adapter_name=adapter,
                dataset_path=dataset,
                judge_runs=judge_runs,
                output_dir=output,
                output_format=fmt,
                verbose=verbose,
                profile=profile,
                dimensions=dimensions,
                scale_test=scale_test,
                limit=limit,
            )
        )
    except click.BadParameter as exc:
        console.print(f"[red]Error:[/red] {exc.format_message()}")
        raise SystemExit(1) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        logger.exception("Benchmark run failed")
        raise SystemExit(1) from exc


@main.command(name="list-adapters")
def list_adapters() -> None:
    """List all registered benchmark adapters."""
    console.print()
    console.print("[bold cyan]═══ Registered Adapters ═══[/bold cyan]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Available", justify="center")

    registry = get_adapter_registry()

    for name in sorted(_ADAPTER_ENTRIES.keys()):
        _mod, _cls, desc, _extra = _ADAPTER_ENTRIES[name]
        available = name in registry
        status = "[green]✓[/green]" if available else "[red]✗[/red]"
        table.add_row(name, desc, status)

    console.print(table)
    console.print()
    console.print("[dim]You can also pass a dotted Python path as --adapter, e.g., 'mypackage.module:MyAdapterClass'.[/dim]")
    console.print()


@main.command(name="list-datasets")
def list_datasets() -> None:
    """List available canonical benchmark datasets."""
    datasets = list_canonical_datasets()

    console.print()
    console.print("[bold cyan]═══ Canonical Datasets ═══[/bold cyan]")
    console.print()

    if not datasets:
        console.print("[yellow]No canonical datasets found.[/yellow]")
        console.print("[dim]Place datasets in datasets/canonical/ or use --dataset <path> to specify a custom dataset.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Messages", justify="right")
    table.add_column("Ground Truth", justify="center")
    table.add_column("Path")

    for ds in datasets:
        msg_count = str(ds.message_count) if ds.message_count is not None else "—"
        gt = "[green]✓[/green]" if ds.has_ground_truth else "[red]✗[/red]"
        table.add_row(ds.name, msg_count, gt, str(ds.path))

    console.print(table)
    console.print()


@main.command(name="validate-dataset")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def validate_dataset_cmd(path: str) -> None:
    """Validate a dataset directory's structure and content.

    PATH is the path to a dataset directory containing conversations.jsonl
    and ground_truth.json.
    """
    console.print()
    console.print(f"[bold]Validating dataset:[/bold] {path}")
    console.print()

    try:
        dataset = load_dataset(Path(path))
    except FileNotFoundError as exc:
        console.print(f"[red]✗ Cannot load dataset: {exc}[/red]")
        raise SystemExit(1) from exc
    except ValueError as exc:
        console.print(f"[red]✗ Invalid dataset format: {exc}[/red]")
        raise SystemExit(1) from exc

    errors = validate_dataset_fn(dataset)

    if errors:
        console.print(f"[red]✗ Validation failed with {len(errors)} error(s):[/red]")
        for err in errors:
            console.print(f"  [red]•[/red] {err}")
        raise SystemExit(1)
    else:
        console.print("[green]✓ Dataset is valid.[/green]")
        console.print(f"  Messages:           {len(dataset.messages)}")
        console.print(f"  Profile dimensions: {len(dataset.ground_truth.final_profile)}")
        console.print(f"  Belief changes:     {len(dataset.ground_truth.changes)}")
        console.print(f"  Conflicts:          {len(dataset.ground_truth.conflicts)}")
        console.print(f"  Temporal facts:     {len(dataset.ground_truth.temporal_facts)}")


if __name__ == "__main__":
    main()
