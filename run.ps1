#Requires -Version 5.1
<#
.SYNOPSIS
    CRI Benchmark — Docker Runner (Windows)

.DESCRIPTION
    Builds the CRI Benchmark Docker image and runs one or more adapters
    against one or more datasets, collecting results and generating a
    Markdown summary.

.EXAMPLE
    .\run.ps1                                                   # All adapters, all datasets
    .\run.ps1 -Limit 50                                         # Smoke test (50 messages)
    .\run.ps1 -Adapter rag                                      # Single adapter, all datasets
    .\run.ps1 -Adapter rag -Dataset src/cri/datasets/persona-1-base
    .\run.ps1 -Adapter rag,full-context                         # Multiple adapters
    .\run.ps1 -Format markdown                                   # Custom format
    .\run.ps1 -Quiet                                              # Suppress verbose output
#>

param(
    [string[]]$Adapter,
    [string[]]$Dataset,
    [int]$Limit,
    [ValidateSet("console","json","markdown")]
    [string]$Format,
    [ValidateSet("core","extended","full")]
    [string]$Profile,
    [string]$Dimensions,
    [int]$JudgeRuns,
    [string]$Output,
    [switch]$Quiet,
    [switch]$ScaleTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Configuration ─────────────────────────────────────────────────
$Image           = "cri-benchmark"
$RunId           = Get-Date -Format "yyyy_MM_dd_HH_mm_ss"
$ResultsDir      = Join-Path (Join-Path $PSScriptRoot "logs") $RunId
$DefaultAdapters = @("no-memory", "full-context", "rag")
$AuthTokenPath   = Join-Path (Split-Path $PSScriptRoot -Parent) ".auth_token"
$ScriptStart     = Get-Date

# ── Helpers ───────────────────────────────────────────────────────
function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $Message"
    if ($script:LogFile) {
        "[$ts] $Message" | Out-File -Append -FilePath $script:LogFile -Encoding utf8
    }
}

function Get-Elapsed {
    param([datetime]$Start)
    $diff = (Get-Date) - $Start
    return "{0}m{1:D2}s" -f [int]$diff.TotalMinutes, $diff.Seconds
}

# ── Resolve adapters ──────────────────────────────────────────────
if (-not $Adapter -or $Adapter.Count -eq 0) {
    $Adapter = $DefaultAdapters
}

# ── Setup logging ─────────────────────────────────────────────────
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
$script:LogFile = Join-Path $ResultsDir "run.log"

# ── Validate auth token ──────────────────────────────────────────
if (-not (Test-Path $AuthTokenPath)) {
    Write-Log "ERROR: Auth token not found at $AuthTokenPath"
    Write-Log "Place your Anthropic OAuth token in that file."
    exit 1
}

# ── Build ─────────────────────────────────────────────────────────
$StepStart = Get-Date
Write-Log "Building CRI Benchmark container..."
docker build -t $Image .
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Docker build failed."
    exit 1
}
Write-Log "Build complete ($(Get-Elapsed $StepStart))"

# ── Discover datasets if none specified ───────────────────────────
if (-not $Dataset -or $Dataset.Count -eq 0) {
    $StepStart = Get-Date
    Write-Log "Discovering datasets..."

    # Discover dataset paths inside the container via Python (Rich table truncates paths)
    $discovered = @(docker run --rm --entrypoint python `
        $Image -c `
        "from cri.datasets.loader import list_datasets; [print(ds.path) for ds in list_datasets()]" `
        2>$null) | Where-Object { $_ } | Sort-Object

    if (-not $discovered -or $discovered.Count -eq 0) {
        Write-Log "ERROR: No datasets found. Pass -Dataset explicitly."
        exit 1
    }

    $Dataset = @($discovered)
    Write-Log "Found $($Dataset.Count) dataset(s) ($(Get-Elapsed $StepStart))"
}

# ── Build common flags ────────────────────────────────────────────
$CommonFlags = @()

if ($Limit -gt 0)    { $CommonFlags += "--limit", "$Limit" }
if (-not $Format)    { $Format   = "json" }
if (-not $Profile)   { $Profile  = "extended" }

$CommonFlags += "--format",  $Format
$CommonFlags += "--profile", $Profile

if (-not $Quiet) { $CommonFlags += "--verbose" }
if ($ScaleTest) { $CommonFlags += "--scale-test" }
if ($Dimensions)  { $CommonFlags += "--dimensions", $Dimensions }
if ($JudgeRuns -gt 0) { $CommonFlags += "--judge-runs", "$JudgeRuns" }

# ── Convert Windows paths to Docker-friendly forward slashes ──────
$AuthTokenDocker  = $AuthTokenPath -replace '\\', '/'
$ResultsDirDocker = $ResultsDir -replace '\\', '/'

# ── Run ───────────────────────────────────────────────────────────
Write-Log "Adapters: $($Adapter -join ', ')"
Write-Log "Datasets: $($Dataset -join ', ')"
if ($Limit -gt 0) { Write-Log "Limit: $Limit messages per dataset" }
Write-Log "Log file: $($script:LogFile)"
Write-Host ""

foreach ($ds in $Dataset) {
    foreach ($adp in $Adapter) {
        $StepStart = Get-Date
        Write-Host "──────────────────────────────────────────────"
        Write-Log "Starting adapter=$adp dataset=$ds"
        Write-Host "──────────────────────────────────────────────"

        # Auto-generate --output if user didn't provide one
        $OutputFlag = @()
        if (-not $Output) {
            $dsBaseName = Split-Path $ds -Leaf
            $OutputFlag = @("--output", "/app/results/${adp}_${dsBaseName}")
        } else {
            $OutputFlag = @("--output", $Output)
        }

        $dockerArgs = @(
            "run", "--rm",
            "-v", "${ResultsDirDocker}:/app/results",
            "-v", "${AuthTokenDocker}:/.auth_token:ro",
            $Image, "run",
            "--adapter", $adp,
            "--dataset", $ds
        ) + $OutputFlag + $CommonFlags

        docker @dockerArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Log "WARNING: Adapter '$adp' failed on '$ds' (skipping)"
        }

        Write-Log "Finished adapter=$adp dataset=$ds ($(Get-Elapsed $StepStart))"
        Write-Host ""
    }
}

$TotalElapsed = Get-Elapsed $ScriptStart
Write-Log "All runs complete (total: $TotalElapsed)"
Write-Log "Results in $ResultsDir"

# ── Generate summary ──────────────────────────────────────────────
$SummaryPath = Join-Path $ResultsDir "summary.md"
Write-Log "Generating summary..."

$summaryScript = Join-Path $ResultsDir "_generate_summary.py"
@'
import json, os, sys

results_dir = sys.argv[1]
summary_path = sys.argv[2]
dims = ['PAS','DBU','MEI','TC','CRQ','QRP']

# Collect results
rows = []
for name in sorted(os.listdir(results_dir)):
    rpath = os.path.join(results_dir, name, 'result.json')
    if not os.path.isfile(rpath):
        continue
    with open(rpath) as f:
        data = json.load(f)
    cri = data['cri_result']
    parts = name.split('_', 1)
    adapter = parts[0] if len(parts) > 0 else name
    dataset = parts[1] if len(parts) > 1 else ''
    rows.append({
        'adapter': cri.get('system_name', adapter),
        'dataset': data.get('dataset_id', dataset),
        'cri': cri.get('cri', 0),
        **{d.lower(): cri.get(d.lower(), None) for d in dims}
    })

if not rows:
    print('No results found.', file=sys.stderr)
    sys.exit(0)

datasets = sorted(set(r['dataset'] for r in rows))
adapters = sorted(set(r['adapter'] for r in rows))

lines = ['# CRI Benchmark Summary', '']

# Overview table
lines.append('## Composite CRI')
lines.append('')
header = '| Adapter |' + '|'.join(f' {ds} ' for ds in datasets) + '|'
sep = '|---------|' + '|'.join(':------:' for _ in datasets) + '|'
lines.append(header)
lines.append(sep)
for adapter in adapters:
    cells = []
    for ds in datasets:
        match = [r for r in rows if r['adapter'] == adapter and r['dataset'] == ds]
        if match:
            cells.append(f' **{match[0]["cri"]:.4f}** ')
        else:
            cells.append(' — ')
    lines.append(f'| {adapter} |' + '|'.join(cells) + '|')
lines.append('')

# Per-dimension tables
for dim in dims:
    dim_lower = dim.lower()
    lines.append(f'## {dim}')
    lines.append('')
    lines.append(header)
    lines.append(sep)
    for adapter in adapters:
        cells = []
        for ds in datasets:
            match = [r for r in rows if r['adapter'] == adapter and r['dataset'] == ds]
            if match and match[0].get(dim_lower) is not None:
                val = match[0][dim_lower]
                cells.append(f' {val:.4f} ')
            else:
                cells.append(' — ')
        lines.append(f'| {adapter} |' + '|'.join(cells) + '|')
    lines.append('')

with open(summary_path, 'w') as f:
    f.write('\n'.join(lines))
'@ | Set-Content -Path $summaryScript -Encoding utf8

python $summaryScript "$ResultsDir" "$SummaryPath"
if ($LASTEXITCODE -ne 0) {
    Write-Log "WARNING: Summary generation failed (python may not be in PATH)"
} else {
    Write-Log "Summary written to $SummaryPath"
}
Remove-Item -Path $summaryScript -ErrorAction SilentlyContinue
