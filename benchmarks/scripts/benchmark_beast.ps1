<#
.SYNOPSIS
    iTaK Torch Benchmark Suite - Beast (Windows)
.DESCRIPTION
    Runs the full benchmark matrix on Beast workstation.
    Tests iTaK Torch and Ollama across GPU and CPU backends.
    Captures both throughput (tok/s) and system resources (VRAM, GPU power, temp, RAM) per engine.
.PARAMETER Category
    Which benchmark category to run: boot, ttft, matrix, throughput, resources, threads, batching, unit, all
    Note: 'throughput' and 'resources' are both aliases for 'matrix' (the unified benchmark).
.PARAMETER Model
    GGUF model path for iTaK Torch benchmarks.
.PARAMETER HfModel
    HuggingFace model name (reserved for future use).
.PARAMETER OllamaModel
    Ollama model tag for comparison benchmarks.
.PARAMETER Iterations
    Number of iterations for throughput tests (default: 3).
.PARAMETER OutputDir
    Directory to save results.
#>
param(
    [ValidateSet("boot", "ttft", "matrix", "throughput", "resources", "threads", "batching", "unit", "all")]
    [string]$Category = "all",

    [string]$Model = "E:\.agent\iTaK Eco\Torch\models\qwen2.5-0.5b-instruct-q4_k_m.gguf",
    [string]$HfModel = "Qwen/Qwen2.5-0.5B-Instruct",
    [string]$OllamaModel = "qwen2.5:0.5b",
    [int]$Iterations = 3,
    [string]$OutputDir = "E:\.agent\iTaK Eco\Torch\benchmarks\results",
    [string]$ProjectDir = "E:\.agent\iTaK Eco\Torch"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$reportFile = Join-Path $OutputDir "benchmark_${timestamp}_beast.md"

# Ensure output directory exists.
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Standardized test prompt.
$testPrompt = "Explain the concept of transformer attention mechanisms in neural networks. Include details about query, key, and value matrices."

function Write-Report {
    param([string]$Content)
    Add-Content -Path $reportFile -Value $Content
}

function Get-SystemInfo {
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
    $gpu = $null
    try {
        $gpuRaw = cmd /c 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>$null
        $memRaw = cmd /c 'nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits' 2>$null
        if ($gpuRaw -and $memRaw) {
            $gpu = "$($gpuRaw.Trim()), $($memRaw.Trim()) MiB"
        }
    }
    catch {}
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)

    return @{
        CPU     = $cpu.Name
        Cores   = $cpu.NumberOfCores
        Threads = $cpu.NumberOfLogicalProcessors
        RAM     = "${ram}GB"
        GPU     = if ($gpu) { $gpu.Trim() } else { "None" }
    }
}

# ============================================================
# GPU & SYSTEM SNAPSHOT HELPERS
# ============================================================

function Get-GpuSnapshot {
    <# Returns a hashtable with VRAM, GPU util, temp, and power draw. #>
    try {
        $raw = cmd /c 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits' 2>$null
        if ($raw) {
            $parts = $raw.Trim() -split ",\s*"
            return @{
                GpuUtil     = [int]$parts[0]
                VramUsedMB  = [int]$parts[1]
                VramTotalMB = [int]$parts[2]
                TempC       = [int]$parts[3]
                PowerW      = [math]::Round([double]$parts[4], 1)
            }
        }
    }
    catch {}
    return $null
}

function Get-SystemRamGB {
    [math]::Round((Get-Process | Measure-Object WorkingSet64 -Sum).Sum / 1GB, 2)
}

function Get-CpuUsagePercent {
    <# Quick CPU usage sample over 1 second. #>
    try {
        $sample = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue
        return [math]::Round($sample, 1)
    }
    catch { return 0 }
}

# ============================================================
# HELPER: Write a standard engine result block
# ============================================================
function Write-EngineResult {
    param(
        [string]$EngineName,
        [double]$MeanTokS,
        [string]$VsOllamaStr,
        [string]$RunsStr,
        [double]$MinTokS,
        [double]$MaxTokS,
        [double]$RamBefore,
        [double]$RamAfter,
        $GpuBefore,
        $GpuAfter,
        [string]$RawOutput = "",
        [double]$CpuPercent = 0
    )

    Write-Report "### $EngineName"
    Write-Report ""
    Write-Report "| Metric | Value |"
    Write-Report "|--------|-------|"
    Write-Report "| **Mean tok/s** | **$MeanTokS** |"
    if ($VsOllamaStr) {
        Write-Report "| vs Ollama (GPU) | $VsOllamaStr |"
    }
    Write-Report "| Runs | $RunsStr |"
    Write-Report "| Min / Max | $MinTokS / $MaxTokS |"
    if ($CpuPercent -gt 0) {
        Write-Report "| CPU Usage | ${CpuPercent}% |"
    }
    Write-Report "| System RAM | ${RamBefore} -> ${RamAfter} GB |"
    if ($GpuBefore -and $GpuAfter) {
        Write-Report "| VRAM | $($GpuBefore.VramUsedMB) -> $($GpuAfter.VramUsedMB) MB |"
        Write-Report "| GPU Temp | $($GpuBefore.TempC) -> $($GpuAfter.TempC) C |"
        Write-Report "| GPU Power | $($GpuBefore.PowerW) -> $($GpuAfter.PowerW) W |"
    }
    Write-Report ""

    if ($RawOutput) {
        Write-Report "<details><summary>Raw Output</summary>"
        Write-Report ""
        Write-Report '```'
        Write-Report $RawOutput
        Write-Report '```'
        Write-Report ""
        Write-Report "</details>"
        Write-Report ""
    }
}

# ============================================================
# OLLAMA INFERENCE RUNNER
# ============================================================

function Run-OllamaInference {
    param(
        [string]$OModel,
        [string]$Prompt,
        [int]$MaxTokens = 256,
        [switch]$CpuOnly
    )

    $startTime = Get-Date

    if ($CpuOnly) {
        # Use the Ollama REST API with num_gpu=0 to force true CPU-only inference.
        # The CLI env-var approach doesn't work because Ollama runs as a service.
        $body = @{
            model   = $OModel
            prompt  = $Prompt
            stream  = $false
            options = @{
                num_gpu     = 0
                num_predict = $MaxTokens
            }
        } | ConvertTo-Json -Depth 3

        try {
            $response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
            $endTime = Get-Date
            $elapsed = ($endTime - $startTime).TotalSeconds

            # Ollama API returns eval_count and eval_duration (nanoseconds).
            $evalCount = if ($response.eval_count) { [int]$response.eval_count } else { 0 }
            $evalDurationNs = if ($response.eval_duration) { [double]$response.eval_duration } else { 0 }
            $tokPerSec = 0
            if ($evalDurationNs -gt 0) {
                $tokPerSec = [math]::Round($evalCount / ($evalDurationNs / 1e9), 1)
            }

            return @{
                TokPerSec = $tokPerSec
                Tokens    = $evalCount
                Elapsed   = $elapsed
                Output    = $response.response
            }
        }
        catch {
            $endTime = Get-Date
            return @{
                TokPerSec = 0
                Tokens    = 0
                Elapsed   = ($endTime - $startTime).TotalSeconds
                Output    = "ERROR: $($_.Exception.Message)"
            }
        }
    }
    else {
        # GPU mode: use the CLI with verbose output.
        $tempFile = [System.IO.Path]::GetTempFileName()
        $Prompt | Set-Content $tempFile -NoNewline
        $result = cmd /c "type `"$tempFile`" | ollama run $OModel --verbose 2>&1"
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
        $endTime = Get-Date

        $elapsed = ($endTime - $startTime).TotalSeconds

        # Parse tok/s from ollama verbose output.
        $tokPerSec = 0
        $evalCount = 0
        foreach ($line in $result) {
            $lineStr = "$line"
            if ($lineStr -match "eval rate:\s*([\d.]+)\s*tokens") {
                $tokPerSec = [double]$Matches[1]
            }
            if ($lineStr -match "eval count:\s*(\d+)") {
                $evalCount = [int]$Matches[1]
            }
        }

        return @{
            TokPerSec = $tokPerSec
            Tokens    = $evalCount
            Elapsed   = $elapsed
            Output    = ($result | Out-String)
        }
    }
}


# ============================================================
# REPORT HEADER
# ============================================================
function Write-Header {
    $sysInfo = Get-SystemInfo
    Write-Report "# iTaK Torch Benchmark Report - Beast"
    Write-Report ""
    Write-Report "**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Report "**Machine:** Beast (Windows)"
    Write-Report "**CPU:** $($sysInfo.CPU)"
    Write-Report "**Cores/Threads:** $($sysInfo.Cores)/$($sysInfo.Threads)"
    Write-Report "**RAM:** $($sysInfo.RAM)"
    Write-Report "**GPU:** $($sysInfo.GPU)"
    Write-Report "**Model (GGUF):** $(Split-Path $Model -Leaf)"
    Write-Report "**Ollama Model:** $OllamaModel"
    Write-Report ""
    Write-Report "---"
    Write-Report ""
}

# ============================================================
# CAT 7: UNIT TESTS (run first to validate correctness)
# ============================================================
function Run-UnitTests {
    Write-Host "`n=== Category 7: Unit Tests ===" -ForegroundColor Cyan

    Write-Report "## Category 7: Unit Tests"
    Write-Report ""

    $testResult = & { Set-Location $ProjectDir; go test ./pkg/torch/ -v -count=1 -short 2>&1 }
    $passed = ($testResult | Select-String "PASS" | Measure-Object).Count
    $failed = ($testResult | Select-String "FAIL" | Measure-Object).Count

    Write-Report "| Package | Passed | Failed |"
    Write-Report "|---------|--------|--------|"
    Write-Report "| pkg/torch/ | $passed | $failed |"

    $configResult = & { Set-Location $ProjectDir; go test ./pkg/config/... -v -count=1 -short 2>&1 }
    $cPassed = ($configResult | Select-String "PASS" | Measure-Object).Count
    $cFailed = ($configResult | Select-String "FAIL" | Measure-Object).Count

    Write-Report "| pkg/config/ | $cPassed | $cFailed |"
    Write-Report ""

    if ($failed -gt 0 -or $cFailed -gt 0) {
        Write-Report "> [!CAUTION]"
        Write-Report "> Unit tests failed. Benchmark results may be unreliable."
        Write-Report ""
        Write-Host "WARNING: Unit tests failed!" -ForegroundColor Red
    }
    else {
        Write-Report "> [!NOTE]"
        Write-Report "> All unit tests passed."
        Write-Report ""
        Write-Host "All unit tests passed." -ForegroundColor Green
    }
}

# ============================================================
# ENGINE MATRIX: UNIFIED THROUGHPUT + RESOURCE MONITORING
# Every engine gets: tok/s, VRAM, GPU temp, GPU power, RAM, CPU%.
# GPU engines tested first, then CPU-only engines.
# ============================================================
function Run-EngineMatrix {
    Write-Host "`n=== Engine Performance Matrix (Throughput + Resources) ===" -ForegroundColor Cyan

    Write-Report "## Engine Performance Matrix"
    Write-Report ""
    Write-Report "Each engine is tested with **$Iterations** iterations. System resources (VRAM, GPU temperature,"
    Write-Report "GPU power draw, CPU usage, and system RAM) are captured before and after each engine's run."
    Write-Report ""

    # Capture idle baseline.
    $baselineGpu = Get-GpuSnapshot
    $baselineRam = Get-SystemRamGB

    Write-Report "### System Baseline (Idle)"
    Write-Report ""
    Write-Report "| Metric | Value |"
    Write-Report "|--------|-------|"
    Write-Report "| System RAM | ${baselineRam} GB |"
    if ($baselineGpu) {
        Write-Report "| VRAM | $($baselineGpu.VramUsedMB) / $($baselineGpu.VramTotalMB) MB |"
        Write-Report "| GPU Utilization | $($baselineGpu.GpuUtil)% |"
        Write-Report "| GPU Temperature | $($baselineGpu.TempC)C |"
        Write-Report "| GPU Power | $($baselineGpu.PowerW) W |"
    }
    Write-Report ""
    Write-Report "---"
    Write-Report ""
    Write-Report "## GPU Engines"
    Write-Report ""

    # Track all results for summary table.
    $summaryRows = @()

    # ================================================================
    # GPU SECTION
    # ================================================================

    # --- Ollama (GPU, default) ---
    Write-Host "`n--- Ollama GPU ($OllamaModel) x$Iterations ---" -ForegroundColor Yellow

    # Warmup: ensure model is loaded in GPU memory before measuring.
    # Without this, the first iteration is a cold start (7x slower).
    Write-Host "  Warming up Ollama (loading model to GPU)..." -ForegroundColor DarkGray
    $warmup = Run-OllamaInference -OModel $OllamaModel -Prompt "hi" -MaxTokens 1
    Write-Host "  Warmup done ($($warmup.TokPerSec) tok/s)" -ForegroundColor DarkGray

    $gpuBefore = Get-GpuSnapshot
    $ramBefore = Get-SystemRamGB

    $ollamaGpuResults = @()
    for ($i = 1; $i -le $Iterations; $i++) {
        Write-Host "  Iteration $i/$Iterations..."
        $r = Run-OllamaInference -OModel $OllamaModel -Prompt $testPrompt
        $ollamaGpuResults += $r.TokPerSec
        Write-Host "    $($r.TokPerSec) tok/s ($($r.Tokens) tokens)"
    }

    $gpuAfter = Get-GpuSnapshot
    $ramAfter = Get-SystemRamGB

    $ollamaGpuMean = [math]::Round(($ollamaGpuResults | Measure-Object -Average).Average, 1)
    $ollamaGpuMin = [math]::Round(($ollamaGpuResults | Measure-Object -Minimum).Minimum, 1)
    $ollamaGpuMax = [math]::Round(($ollamaGpuResults | Measure-Object -Maximum).Maximum, 1)
    $ollamaGpuRuns = ($ollamaGpuResults | ForEach-Object { [math]::Round($_, 1) }) -join " / "

    Write-EngineResult -EngineName "Ollama GPU ($OllamaModel)" `
        -MeanTokS $ollamaGpuMean -VsOllamaStr "" `
        -RunsStr $ollamaGpuRuns -MinTokS $ollamaGpuMin -MaxTokS $ollamaGpuMax `
        -RamBefore $ramBefore -RamAfter $ramAfter `
        -GpuBefore $gpuBefore -GpuAfter $gpuAfter

    $summaryRows += @{
        Engine   = "Ollama GPU"
        TokS     = $ollamaGpuMean
        VsOllama = "baseline"
        VramMB   = if ($gpuAfter) { $gpuAfter.VramUsedMB } else { "N/A" }
        PowerW   = if ($gpuAfter) { $gpuAfter.PowerW } else { "N/A" }
        TempC    = if ($gpuAfter) { $gpuAfter.TempC } else { "N/A" }
        RamGB    = $ramAfter
    }

    # --- iTaK Torch GPU Backends ---
    $env:YZMA_BENCHMARK_MODEL = $Model

    $gpuBackends = @(
        @{ Name = "Vulkan (dGPU)"; Path = Join-Path $ProjectDir "lib\windows_amd64_vulkan" },
        @{ Name = "CUDA (dGPU)"; Path = Join-Path $ProjectDir "lib\windows_amd64_cuda" }
    )

    foreach ($backend in $gpuBackends) {
        $backendName = $backend.Name
        $libPath = $backend.Path

        if (-not (Test-Path (Join-Path $libPath "llama.dll"))) {
            Write-Host "  SKIP: $backendName - no llama.dll at $libPath" -ForegroundColor DarkYellow
            Write-Report "### iTaK Torch ($backendName)"
            Write-Report ""
            Write-Report "SKIPPED: llama.dll not found at ``$libPath``"
            Write-Report ""
            continue
        }

        Write-Host "`n--- iTaK Torch [$backendName] ---" -ForegroundColor Yellow
        $env:ITAK_TORCH_LIB = $libPath

        $gpuBefore = Get-GpuSnapshot
        $ramBefore = Get-SystemRamGB

        $goBenchRaw = ""
        try {
            Push-Location $ProjectDir
            $goBenchRaw = go test ./pkg/torch/llama/ -run='^$' -bench=BenchmarkInference -benchtime=10s "-count=$Iterations" -timeout=120s 2>&1
            Pop-Location
        }
        catch {
            Pop-Location
            $goBenchRaw = "ERROR: $($_.Exception.Message)"
        }

        $gpuAfter = Get-GpuSnapshot
        $ramAfter = Get-SystemRamGB

        # Parse tok/s from Go bench output.
        $benchTokS = @()
        foreach ($line in $goBenchRaw) {
            $lineStr = "$line"
            if ($lineStr -match "([\d.]+)\s*tokens/s") {
                $benchTokS += [double]$Matches[1]
            }
        }

        $benchMean = 0; $benchMin = 0; $benchMax = 0; $benchRuns = "N/A"
        if ($benchTokS.Count -gt 0) {
            $benchMean = [math]::Round(($benchTokS | Measure-Object -Average).Average, 1)
            $benchMin = [math]::Round(($benchTokS | Measure-Object -Minimum).Minimum, 1)
            $benchMax = [math]::Round(($benchTokS | Measure-Object -Maximum).Maximum, 1)
            $benchRuns = ($benchTokS | ForEach-Object { [math]::Round($_, 1) }) -join " / "
        }

        $vsOllama = "N/A"
        if ($ollamaGpuMean -gt 0 -and $benchMean -gt 0) {
            $pctDiff = [math]::Round((($benchMean - $ollamaGpuMean) / $ollamaGpuMean) * 100, 1)
            $sign = if ($pctDiff -ge 0) { "+" } else { "" }
            $vsOllama = "${sign}${pctDiff}%"
        }

        Write-EngineResult -EngineName "iTaK Torch ($backendName)" `
            -MeanTokS $benchMean -VsOllamaStr $vsOllama `
            -RunsStr $benchRuns -MinTokS $benchMin -MaxTokS $benchMax `
            -RamBefore $ramBefore -RamAfter $ramAfter `
            -GpuBefore $gpuBefore -GpuAfter $gpuAfter `
            -RawOutput ($goBenchRaw | Out-String)

        Write-Host "  Mean: $benchMean tok/s ($vsOllama)" -ForegroundColor Green

        $summaryRows += @{
            Engine   = "iTaK Torch ($backendName)"
            TokS     = $benchMean
            VsOllama = $vsOllama
            VramMB   = if ($gpuAfter) { $gpuAfter.VramUsedMB } else { "N/A" }
            PowerW   = if ($gpuAfter) { $gpuAfter.PowerW } else { "N/A" }
            TempC    = if ($gpuAfter) { $gpuAfter.TempC } else { "N/A" }
            RamGB    = $ramAfter
        }
    }

    Remove-Item Env:\ITAK_TORCH_LIB -ErrorAction SilentlyContinue


    # ================================================================
    # CPU SECTION
    # ================================================================
    Write-Report "---"
    Write-Report ""
    Write-Report "## CPU Engines"
    Write-Report ""

    # --- Ollama CPU (forced) ---
    Write-Host "`n--- Ollama CPU ($OllamaModel) x$Iterations ---" -ForegroundColor Yellow
    Write-Host "  (CUDA_VISIBLE_DEVICES='' to force CPU)" -ForegroundColor DarkGray

    $gpuBefore = Get-GpuSnapshot
    $ramBefore = Get-SystemRamGB

    $ollamaCpuResults = @()
    for ($i = 1; $i -le $Iterations; $i++) {
        Write-Host "  Iteration $i/$Iterations..."
        $r = Run-OllamaInference -OModel $OllamaModel -Prompt $testPrompt -CpuOnly
        $ollamaCpuResults += $r.TokPerSec
        Write-Host "    $($r.TokPerSec) tok/s ($($r.Tokens) tokens)"
    }

    $gpuAfter = Get-GpuSnapshot
    $ramAfter = Get-SystemRamGB

    $ollamaCpuMean = 0; $ollamaCpuMin = 0; $ollamaCpuMax = 0; $ollamaCpuRuns = "N/A"
    if ($ollamaCpuResults.Count -gt 0) {
        $ollamaCpuMean = [math]::Round(($ollamaCpuResults | Measure-Object -Average).Average, 1)
        $ollamaCpuMin = [math]::Round(($ollamaCpuResults | Measure-Object -Minimum).Minimum, 1)
        $ollamaCpuMax = [math]::Round(($ollamaCpuResults | Measure-Object -Maximum).Maximum, 1)
        $ollamaCpuRuns = ($ollamaCpuResults | ForEach-Object { [math]::Round($_, 1) }) -join " / "
    }

    $vsOllamaCpu = "N/A"
    if ($ollamaGpuMean -gt 0 -and $ollamaCpuMean -gt 0) {
        $pct = [math]::Round((($ollamaCpuMean - $ollamaGpuMean) / $ollamaGpuMean) * 100, 1)
        $vsOllamaCpu = "$(if ($pct -ge 0) {'+'})$pct%"
    }

    Write-EngineResult -EngineName "Ollama CPU ($OllamaModel)" `
        -MeanTokS $ollamaCpuMean -VsOllamaStr $vsOllamaCpu `
        -RunsStr $ollamaCpuRuns -MinTokS $ollamaCpuMin -MaxTokS $ollamaCpuMax `
        -RamBefore $ramBefore -RamAfter $ramAfter `
        -GpuBefore $gpuBefore -GpuAfter $gpuAfter

    $summaryRows += @{
        Engine   = "Ollama CPU"
        TokS     = $ollamaCpuMean
        VsOllama = $vsOllamaCpu
        VramMB   = if ($gpuAfter) { $gpuAfter.VramUsedMB } else { "N/A" }
        PowerW   = if ($gpuAfter) { $gpuAfter.PowerW } else { "N/A" }
        TempC    = if ($gpuAfter) { $gpuAfter.TempC } else { "N/A" }
        RamGB    = $ramAfter
    }

    # --- iTaK Torch CPU ---
    $cpuLibPath = Join-Path $ProjectDir "lib\windows_amd64"
    if (Test-Path (Join-Path $cpuLibPath "llama.dll")) {
        Write-Host "`n--- iTaK Torch [CPU] ---" -ForegroundColor Yellow
        $env:ITAK_TORCH_LIB = $cpuLibPath
        $env:YZMA_BENCHMARK_MODEL = $Model

        $gpuBefore = Get-GpuSnapshot
        $ramBefore = Get-SystemRamGB

        $goBenchRaw = ""
        try {
            Push-Location $ProjectDir
            $goBenchRaw = go test ./pkg/torch/llama/ -run='^$' -bench=BenchmarkInference -benchtime=10s "-count=$Iterations" -timeout=120s 2>&1
            Pop-Location
        }
        catch {
            Pop-Location
            $goBenchRaw = "ERROR: $($_.Exception.Message)"
        }

        $gpuAfter = Get-GpuSnapshot
        $ramAfter = Get-SystemRamGB

        $benchTokS = @()
        foreach ($line in $goBenchRaw) {
            $lineStr = "$line"
            if ($lineStr -match "([\d.]+)\s*tokens/s") {
                $benchTokS += [double]$Matches[1]
            }
        }

        $benchMean = 0; $benchMin = 0; $benchMax = 0; $benchRuns = "N/A"
        if ($benchTokS.Count -gt 0) {
            $benchMean = [math]::Round(($benchTokS | Measure-Object -Average).Average, 1)
            $benchMin = [math]::Round(($benchTokS | Measure-Object -Minimum).Minimum, 1)
            $benchMax = [math]::Round(($benchTokS | Measure-Object -Maximum).Maximum, 1)
            $benchRuns = ($benchTokS | ForEach-Object { [math]::Round($_, 1) }) -join " / "
        }

        $vsOllama = "N/A"
        if ($ollamaGpuMean -gt 0 -and $benchMean -gt 0) {
            $pct = [math]::Round((($benchMean - $ollamaGpuMean) / $ollamaGpuMean) * 100, 1)
            $vsOllama = "$(if ($pct -ge 0) {'+'})$pct%"
        }

        Write-EngineResult -EngineName "iTaK Torch (CPU)" `
            -MeanTokS $benchMean -VsOllamaStr $vsOllama `
            -RunsStr $benchRuns -MinTokS $benchMin -MaxTokS $benchMax `
            -RamBefore $ramBefore -RamAfter $ramAfter `
            -GpuBefore $gpuBefore -GpuAfter $gpuAfter `
            -RawOutput ($goBenchRaw | Out-String)

        $summaryRows += @{
            Engine   = "iTaK Torch CPU"
            TokS     = $benchMean
            VsOllama = $vsOllama
            VramMB   = if ($gpuAfter) { $gpuAfter.VramUsedMB } else { "N/A" }
            PowerW   = if ($gpuAfter) { $gpuAfter.PowerW } else { "N/A" }
            TempC    = if ($gpuAfter) { $gpuAfter.TempC } else { "N/A" }
            RamGB    = $ramAfter
        }

        Remove-Item Env:\ITAK_TORCH_LIB -ErrorAction SilentlyContinue
    }
    else {
        Write-Host "  SKIP: iTaK Torch CPU - no llama.dll" -ForegroundColor DarkYellow
        Write-Report "### iTaK Torch (CPU)"
        Write-Report ""
        Write-Report "SKIPPED: llama.dll not found at ``$cpuLibPath``"
        Write-Report ""
    }


    # ================================================================
    # SUMMARY COMPARISON TABLE
    # ================================================================
    Write-Report "---"
    Write-Report ""
    Write-Report "## Summary Comparison"
    Write-Report ""
    Write-Report "| Engine | tok/s | vs Ollama GPU | VRAM (MB) | Power (W) | Temp (C) | RAM (GB) |"
    Write-Report "|--------|-------|---------------|-----------|-----------|----------|----------|"
    foreach ($row in $summaryRows) {
        $bold = if ($row.TokS -eq ($summaryRows | ForEach-Object { $_.TokS } | Measure-Object -Maximum).Maximum) { "**" } else { "" }
        Write-Report "| $($row.Engine) | ${bold}$($row.TokS)${bold} | $($row.VsOllama) | $($row.VramMB) | $($row.PowerW) | $($row.TempC) | $($row.RamGB) |"
    }
    Write-Report ""
}

# ============================================================
# CAT 5: THREAD SCALING
# ============================================================
function Run-ThreadScaling {
    Write-Host "`n=== Category 5: Thread Scaling ===" -ForegroundColor Cyan

    Write-Report "## Category 5: Thread Scaling"
    Write-Report ""

    $threadResult = & { Set-Location $ProjectDir; go test ./pkg/torch/ -run TestDetectOptimal -v -count=1 2>&1 }

    Write-Report '```'
    Write-Report ($threadResult | Out-String)
    Write-Report '```'
    Write-Report ""
}

# ============================================================
# CAT 1: COLD BOOT TIME
# ============================================================
function Run-BootTime {
    Write-Host "`n=== Category 1: Cold Boot Time ===" -ForegroundColor Cyan

    Write-Report "## Category 1: Cold Boot Time"
    Write-Report ""

    Write-Host "Testing Ollama cold boot..." -ForegroundColor Yellow
    cmd /c "ollama stop $OllamaModel" 2>$null

    $gpuBefore = Get-GpuSnapshot
    $ramBefore = Get-SystemRamGB

    $bootStart = Get-Date
    $bootResult = cmd /c "echo hi | ollama run $OllamaModel 2>&1"
    $bootEnd = Get-Date
    $bootMs = [math]::Round(($bootEnd - $bootStart).TotalMilliseconds)

    $gpuAfter = Get-GpuSnapshot
    $ramAfter = Get-SystemRamGB

    Write-Report "| Engine | Boot (ms) | RAM Before/After (GB) | VRAM Before/After (MB) | GPU Temp (C) |"
    Write-Report "|--------|----------|----------------------|----------------------|-------------|"
    $vramStr = if ($gpuBefore -and $gpuAfter) { "$($gpuBefore.VramUsedMB) / $($gpuAfter.VramUsedMB)" } else { "N/A" }
    $tempStr = if ($gpuBefore -and $gpuAfter) { "$($gpuBefore.TempC) / $($gpuAfter.TempC)" } else { "N/A" }
    Write-Report "| Ollama ($OllamaModel) | $bootMs | $ramBefore / $ramAfter | $vramStr | $tempStr |"
    Write-Report ""
}

# ============================================================
# MAIN EXECUTION
# ============================================================
Write-Host "================================================" -ForegroundColor Green
Write-Host "  iTaK Torch Benchmark Suite v2.1 - Beast" -ForegroundColor Green
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

Write-Header

switch ($Category) {
    "all" {
        Run-UnitTests
        Run-BootTime
        Run-EngineMatrix
        Run-ThreadScaling
    }
    "unit" { Run-UnitTests }
    "boot" { Run-BootTime }
    # 'throughput' and 'resources' are both aliases for the unified matrix.
    "matrix" { Run-EngineMatrix }
    "throughput" { Run-EngineMatrix }
    "resources" { Run-EngineMatrix }
    "threads" { Run-ThreadScaling }
}

Write-Report "---"
Write-Report ""
Write-Report "*Report generated by iTaK Torch Benchmark Skill v2.1*"

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  Benchmark complete!" -ForegroundColor Green
Write-Host "  Report: $reportFile" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
