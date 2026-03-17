<#
.SYNOPSIS
    Nemotron MoE Shared Memory Benchmark
.DESCRIPTION
    Tests nemotron-3-nano:30b-a3b-q4_K_M (24 GB MoE, 3B active params) across
    multiple GPU layer configurations to measure shared memory performance.
    
    RTX 4070 Ti SUPER has 16 GB VRAM - this model exceeds that by 8 GB.
    Tests what happens when the model spills into Windows shared GPU memory.

.PARAMETER Iterations
    Number of iterations per config (default: 3).
.PARAMETER OutputDir
    Directory for results report.
#>
param(
    [int]$Iterations = 3,
    [string]$OutputDir = "e:\.agent\iTaK Eco\Torch\benchmarks\results"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$reportFile = Join-Path $OutputDir "nemotron_moe_${timestamp}.md"

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$ollamaAPI = "http://localhost:11434"

# Both nemotron aliases share the same blob (b725f1117407), testing just one.
$models = @(
    @{ Name = "nemotron-3-nano:30b-a3b-q4_K_M";  Tag = "30b-a3b-q4_K_M" }
)

# GPU layer configs to test.
$gpuConfigs = @(
    @{ Name = "CPU Only";        NumGPU = 0;   Desc = "All layers in system RAM, no GPU" },
    @{ Name = "GPU 20 layers";   NumGPU = 20;  Desc = "20 layers on VRAM, rest on CPU" },
    @{ Name = "GPU 40 layers";   NumGPU = 40;  Desc = "40 layers - starts filling VRAM" },
    @{ Name = "GPU 60 layers";   NumGPU = 60;  Desc = "60 layers - heavy VRAM, some shared" },
    @{ Name = "GPU Max (999)";   NumGPU = 999; Desc = "All layers to GPU - spills into shared memory" }
)

$testPrompt = "Explain the concept of mixture-of-experts in neural networks in exactly 3 sentences."

# ============================================================
# RESOURCE MONITORING HELPERS
# ============================================================

function Get-GpuSnapshot {
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
    } catch {}
    return $null
}

function Get-SystemRamGB {
    [math]::Round((Get-Process | Measure-Object WorkingSet64 -Sum).Sum / 1GB, 2)
}

function Get-CpuUsagePercent {
    try {
        $sample = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue
        return [math]::Round($sample, 1)
    } catch { return 0 }
}

function Unload-OllamaModel {
    param([string]$ModelName)
    try {
        $body = @{ model = $ModelName; keep_alive = 0 } | ConvertTo-Json
        Invoke-RestMethod -Uri "$ollamaAPI/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 15 2>$null | Out-Null
        Start-Sleep -Seconds 5
    } catch {}
}

function Run-OllamaInference {
    param(
        [string]$ModelName,
        [string]$Prompt,
        [int]$NumGPU,
        [int]$MaxTokens = 256
    )

    $body = @{
        model   = $ModelName
        prompt  = $Prompt
        stream  = $false
        options = @{
            num_gpu     = $NumGPU
            num_predict = $MaxTokens
            num_ctx     = 2048
        }
    } | ConvertTo-Json -Depth 3

    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-RestMethod -Uri "$ollamaAPI/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
        $sw.Stop()

        $evalCount = if ($response.eval_count) { [int]$response.eval_count } else { 0 }
        $evalDurationNs = if ($response.eval_duration) { [double]$response.eval_duration } else { 0 }
        $loadDurationNs = if ($response.load_duration) { [double]$response.load_duration } else { 0 }
        $promptEvalNs = if ($response.prompt_eval_duration) { [double]$response.prompt_eval_duration } else { 0 }

        $tokPerSec = 0
        if ($evalDurationNs -gt 0) {
            $tokPerSec = [math]::Round($evalCount / ($evalDurationNs / 1e9), 1)
        }

        return @{
            TokPerSec     = $tokPerSec
            Tokens        = $evalCount
            LoadTimeSec   = [math]::Round($loadDurationNs / 1e9, 2)
            PromptEvalSec = [math]::Round($promptEvalNs / 1e9, 2)
            EvalTimeSec   = [math]::Round($evalDurationNs / 1e9, 2)
            TotalTimeSec  = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            ResponseSnip  = if ($response.response.Length -gt 120) { $response.response.Substring(0, 120) + "..." } else { $response.response }
            Success       = $true
        }
    }
    catch {
        $sw.Stop()
        return @{
            TokPerSec     = 0
            Tokens        = 0
            LoadTimeSec   = 0
            PromptEvalSec = 0
            EvalTimeSec   = 0
            TotalTimeSec  = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            ResponseSnip  = "ERROR: $($_.Exception.Message)"
            Success       = $false
        }
    }
}

# ============================================================
# REPORT WRITER
# ============================================================
function Write-Report {
    param([string]$Content)
    Add-Content -Path $reportFile -Value $Content
}

# ============================================================
# MAIN EXECUTION
# ============================================================

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Nemotron MoE Shared Memory Benchmark" -ForegroundColor Green
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# System info.
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
$totalRam = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$gpuName = (cmd /c 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>$null).Trim()
$gpuVram = (cmd /c 'nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits' 2>$null).Trim()

Write-Report "# Nemotron MoE Shared Memory Benchmark"
Write-Report ""
Write-Report "**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Report "**CPU:** $($cpu.Name) ($($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T)"
Write-Report "**RAM:** ${totalRam} GB"
Write-Report "**GPU:** $gpuName ($gpuVram MiB dedicated VRAM)"
Write-Report "**Model:** nemotron-3-nano:30b-a3b-q4_K_M (24 GB, MoE 3B active params)"
Write-Report "**Iterations:** $Iterations per config"
Write-Report "**Prompt:** ``$testPrompt``"
Write-Report ""

# Idle baseline.
$baseGpu = Get-GpuSnapshot
$baseRam = Get-SystemRamGB

Write-Report "## Idle Baseline"
Write-Report ""
Write-Report "| Metric | Value |"
Write-Report "|--------|-------|"
Write-Report "| System RAM | ${baseRam} GB |"
if ($baseGpu) {
    Write-Report "| VRAM | $($baseGpu.VramUsedMB) / $($baseGpu.VramTotalMB) MB |"
    Write-Report "| GPU Temp | $($baseGpu.TempC) C |"
    Write-Report "| GPU Power | $($baseGpu.PowerW) W |"
}
Write-Report ""
Write-Report "---"
Write-Report ""

# Track all results for summary table.
$allResults = @()

# ============================================================
# RUN TESTS: each model x each GPU config
# ============================================================

foreach ($model in $models) {
    $modelName = $model.Name
    $modelTag = $model.Tag

    Write-Report "## Model: $modelName"
    Write-Report ""

    foreach ($cfg in $gpuConfigs) {
        $configName = "$($cfg.Name)"
        $fullLabel = "$modelTag | $configName"

        Write-Host ""
        Write-Host "=== $fullLabel ===" -ForegroundColor Cyan
        Write-Host "  $($cfg.Desc)" -ForegroundColor DarkGray

        # Unload model before each config to get clean measurements.
        Write-Host "  Unloading model..." -ForegroundColor DarkGray
        Unload-OllamaModel -ModelName $modelName
        Start-Sleep -Seconds 3

        # Capture resources before.
        $gpuBefore = Get-GpuSnapshot
        $ramBefore = Get-SystemRamGB

        # Run iterations.
        $tokResults = @()
        $loadTimes = @()
        $firstResult = $null

        for ($i = 1; $i -le $Iterations; $i++) {
            Write-Host "  Iteration $i/$Iterations..." -NoNewline
            $result = Run-OllamaInference -ModelName $modelName -Prompt $testPrompt -NumGPU $cfg.NumGPU

            if ($result.Success) {
                $tokResults += $result.TokPerSec
                $loadTimes += $result.LoadTimeSec
                if ($i -eq 1) { $firstResult = $result }
                Write-Host " $($result.TokPerSec) tok/s ($($result.Tokens) tokens, load: $($result.LoadTimeSec)s)" -ForegroundColor Green
            } else {
                Write-Host " FAILED: $($result.ResponseSnip)" -ForegroundColor Red
            }
        }

        # Capture resources after (model still loaded).
        $gpuAfter = Get-GpuSnapshot
        $ramAfter = Get-SystemRamGB
        $cpuPct = Get-CpuUsagePercent

        # Calculate stats.
        if ($tokResults.Count -gt 0) {
            $meanTok = [math]::Round(($tokResults | Measure-Object -Average).Average, 1)
            $minTok = [math]::Round(($tokResults | Measure-Object -Minimum).Minimum, 1)
            $maxTok = [math]::Round(($tokResults | Measure-Object -Maximum).Maximum, 1)
            $runsStr = ($tokResults | ForEach-Object { [math]::Round($_, 1) }) -join " / "
            $coldLoadSec = $loadTimes[0]
            $warmLoadSec = if ($loadTimes.Count -gt 1) { $loadTimes[-1] } else { $loadTimes[0] }
        } else {
            $meanTok = 0; $minTok = 0; $maxTok = 0; $runsStr = "FAILED"
            $coldLoadSec = 0; $warmLoadSec = 0
        }

        $vramDelta = if ($gpuBefore -and $gpuAfter) { $gpuAfter.VramUsedMB - $gpuBefore.VramUsedMB } else { 0 }
        $ramDelta = [math]::Round($ramAfter - $ramBefore, 2)

        # Write report block.
        Write-Report "### $configName (num_gpu=$($cfg.NumGPU))"
        Write-Report ""
        Write-Report "*$($cfg.Desc)*"
        Write-Report ""
        Write-Report "| Metric | Value |"
        Write-Report "|--------|-------|"
        Write-Report "| **Mean tok/s** | **$meanTok** |"
        Write-Report "| Runs | $runsStr |"
        Write-Report "| Min / Max | $minTok / $maxTok |"
        Write-Report "| Cold Load | ${coldLoadSec}s |"
        Write-Report "| Warm Load | ${warmLoadSec}s |"
        Write-Report "| CPU Usage | ${cpuPct}% |"
        Write-Report "| System RAM | ${ramBefore} -> ${ramAfter} GB (delta: ${ramDelta}) |"
        if ($gpuBefore -and $gpuAfter) {
            Write-Report "| VRAM | $($gpuBefore.VramUsedMB) -> $($gpuAfter.VramUsedMB) MB (delta: ${vramDelta}) |"
            Write-Report "| GPU Temp | $($gpuBefore.TempC) -> $($gpuAfter.TempC) C |"
            Write-Report "| GPU Power | $($gpuBefore.PowerW) -> $($gpuAfter.PowerW) W |"
        }
        Write-Report ""

        # Track for summary.
        $allResults += @{
            Label     = $fullLabel
            Model     = $modelTag
            Config    = $configName
            NumGPU    = $cfg.NumGPU
            MeanTokS  = $meanTok
            ColdLoad  = $coldLoadSec
            VramMB    = if ($gpuAfter) { $gpuAfter.VramUsedMB } else { 0 }
            VramDelta = $vramDelta
            RamGB     = $ramAfter
            RamDelta  = $ramDelta
            PowerW    = if ($gpuAfter) { $gpuAfter.PowerW } else { 0 }
            TempC     = if ($gpuAfter) { $gpuAfter.TempC } else { 0 }
            CpuPct    = $cpuPct
        }

        Write-Host "  Summary: $meanTok tok/s | VRAM: +${vramDelta}MB | RAM: +${ramDelta}GB" -ForegroundColor Yellow
    }

    Write-Report "---"
    Write-Report ""
}

# ============================================================
# SUMMARY COMPARISON TABLE
# ============================================================

Write-Report "## Summary Comparison"
Write-Report ""
Write-Report "| Model | Config | num_gpu | tok/s | Cold Load | VRAM delta | VRAM total | RAM delta | Power | Temp |"
Write-Report "|-------|--------|---------|-------|-----------|-----------|------------|-----------|-------|------|"

# Sort by tok/s descending.
$sorted = $allResults | Sort-Object { $_.MeanTokS } -Descending
$bestTokS = ($sorted | Select-Object -First 1).MeanTokS

foreach ($row in $sorted) {
    $bold = if ($row.MeanTokS -eq $bestTokS -and $row.MeanTokS -gt 0) { "**" } else { "" }
    Write-Report "| $($row.Model) | $($row.Config) | $($row.NumGPU) | ${bold}$($row.MeanTokS)${bold} | $($row.ColdLoad)s | +$($row.VramDelta)MB | $($row.VramMB)MB | +$($row.RamDelta)GB | $($row.PowerW)W | $($row.TempC)C |"
}
Write-Report ""

# ============================================================
# ANALYSIS
# ============================================================

Write-Report "## Analysis"
Write-Report ""
Write-Report "### MoE Shared Memory Hypothesis"
Write-Report ""
Write-Report "nemotron-3-nano has 30B total parameters but only 3B active per forward pass."
Write-Report "This MoE architecture means most weights are 'cold' during inference."
Write-Report "The question: does placing cold expert weights in shared GPU memory (system RAM"
Write-Report "accessible via PCIe) perform differently than dedicated VRAM or pure CPU?"
Write-Report ""

# Calculate interesting comparisons.
$cpuResult = $allResults | Where-Object { $_.NumGPU -eq 0 } | Select-Object -First 1
$maxGpuResult = $allResults | Where-Object { $_.NumGPU -eq 999 } | Select-Object -First 1

if ($cpuResult -and $maxGpuResult -and $cpuResult.MeanTokS -gt 0 -and $maxGpuResult.MeanTokS -gt 0) {
    $speedup = [math]::Round($maxGpuResult.MeanTokS / $cpuResult.MeanTokS, 2)
    Write-Report "### Key Finding"
    Write-Report ""
    Write-Report "- **GPU Max (shared memory spill) vs CPU Only:** ${speedup}x speedup"
    Write-Report "- CPU Only: $($cpuResult.MeanTokS) tok/s"
    Write-Report "- GPU Max (spill): $($maxGpuResult.MeanTokS) tok/s"
    Write-Report ""
}

Write-Report "---"
Write-Report ""
Write-Report "*Report generated by iTaK Torch MoE Benchmark v1.0*"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Benchmark complete!" -ForegroundColor Green
Write-Host "  Report: $reportFile" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
