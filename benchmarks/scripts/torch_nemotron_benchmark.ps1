<#
.SYNOPSIS
    iTaK Torch Nemotron Benchmark - Windows + WSL
.DESCRIPTION
    Benchmarks nemotron-3-nano:30b-a3b-q4_K_M via Torch's llama.cpp FFI
    on both Windows (DLLs) and WSL (SOs), CPU-only mode.
    Measures tokens/sec from the /v1/chat/completions streaming endpoint.
#>

$ErrorActionPreference = "Stop"
$GGUF = "e:\.agent\iTaK Eco\Torch\~\.itaktorch\models\ollama-nemotron-3-nano-30b-a3b-q4_K_M.gguf"
$WIN_TORCH = "$env:TEMP\itaktorch.exe"
$LIN_TORCH = "$env:TEMP\itaktorch-linux"
$PORT_WIN = 28345
$PORT_WSL = 28346
$PROMPT = "Explain the concept of mixture-of-experts in neural networks in exactly 3 sentences."
$ITERATIONS = 2
$REPORT_DIR = "e:\.agent\iTaK Eco\Torch\benchmarks\results"

# ---------- Helper Functions ----------

function Get-GpuSnapshot {
    try {
        $out = nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
        if ($out) {
            $parts = $out.Trim() -split ",\s*"
            return @{
                VramUsed  = [int]$parts[0]
                VramTotal = [int]$parts[1]
                TempC     = [int]$parts[2]
                PowerW    = [double]$parts[3]
            }
        }
    } catch {}
    return @{ VramUsed=0; VramTotal=0; TempC=0; PowerW=0 }
}

function Get-SystemRamGB {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 2)
}

function Invoke-TorchInference {
    param(
        [string]$Port,
        [string]$Prompt,
        [int]$MaxTokens = 256
    )
    $body = @{
        model = "default"
        messages = @(@{ role = "user"; content = $Prompt })
        max_tokens = $MaxTokens
        temperature = 0.1
        stream = $false
    } | ConvertTo-Json -Depth 5

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:$Port/v1/chat/completions" `
            -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
        $sw.Stop()

        $tokens = 0
        if ($resp.usage) { $tokens = $resp.usage.completion_tokens }
        if ($tokens -eq 0 -and $resp.choices) {
            $text = $resp.choices[0].message.content
            $tokens = ($text -split "\s+").Count  # rough estimate
        }

        $elapsed = $sw.Elapsed.TotalSeconds
        $tokPerSec = if ($elapsed -gt 0) { [math]::Round($tokens / $elapsed, 1) } else { 0 }

        return @{
            TokPerSec = $tokPerSec
            Tokens    = $tokens
            Elapsed   = [math]::Round($elapsed, 2)
            Text      = if ($resp.choices) { $resp.choices[0].message.content.Substring(0, [math]::Min(100, $resp.choices[0].message.content.Length)) } else { "" }
        }
    } catch {
        $sw.Stop()
        return @{ TokPerSec = 0; Tokens = 0; Elapsed = $sw.Elapsed.TotalSeconds; Text = "ERROR: $_" }
    }
}

function Wait-ForServer {
    param([string]$Port, [int]$TimeoutSec = 300)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 2
    }
    return $false
}

# ---------- Report ----------

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$reportFile = "$REPORT_DIR\torch_nemotron_$timestamp.md"
New-Item -ItemType Directory -Force -Path $REPORT_DIR | Out-Null

$report = @()
$report += "# Torch Nemotron Benchmark"
$report += ""
$report += "**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$report += "**Model:** nemotron-3-nano:30b-a3b-q4_K_M (22.6 GB GGUF, MoE 3B active)"
$report += "**Engine:** iTaK Torch (llama.cpp b8250 FFI)"
$report += "**Iterations:** $ITERATIONS per config"
$report += "**Prompt:** ``$PROMPT``"
$report += ""

# ---------- Idle Baseline ----------

$gpuIdle = Get-GpuSnapshot
$ramIdle = Get-SystemRamGB
$report += "## Idle Baseline"
$report += ""
$report += "| Metric | Value |"
$report += "|--------|-------|"
$report += "| System RAM | $ramIdle GB |"
$report += "| VRAM | $($gpuIdle.VramUsed) / $($gpuIdle.VramTotal) MB |"
$report += "| GPU Temp | $($gpuIdle.TempC) C |"
$report += "| GPU Power | $($gpuIdle.PowerW) W |"
$report += ""

# ========================================================
# TEST 1: Windows CPU
# ========================================================

Write-Host "============================================================"
Write-Host "  TEST 1: Torch Windows CPU"
Write-Host "============================================================"

$report += "## Test 1: Torch Windows CPU"
$report += ""
$report += "*Using DLLs from lib/windows_amd64/*"
$report += ""

$gpuBefore = Get-GpuSnapshot
$ramBefore = Get-SystemRamGB

Write-Host "  Starting Torch serve on port $PORT_WIN..."
$winProc = Start-Process -FilePath $WIN_TORCH -ArgumentList "serve","--model",$GGUF,"--port",$PORT_WIN,"--gpu-layers","0","--ctx","2048","--threads","0" -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\torch_win_out.txt" -RedirectStandardError "$env:TEMP\torch_win_err.txt"

Write-Host "  Waiting for model to load (22.6 GB)..."
$ready = Wait-ForServer -Port $PORT_WIN -TimeoutSec 300
if (-not $ready) {
    Write-Host "  FAILED: Server did not start in time"
    $report += "**FAILED:** Server did not start within 300s"
    $report += ""
    if ($winProc -and -not $winProc.HasExited) { $winProc.Kill() }
} else {
    $gpuAfter = Get-GpuSnapshot
    $ramAfter = Get-SystemRamGB
    $loadTime = "see logs"  # we'll get this from stderr

    Write-Host "  Server ready! Running $ITERATIONS iterations..."
    $results = @()
    for ($i = 1; $i -le $ITERATIONS; $i++) {
        Write-Host "  Iteration $i/$ITERATIONS..." -NoNewline
        $r = Invoke-TorchInference -Port $PORT_WIN -Prompt $PROMPT
        Write-Host " $($r.TokPerSec) tok/s ($($r.Tokens) tokens, $($r.Elapsed)s)"
        $results += $r
    }

    $meanTokSec = [math]::Round(($results | Measure-Object -Property TokPerSec -Average).Average, 1)
    $minTokSec = ($results | Measure-Object -Property TokPerSec -Minimum).Minimum
    $maxTokSec = ($results | Measure-Object -Property TokPerSec -Maximum).Maximum
    $runs = ($results | ForEach-Object { $_.TokPerSec }) -join " / "

    $report += "| Metric | Value |"
    $report += "|--------|-------|"
    $report += "| **Mean tok/s** | **$meanTokSec** |"
    $report += "| Runs | $runs |"
    $report += "| Min / Max | $minTokSec / $maxTokSec |"
    $report += "| System RAM | $ramBefore -> $ramAfter GB (delta: $([math]::Round($ramAfter - $ramBefore, 2))) |"
    $report += "| VRAM | $($gpuBefore.VramUsed) -> $($gpuAfter.VramUsed) MB (delta: $($gpuAfter.VramUsed - $gpuBefore.VramUsed)) |"
    $report += "| GPU Temp | $($gpuBefore.TempC) -> $($gpuAfter.TempC) C |"
    $report += "| GPU Power | $($gpuBefore.PowerW) -> $($gpuAfter.PowerW) W |"
    $report += ""

    Write-Host "  Summary: $meanTokSec tok/s | VRAM: +$($gpuAfter.VramUsed - $gpuBefore.VramUsed)MB | RAM: +$([math]::Round($ramAfter - $ramBefore, 2))GB"

    # Stop server
    if ($winProc -and -not $winProc.HasExited) { $winProc.Kill(); $winProc.WaitForExit(5000) }
    Start-Sleep -Seconds 5  # let resources settle
}

# ========================================================
# TEST 2: WSL CPU
# ========================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "  TEST 2: Torch WSL CPU"
Write-Host "============================================================"

$report += "## Test 2: Torch WSL CPU"
$report += ""
$report += "*Using SOs from lib/linux_amd64/ via WSL*"
$report += ""

$gpuBefore = Get-GpuSnapshot
$ramBefore = Get-SystemRamGB

# Write the WSL launch script
$wslScript = @"
#!/bin/bash
export ITAK_TORCH_LIB="/mnt/e/.agent/iTaK Eco/Torch/lib/linux_amd64"
export LD_LIBRARY_PATH="/mnt/e/.agent/iTaK Eco/Torch/lib/linux_amd64"
TORCH="/mnt/c/Users/David/AppData/Local/Temp/itaktorch-linux"
GGUF="/mnt/e/.agent/iTaK Eco/Torch/~/.itaktorch/models/ollama-nemotron-3-nano-30b-a3b-q4_K_M.gguf"
chmod +x "`$TORCH"
exec "`$TORCH" serve --model "`$GGUF" --port $PORT_WSL --gpu-layers 0 --ctx 2048 --threads 0 2>&1
"@
$wslScript | Set-Content -Path "$env:TEMP\torch_wsl_serve.sh" -Encoding UTF8 -NoNewline

Write-Host "  Starting Torch serve on WSL port $PORT_WSL..."
$wslProc = Start-Process -FilePath "wsl" -ArgumentList "-e","bash","/mnt/c/Users/David/AppData/Local/Temp/torch_wsl_serve.sh" -PassThru -NoNewWindow

Write-Host "  Waiting for model to load in WSL (22.6 GB)..."
$ready = Wait-ForServer -Port $PORT_WSL -TimeoutSec 300
if (-not $ready) {
    Write-Host "  FAILED: WSL server did not start in time"
    $report += "**FAILED:** WSL server did not start within 300s"
    $report += ""
    if ($wslProc -and -not $wslProc.HasExited) { $wslProc.Kill() }
} else {
    $gpuAfter = Get-GpuSnapshot
    $ramAfter = Get-SystemRamGB

    Write-Host "  Server ready! Running $ITERATIONS iterations..."
    $results = @()
    for ($i = 1; $i -le $ITERATIONS; $i++) {
        Write-Host "  Iteration $i/$ITERATIONS..." -NoNewline
        $r = Invoke-TorchInference -Port $PORT_WSL -Prompt $PROMPT
        Write-Host " $($r.TokPerSec) tok/s ($($r.Tokens) tokens, $($r.Elapsed)s)"
        $results += $r
    }

    $meanTokSec = [math]::Round(($results | Measure-Object -Property TokPerSec -Average).Average, 1)
    $minTokSec = ($results | Measure-Object -Property TokPerSec -Minimum).Minimum
    $maxTokSec = ($results | Measure-Object -Property TokPerSec -Maximum).Maximum
    $runs = ($results | ForEach-Object { $_.TokPerSec }) -join " / "

    $report += "| Metric | Value |"
    $report += "|--------|-------|"
    $report += "| **Mean tok/s** | **$meanTokSec** |"
    $report += "| Runs | $runs |"
    $report += "| Min / Max | $minTokSec / $maxTokSec |"
    $report += "| System RAM | $ramBefore -> $ramAfter GB (delta: $([math]::Round($ramAfter - $ramBefore, 2))) |"
    $report += "| VRAM | $($gpuBefore.VramUsed) -> $($gpuAfter.VramUsed) MB (delta: $($gpuAfter.VramUsed - $gpuBefore.VramUsed)) |"
    $report += "| GPU Temp | $($gpuBefore.TempC) -> $($gpuAfter.TempC) C |"
    $report += "| GPU Power | $($gpuBefore.PowerW) -> $($gpuAfter.PowerW) W |"
    $report += ""

    Write-Host "  Summary: $meanTokSec tok/s | VRAM: +$($gpuAfter.VramUsed - $gpuBefore.VramUsed)MB | RAM: +$([math]::Round($ramAfter - $ramBefore, 2))GB"

    # Stop WSL server
    if ($wslProc -and -not $wslProc.HasExited) { $wslProc.Kill(); $wslProc.WaitForExit(5000) }
}

# ========================================================
# Summary Comparison
# ========================================================

$report += "---"
$report += ""
$report += "## Summary: Torch vs Ollama"
$report += ""
$report += "| Engine | Platform | tok/s | Notes |"
$report += "|--------|----------|-------|-------|"
$report += "| Ollama | Windows | **17.6** | GPU 40 layers (best config) |"
$report += "| Ollama | Windows | 15.8 | CPU Only |"
$report += "| Torch | Windows | *pending* | CPU via DLLs |"
$report += "| Torch | WSL | *pending* | CPU via SOs |"
$report += ""
$report += "*Report generated by iTaK Torch Benchmark v1.0*"

# Write report
$report | Out-File -FilePath $reportFile -Encoding UTF8
Write-Host ""
Write-Host "============================================================"
Write-Host "  Report: $reportFile"
Write-Host "============================================================"
