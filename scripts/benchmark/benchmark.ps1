# iTaK Torch vs Ollama Benchmark Script
# Tracks: tok/s, CPU%, RAM (MB), GPU%, VRAM (MB), model load time
# Usage: .\benchmark.ps1 -Engine itaktorch -Mode cpu
#        .\benchmark.ps1 -Engine ollama -Mode gpu

param(
    [ValidateSet("itaktorch", "ollama")]
    [string]$Engine = "itaktorch",

    [ValidateSet("cpu", "gpu")]
    [string]$Mode = "cpu",

    [string]$Model = "qwen3-0.6b-q4_k_m",
    [string]$ModelPath = "$env:USERPROFILE\.itaktorch\models\qwen3-0.6b-q4_k_m.gguf",
    [int]$Runs = 3,
    [int]$MaxTokens = 100,
    [int]$Threads = 8,
    [int]$Port = 41950,
    [string]$Prompt = "Write a haiku about the sunrise. Be creative and original."
)

$ErrorActionPreference = "Stop"

# --- Helper Functions ---

function Get-GpuStats {
    try {
        $raw = nvidia-smi --query-gpu=utilization.gpu, memory.used, memory.total --format=csv, noheader, nounits 2>$null
        if ($raw) {
            $parts = $raw.Trim() -split ",\s*"
            return @{
                GpuPercent  = [int]$parts[0]
                VramUsedMB  = [int]$parts[1]
                VramTotalMB = [int]$parts[2]
            }
        }
    }
    catch {}
    return @{ GpuPercent = 0; VramUsedMB = 0; VramTotalMB = 0 }
}

function Get-ProcessStats($ProcessName) {
    $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc) {
        return @{
            CpuTime = $proc.TotalProcessorTime.TotalSeconds
            RamMB   = [math]::Round($proc.WorkingSet64 / 1MB, 1)
            PID     = $proc.Id
        }
    }
    return $null
}

function Send-InferenceRequest($Port, $MaxTokens, $Prompt) {
    $body = @{
        model      = $Model
        messages   = @(@{ role = "user"; content = $Prompt })
        max_tokens = $MaxTokens
        stream     = $false
    } | ConvertTo-Json -Depth 5

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
        -Method POST -ContentType "application/json" -Body $body -TimeoutSec 60
    $sw.Stop()

    return @{
        CompletionTokens = $resp.usage.completion_tokens
        PromptTokens     = $resp.usage.prompt_tokens
        WallTimeMs       = $sw.ElapsedMilliseconds
        TokPerSec        = if ($sw.Elapsed.TotalSeconds -gt 0) {
            [math]::Round($resp.usage.completion_tokens / $sw.Elapsed.TotalSeconds, 1)
        }
        else { 0 }
    }
}

# --- Main ---

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  iTaK Torch Benchmark: $Engine ($Mode)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Model:      $Model"
Write-Host "Runs:       $Runs"
Write-Host "MaxTokens:  $MaxTokens"
Write-Host "Threads:    $Threads"
Write-Host "Port:       $Port"
Write-Host ""

# Baseline GPU stats before starting
$gpuBaseline = Get-GpuStats
Write-Host "[baseline] GPU: $($gpuBaseline.GpuPercent)% | VRAM: $($gpuBaseline.VramUsedMB)/$($gpuBaseline.VramTotalMB) MB"

# Start the engine
$serverProc = $null
$loadStart = Get-Date

if ($Engine -eq "itaktorch") {
    $processName = "itaktorch"
    $args = @("serve", "--model", $ModelPath, "--port", $Port, "--threads", $Threads)
    if ($Mode -eq "gpu") {
        $args += @("--gpu-layers", "99")
    }
    Write-Host "[start] itaktorch.exe $($args -join ' ')" -ForegroundColor Yellow
    $serverProc = Start-Process -FilePath ".\itaktorch.exe" -ArgumentList $args `
        -PassThru -NoNewWindow -RedirectStandardOutput ".\bench_stdout.log" -RedirectStandardError ".\bench_stderr.log"
}
elseif ($Engine -eq "ollama") {
    $processName = "ollama_llama_server"
    # Ollama runs as a service - just make sure it's running and pull the model
    Write-Host "[start] Using existing Ollama service" -ForegroundColor Yellow
    ollama pull $Model 2>$null
    if ($Mode -eq "cpu") {
        $env:CUDA_VISIBLE_DEVICES = "-1"
    }
    else {
        Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    }
    # Ollama uses port 11434 by default
    $Port = 11434
}

# Wait for server to be ready
Write-Host "[wait] Waiting for server on port $Port..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($health) { $ready = $true; break }
    }
    catch {}
    Start-Sleep -Milliseconds 500
}

if (-not $ready) {
    Write-Host "[ERROR] Server did not start within 30 seconds" -ForegroundColor Red
    if ($serverProc) { $serverProc | Stop-Process -Force -ErrorAction SilentlyContinue }
    exit 1
}

$loadTime = ((Get-Date) - $loadStart).TotalMilliseconds
Write-Host "[ready] Server ready in $([math]::Round($loadTime))ms" -ForegroundColor Green

# Post-load resource snapshot
$gpuPostLoad = Get-GpuStats
$procStats = Get-ProcessStats $processName
Write-Host "[loaded] GPU: $($gpuPostLoad.GpuPercent)% | VRAM: $($gpuPostLoad.VramUsedMB) MB"
if ($procStats) {
    Write-Host "[loaded] Process RAM: $($procStats.RamMB) MB (PID: $($procStats.PID))"
}

# Run benchmark
Write-Host ""
Write-Host "--- Running $Runs inference requests ---" -ForegroundColor Cyan

$results = @()
for ($run = 1; $run -le $Runs; $run++) {
    # Capture pre-request stats
    $gpuPre = Get-GpuStats
    $procPre = Get-ProcessStats $processName

    # Send request
    $resp = Send-InferenceRequest -Port $Port -MaxTokens $MaxTokens -Prompt $Prompt

    # Capture post-request stats
    $gpuPost = Get-GpuStats
    $procPost = Get-ProcessStats $processName

    # Calculate CPU% from process time delta
    $cpuPercent = 0
    if ($procPre -and $procPost) {
        $cpuDelta = $procPost.CpuTime - $procPre.CpuTime
        $wallSec = $resp.WallTimeMs / 1000.0
        if ($wallSec -gt 0) {
            $cpuPercent = [math]::Round(($cpuDelta / $wallSec) * 100, 1)
        }
    }

    $result = @{
        Run        = $run
        TokPerSec  = $resp.TokPerSec
        Tokens     = $resp.CompletionTokens
        WallMs     = $resp.WallTimeMs
        CpuPercent = $cpuPercent
        RamMB      = if ($procPost) { $procPost.RamMB } else { 0 }
        GpuPercent = [math]::Max($gpuPre.GpuPercent, $gpuPost.GpuPercent)
        VramMB     = [math]::Max($gpuPre.VramUsedMB, $gpuPost.VramUsedMB)
    }
    $results += $result

    $cpuStr = if ($cpuPercent -gt 0) { "$($cpuPercent)%" } else { "n/a" }
    Write-Host ("  RUN {0}: {1,5:F1} tok/s | {2,4}ms | CPU: {3,6} | RAM: {4,6}MB | GPU: {5,3}% | VRAM: {6,5}MB" -f `
            $run, $result.TokPerSec, $result.WallMs, $cpuStr, $result.RamMB, $result.GpuPercent, $result.VramMB)
}

# Summary
Write-Host ""
Write-Host "--- Summary: $Engine ($Mode) ---" -ForegroundColor Green

$avgTokS = [math]::Round(($results | Measure-Object -Property TokPerSec -Average).Average, 1)
$avgCpu = [math]::Round(($results | Measure-Object -Property CpuPercent -Average).Average, 1)
$avgRam = [math]::Round(($results | Measure-Object -Property RamMB -Average).Average, 1)
$avgGpu = [math]::Round(($results | Measure-Object -Property GpuPercent -Average).Average, 0)
$avgVram = [math]::Round(($results | Measure-Object -Property VramMB -Average).Average, 0)

Write-Host "  Avg tok/s:  $avgTokS"
Write-Host "  Avg CPU%:   $avgCpu"
Write-Host "  Avg RAM:    $avgRam MB"
Write-Host "  Avg GPU%:   $avgGpu"
Write-Host "  Avg VRAM:   $avgVram MB"
Write-Host "  Load time:  $([math]::Round($loadTime))ms"
Write-Host ""

# Output CSV-friendly line
Write-Host "--- CSV ---" -ForegroundColor DarkGray
Write-Host "engine,mode,avg_tok_s,avg_cpu_pct,avg_ram_mb,avg_gpu_pct,avg_vram_mb,load_ms"
Write-Host "$Engine,$Mode,$avgTokS,$avgCpu,$avgRam,$avgGpu,$avgVram,$([math]::Round($loadTime))"

# Cleanup
if ($serverProc -and -not $serverProc.HasExited) {
    Write-Host ""
    Write-Host "[cleanup] Stopping $Engine server..." -ForegroundColor Yellow
    $serverProc | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
if ($Engine -eq "ollama" -and $Mode -eq "cpu") {
    Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}

# Clean up log files
Remove-Item ".\bench_stdout.log" -ErrorAction SilentlyContinue
Remove-Item ".\bench_stderr.log" -ErrorAction SilentlyContinue

Write-Host "[done] Benchmark complete." -ForegroundColor Green
