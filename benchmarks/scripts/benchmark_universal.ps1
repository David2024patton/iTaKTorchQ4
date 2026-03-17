<#
.SYNOPSIS
    iTaK Torch Universal Benchmark Suite (Windows)
.DESCRIPTION
    Auto-detects hardware (CPU, GPU, iGPU) and available runners (iTaK Torch, Ollama).
    Runs comparative benchmarks for throughput and latency.
    Optimized for Workstations, Laptops, and Mini PCs.
.PARAMETER ModelPath
    Path to the GGUF model file for iTaK Torch.
.PARAMETER OllamaModel
    Ollama model tag (e.g., qwen2.5:0.5b).
.PARAMETER Iterations
    Number of test runs per configuration.
.PARAMETER OutputDir
    Where to save the markdown report.
#>
param(
    [string]$ModelPath = "E:\.agent\GOAgent\models\qwen2.5-0.5b-instruct-q4_k_m.gguf",
    [string]$OllamaModel = "qwen2.5:0.5b",
    [int]$Iterations = 3,
    [string]$OutputDir = "E:\.agent\iTaK Torch\benchmarks\results"
)

$ErrorActionPreference = "Continue"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$MachineName = $env:COMPUTERNAME
$ReportFile = Join-Path $OutputDir "benchmark_${Timestamp}_${MachineName}.md"

# Ensure output dir exists
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

# ============================================================
# 1. AUTO-DETECTION
# ============================================================

function Get-HardwareInfo {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    
    $gpus = @()
    $videoControllers = Get-CimInstance Win32_VideoController
    foreach ($vc in $videoControllers) {
        $type = "Unknown"
        if ($vc.Name -match "NVIDIA") { $type = "Discrete GPU" }
        elseif ($vc.Name -match "AMD" -and $vc.Name -match "Radeon") { $type = "Discrete/Integrated GPU" }
        elseif ($vc.Name -match "Intel" -or $vc.Name -match "UHD" -or $vc.Name -match "Iris") { $type = "Integrated GPU (iGPU)" }
        
        $gpus += @{
            Name = $vc.Name
            Type = $type
            VRAM = [math]::Round($vc.AdapterRAM / 1MB, 0) # Estimate
        }
    }

    return @{
        CPU      = $cpu.Name
        Cores    = $cpu.NumberOfCores
        Threads  = $cpu.NumberOfLogicalProcessors
        RAM      = "${ram}GB"
        GPUs     = $gpus
        IsMiniPC = ($cpu.NumberOfCores -le 6 -or $ram -le 16)
    }
}

$HW = Get-HardwareInfo

# Detect Runners
$HasTorch = $false
$TorchBin = "E:\.agent\GOAgent\itaktorch.exe" # Default location
if (Test-Path $TorchBin) { $HasTorch = $true }
elseif (Get-Command "itaktorch" -ErrorAction SilentlyContinue) { 
    $HasTorch = $true 
    $TorchBin = "itaktorch"
}

$HasOllama = [bool](Get-Command "ollama" -ErrorAction SilentlyContinue)

# ============================================================
# 2. REPORTING
# ============================================================

function Write-Report {
    param([string]$Text)
    Add-Content -Path $ReportFile -Value $Text
    Write-Host $Text
}

Write-Report "# iTaK Torch Universal Benchmark Report"
Write-Report ""
Write-Report "**Date:** $(Get-Date)"
Write-Report "**Machine:** $MachineName"
Write-Report "**CPU:** $($HW.CPU) ($($HW.Cores)C/$($HW.Threads)T)"
Write-Report "**RAM:** $($HW.RAM)"
if ($HW.IsMiniPC) { Write-Report "**Type:** Mini PC / Low Power Device Detected" }
Write-Report "**GPUs:**"
foreach ($g in $HW.GPUs) {
    Write-Report "- **$($g.Name)** ($($g.Type))"
}
Write-Report ""
Write-Report "---"
Write-Report ""

# ============================================================
# 3. BENCHMARKING LOGIC
# ============================================================

$Prompt = "Explain the concept of transformer attention mechanisms in neural networks. Include details about query, key, and value matrices."

function Run-OllamaBenchmark {
    if (-not $HasOllama) { return }
    
    Write-Report "## Ollama Benchmark ($OllamaModel)"
    Write-Report "| Run | Backend | Tok/s | Latency (ms) |"
    Write-Report "|---|---|---|---|"

    # Ollama auto-selects backend, but we can hint via OLLAMA_NUM_GPU environment var or similar if supported.
    # For now, we run standard inference which usually picks the best available GPU.
    
    for ($i = 1; $i -le $Iterations; $i++) {
        Write-Host "  Running Ollama iteration $i..." -ForegroundColor Cyan
        
        # Try to run interactively to capture output in variable for parsing
        try {
            $proc = Start-Process -FilePath "ollama" -ArgumentList "run $OllamaModel --verbose" -RedirectStandardInput (New-TemporaryFile | % { Set-Content $_ $Prompt; $_ }) -RedirectStandardOutput (New-TemporaryFile) -RedirectStandardError (New-TemporaryFile) -PassThru -NoNewWindow -Wait
            
            # Read stderr from the temp file used in Start-Process above
            # Note: We need to define temp files variables first to read them back
            
            # Better approach: use System.Diagnostics.Process directly for capture
            $pinfo = New-Object System.Diagnostics.ProcessStartInfo
            $pinfo.FileName = "ollama"
            $pinfo.Arguments = "run $OllamaModel --verbose"
            $pinfo.RedirectStandardInput = $true
            $pinfo.RedirectStandardError = $true
            $pinfo.RedirectStandardOutput = $true
            $pinfo.UseShellExecute = $false
            $pinfo.CreateNoWindow = $true
            
            $p = New-Object System.Diagnostics.Process
            $p.StartInfo = $pinfo
            $p.Start() | Out-Null
            
            $p.StandardInput.WriteLine($Prompt)
            $p.StandardInput.Close()
            
            $stdout = $p.StandardOutput.ReadToEnd()
            $stderr = $p.StandardError.ReadToEnd()
            $p.WaitForExit()
            
            $output = "$stdout`n$stderr"
            if ($output -match "eval rate:\s+([\d.]+)") { $tok_s = $matches[1] }
        }
        catch {}

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $sw.Stop()

        Write-Report "| $i | Auto | $tok_s | $($sw.ElapsedMilliseconds) |"
    }
    Write-Report ""
}

function Run-TorchBenchmark {
    if (-not $HasTorch) { 
        Write-Report "> **Note:** iTaK Torch binary not found at $TorchBin. Skipping."
        return 
    }

    Write-Report "## iTaK Torch Benchmark"
    Write-Report "| Run | Backend | Threads | Tok/s | Input |"
    Write-Report "|---|---|---|---|---|"

    # Define Backends to Test
    $Backends = @("cpu")
    foreach ($g in $HW.GPUs) {
        if ($g.Type -match "NVIDIA") { $Backends += "cuda" }
        if ($g.Type -match "Integrated") { $Backends += "vulkan" } # Use Vulkan for iGPU usually
    }

    foreach ($backend in $Backends) {
        for ($i = 1; $i -le $Iterations; $i++) {
            Write-Host "  Running Torch ($backend) iteration $i..." -ForegroundColor Green
            
            # Construct command args
            # Using 'bench' command if available, or 'run' with metrics
            $args = "bench --model `"$ModelPath`" --backend $backend --prompt `"$Prompt`""
            
            # Mini PC optimization override
            if ($HW.IsMiniPC) {
                $args += " --threads 4 --batch-size 512"
            }

            try {
                # Execute and capture
                $output = & $TorchBin $args.Split(" ") 2>&1
                
                # Parse output (hypothetical output format based on standard llama.cpp bench)
                $tok_s = "0"
                if ($output -match "eval time =.*=\s+([\d.]+)\s+T/s") { $tok_s = $matches[1] }
                elseif ($output -match "throughput:\s+([\d.]+)") { $tok_s = $matches[1] }
                
                Write-Report "| $i | $backend | Auto | $tok_s | -"
            }
            catch {
                Write-Report "| $i | $backend | Error | - | -"
                Write-Host "Error running torch: $_" -ForegroundColor Red
            }
        }
    }
    Write-Report ""
}

# Run All
Run-OllamaBenchmark
Run-TorchBenchmark

Write-Report "Benchmark complete. Results saved to: $ReportFile"
Write-Host "Done!" -ForegroundColor Green
