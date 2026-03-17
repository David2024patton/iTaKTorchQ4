# iTaK Torch Setup Script (Run As Administrator)
# Purpose: One-time setup for iTaK Torch on Windows. Must be run elevated.
# Creates a Windows Firewall allow rule so iTaK Torch never triggers a popup.

param(
    [string]$iTaK TorchPath = "E:\.agent\iTaK Agent\itaktorch.exe"
)

$ErrorActionPreference = "Stop"

# Check if running as admin.
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] This script must be run as Administrator." -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as administrator', then run this script again."
    exit 1
}

# Remove any existing iTaK Torch firewall rules.
$existing = netsh advfirewall firewall show rule name="iTaK Torch Inference Engine" 2>$null
if ($existing -match "iTaK Torch") {
    Write-Host "[INFO] Removing existing firewall rule..." -ForegroundColor Yellow
    netsh advfirewall firewall delete rule name="iTaK Torch Inference Engine" | Out-Null
}

# Add inbound allow rule for the iTaK Torch binary.
Write-Host "[INFO] Adding firewall rule for: $iTaK TorchPath" -ForegroundColor Cyan
netsh advfirewall firewall add rule `
    name="iTaK Torch Inference Engine" `
    dir=in `
    action=allow `
    program="$iTaK TorchPath" `
    enable=yes `
    profile=any | Out-Null

Write-Host "[PASS] Firewall rule added. iTaK Torch will no longer trigger firewall prompts." -ForegroundColor Green

# Also add outbound rule (for model downloads).
netsh advfirewall firewall add rule `
    name="iTaK Torch Outbound" `
    dir=out `
    action=allow `
    program="$iTaK TorchPath" `
    enable=yes `
    profile=any | Out-Null

Write-Host "[PASS] Outbound rule added (for model downloads)." -ForegroundColor Green

# Also add itakagent.exe if it exists.
$itakagentPath = Join-Path (Split-Path $iTaK TorchPath -Parent) "itakagent.exe"
if (Test-Path $itakagentPath) {
    $existing2 = netsh advfirewall firewall show rule name="iTaK Agent Framework" 2>$null
    if (-not ($existing2 -match "iTaK Agent")) {
        netsh advfirewall firewall add rule `
            name="iTaK Agent Framework" `
            dir=in `
            action=allow `
            program="$itakagentPath" `
            enable=yes `
            profile=any | Out-Null
        netsh advfirewall firewall add rule `
            name="iTaK Agent Outbound" `
            dir=out `
            action=allow `
            program="$itakagentPath" `
            enable=yes `
            profile=any | Out-Null
        Write-Host "[PASS] iTaK Agent firewall rules added." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Setup complete. iTaK Torch and iTaK Agent will not trigger Windows Firewall." -ForegroundColor Green
