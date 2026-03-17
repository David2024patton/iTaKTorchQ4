---
description: General purpose turbo workflow - auto-runs all commands without requiring manual approval clicks
---
// turbo-all
Auto-runs ALL commands without user approval. This includes but is not limited to:

PowerShell: Get-ChildItem, Copy-Item, Move-Item, Remove-Item, Set-Content, Add-Content,
  New-Item, Rename-Item, Get-Content, Select-String, Invoke-WebRequest, Start-Process,
  Start-Sleep, Write-Host, ForEach-Object, Where-Object, Set-Location, Test-Path,
  Get-CimInstance, Stop-Process, Get-Process, Expand-Archive, Compress-Archive

Git: add, commit, push, pull, checkout, branch, merge, stash, log, diff, status, reset, tag
Go: build, test, run, mod, vet, fmt, generate, install
Python: python, pip, py_compile, pytest
Node: npm, npx, node
Docker: docker exec, docker logs, docker restart, docker build, docker compose
SSH/SCP: ssh, scp to any host (VPS, Skynet, etc.)
File ops: mkdir, rmdir, cat, head, tail, find, grep, chmod, chown
