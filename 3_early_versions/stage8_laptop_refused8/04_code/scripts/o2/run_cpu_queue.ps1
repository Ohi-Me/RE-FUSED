# O2 CPU queue (resumable): LightGBM quantile models for all targets.
Set-Location (Join-Path $PSScriptRoot "..\..\..")
$env:PYTHONIOENCODING = "utf-8"
$log = "12_logs\o2_cpu_queue.log"
"CPU queue start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o2\o2_02_lgbm.py T1 T2 T3 T4 T5 *>> $log
"CPU_QUEUE_DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
