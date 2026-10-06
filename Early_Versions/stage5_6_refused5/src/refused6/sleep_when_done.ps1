# RE-FUSED-6 sleep watchdog.
# Puts the PC to sleep ONLY when:
#   (a) the model writes results\refused6\_FINAL_PACKAGE_DONE.txt (the whole program is finished), or
#   (b) safety net: all compute has finished, no Python process is running, and the model session has
#       been silent (no heartbeat) for 3 hours - e.g. the session hit a usage limit overnight.
# It changes no settings; it only issues a normal Sleep at the end.
$root   = 'D:\\REFUSED5\results'
$status = Join-Path $root 'refused6\PIPELINE_STATUS.md'
$final  = Join-Path $root 'refused6\_FINAL_PACKAGE_DONE.txt'
$hb     = Join-Path $root 'refused6\_model_heartbeat.txt'

function Log($m) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $status -Value ("- ``" + $ts + "`` [sleep-watchdog] " + $m) -Encoding utf8
}

Log 'started: sleeps the PC when _FINAL_PACKAGE_DONE.txt appears, or 3 h after compute ends with no activity'
while ($true) {
    Start-Sleep -Seconds 300
    $computeDone = Test-Path (Join-Path $root '_allstats_done.txt')
    $pyCount     = @(Get-CimInstance Win32_Process -Filter "name='python.exe'").Count
    $hbAgeMin    = if (Test-Path $hb) { ((Get-Date) - (Get-Item $hb).LastWriteTime).TotalMinutes } else { 99999 }
    if (Test-Path $final) {
        Log 'final package complete - putting the PC to sleep'
        break
    }
    if ($computeDone -and $pyCount -eq 0 -and $hbAgeMin -gt 180) {
        Log ("safety net: compute finished, no Python running, model silent for {0:N0} min - putting the PC to sleep" -f $hbAgeMin)
        break
    }
}
Start-Sleep -Seconds 30
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::SetSuspendState([System.Windows.Forms.PowerState]::Suspend, $false, $false) | Out-Null
