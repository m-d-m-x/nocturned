# Deploy nocturned WITH wake-word detection, plus the TFLite runtime and models.
#
#   powershell -File .\deploy-wakeword.ps1            # build in WSL, then deploy
#   powershell -File .\deploy-wakeword.ps1 -SkipBuild # deploy what is already staged
#   powershell -File .\deploy-wakeword.ps1 -WithTools # also install wakescore
#
# Two machines are involved on purpose. The build runs in WSL, where the arm
# cross-compiler, the TFLite build and the trained models live. The deploy runs
# here, because WSL2 has its own network namespace and cannot reach the device
# over USB networking. build-wakeword.sh bridges them by staging into
# dist-wakeword\, which both sides can see.

param(
    [switch]$SkipBuild,
    [switch]$WithTools
)

$ErrorActionPreference = "Stop"

$Key = "../id_rsa"
$Dev = "root@172.16.42.2"
$Dist = Join-Path $PSScriptRoot "dist-wakeword"
$ModelDir = "/etc/nocturne/wakeword"

function Invoke-Device([string]$Command, [string]$What) {
    ssh -i $Key -o StrictHostKeyChecking=no $Dev $Command
    if ($LASTEXITCODE -ne 0) { throw "failed: $What" }
}

if (-not $SkipBuild) {
    Write-Host "=== building in WSL ===" -ForegroundColor Cyan
    $script = "/mnt/d/Marshall_Files/MapleLab/Nocturne_Project/nocturned/build-wakeword.sh"
    # Strip CRLF: the file lives on a Windows filesystem but runs under bash.
    wsl -d Ubuntu -- bash -c "sed 's/\r`$//' $script > /tmp/bw.sh && bash /tmp/bw.sh"
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$Payload = @("nocturned", "libtensorflowlite_c.so", "melspectrogram.tflite",
             "embedding_model.tflite", "hey_spotify.tflite")
if ($WithTools) { $Payload += "wakescore" }

foreach ($f in $Payload) {
    if (-not (Test-Path (Join-Path $Dist $f))) {
        throw "missing $f in dist-wakeword - run without -SkipBuild"
    }
}

$NeedMB = [math]::Ceiling((($Payload | ForEach-Object {
    (Get-Item (Join-Path $Dist $_)).Length } | Measure-Object -Sum).Sum / 1MB)) + 2
$AvailMB = [int](ssh -i $Key -o StrictHostKeyChecking=no $Dev "df -Pm / | awk 'NR==2{print `$4}'")
Write-Host "=== space: need ~${NeedMB}M, device has ${AvailMB}M free ===" -ForegroundColor Cyan
if ($AvailMB -le $NeedMB) {
    throw "not enough space on the device rootfs (${AvailMB}M free, need ~${NeedMB}M)"
}

# Stop first: Linux refuses to write a running executable (ETXTBSY). The backup
# is the rollback if a copy dies midway.
Write-Host "=== preparing device ===" -ForegroundColor Cyan
Invoke-Device "mount -o remount,rw / && (sv stop nocturned || true) && cp -a /usr/sbin/nocturned /usr/sbin/nocturned.bak && mkdir -p $ModelDir" "prepare device"

function Restore-Device([string]$Reason) {
    Write-Warning "$Reason - rolling back"
    ssh -i $Key -o StrictHostKeyChecking=no $Dev `
        "cp -a /usr/sbin/nocturned.bak /usr/sbin/nocturned; sync; mount -o remount,ro /; sv start nocturned"
    throw "deploy aborted, device rolled back"
}

try {
    # The runtime and models go first: if the binary lands without them it will
    # start, fail to arm detection, and look like a code fault.
    scp -i $Key -o StrictHostKeyChecking=no (Join-Path $Dist "libtensorflowlite_c.so") "${Dev}:/usr/lib/"
    if ($LASTEXITCODE -ne 0) { Restore-Device "runtime copy failed" }

    foreach ($m in @("melspectrogram.tflite", "embedding_model.tflite", "hey_spotify.tflite")) {
        scp -i $Key -o StrictHostKeyChecking=no (Join-Path $Dist $m) "${Dev}:$ModelDir/"
        if ($LASTEXITCODE -ne 0) { Restore-Device "model copy failed ($m)" }
    }

    if ($WithTools) {
        scp -i $Key -o StrictHostKeyChecking=no (Join-Path $Dist "wakescore") "${Dev}:/usr/bin/wakescore"
        if ($LASTEXITCODE -ne 0) { Restore-Device "wakescore copy failed" }
    }

    scp -i $Key -o StrictHostKeyChecking=no (Join-Path $Dist "nocturned") "${Dev}:/usr/sbin/nocturned"
    if ($LASTEXITCODE -ne 0) { Restore-Device "daemon copy failed" }
}
catch {
    Restore-Device $_.Exception.Message
}

Invoke-Device "chmod 755 /usr/sbin/nocturned; [ -f /usr/bin/wakescore ] && chmod 755 /usr/bin/wakescore; ldconfig 2>/dev/null || true; sync; mount -o remount,ro /; sv start nocturned" "restart nocturned"

Start-Sleep -Seconds 3
Write-Host "=== status ===" -ForegroundColor Cyan
ssh -i $Key -o StrictHostKeyChecking=no $Dev `
    "sv status nocturned; echo; echo -n '  wake endpoint: '; curl -s localhost:5000/audio/transcribe/wake; echo; echo -n '  rootfs free: '; df -Pm / | tail -1 | tr -s ' ' | cut -d' ' -f4"

Write-Host ""
Write-Host "Arm detection with:" -ForegroundColor Green
Write-Host '  curl -s -X POST localhost:5000/audio/transcribe/wake -d ''{"enabled":true,"provider":"groq","apiKey":"<KEY>"}'''
