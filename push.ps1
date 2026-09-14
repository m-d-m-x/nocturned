# Build nocturned for the Car Thing (armv7) and deploy it over USB networking.
# If you ever lose connection: New-NetIPAddress -InterfaceAlias "Ethernet 3" -IPAddress 172.16.42.1 -PrefixLength 24

$ErrorActionPreference = "Stop"

$Key = "../id_rsa"
$Dev = "root@172.16.42.2"

# armv7, static. Matches XBPS_ARCH=armv7l in the image builder - NOT the arm64
# in .github/workflows/build.yml, which does not match the rootfs.
$env:GOOS = "linux"; $env:GOARCH = "arm"; $env:GOARM = "7"; $env:CGO_ENABLED = "0"
try {
    go build -ldflags "-s -w" -o nocturned .
    if ($LASTEXITCODE -ne 0) { throw "go build failed" }
}
finally {
    # Don't leave the shell cross-compiling for every later `go` command.
    Remove-Item Env:GOOS, Env:GOARCH, Env:GOARM, Env:CGO_ENABLED -ErrorAction SilentlyContinue
}

# Open the rootfs and stop the service first: Linux refuses to write a running
# executable (ETXTBSY), and the backup is the rollback if the copy dies midway.
ssh -i $Key $Dev "mount -o remount,rw / && sv stop nocturned && cp -a /usr/sbin/nocturned /usr/sbin/nocturned.bak"
if ($LASTEXITCODE -ne 0) { throw "failed to prepare device" }

scp -i $Key ./nocturned "${Dev}:/usr/sbin/nocturned"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "scp failed - restoring backup"
    ssh -i $Key $Dev "cp -a /usr/sbin/nocturned.bak /usr/sbin/nocturned && sync && mount -o remount,ro / && sv start nocturned"
    throw "deploy aborted, device rolled back"
}

ssh -i $Key $Dev "chmod 755 /usr/sbin/nocturned && sync && mount -o remount,ro / && sv start nocturned"
if ($LASTEXITCODE -ne 0) { throw "failed to restart nocturned" }

Start-Sleep -Seconds 2
ssh -i $Key $Dev "sv status nocturned; curl -s localhost:5000/info; echo"
