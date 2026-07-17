param(
    [string]$PythonFile,
    [string[]]$ScriptArgs = @(),
    [string]$BlenderExe = ""
)
$blender = $BlenderExe
if (-not $blender) { $blender = $env:TGEL_BLENDER_EXE }
if (-not $blender) { $blender = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" }
if (-not (Test-Path $blender)) { Write-Error "Pinned Blender not found: $blender"; exit 2 }
$argv = @("--background", "--factory-startup", "--python-exit-code", "1")
if ($PythonFile) { $argv += @("--python", $PythonFile) }
if ($ScriptArgs.Count -gt 0) { $argv += "--"; $argv += $ScriptArgs }
& $blender @argv
$blenderExitCode = $LASTEXITCODE
exit $blenderExitCode
