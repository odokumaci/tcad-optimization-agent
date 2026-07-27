# Activate DEVSIM environment
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MklBin = Join-Path $ProjectRoot ".venv\Library\bin"

$VenvScripts = Join-Path $ProjectRoot ".venv\Scripts"
$env:PATH = "$VenvScripts;$MklBin;$env:PATH"
$env:DEVSIM_MATH_LIBS = "mkl_rt.3.dll"

Write-Host "DEVSIM environment ready."
Write-Host "Python: $VenvPython"
Write-Host "Run examples: cd .venv\devsim_data\testing; python cap2.py"
