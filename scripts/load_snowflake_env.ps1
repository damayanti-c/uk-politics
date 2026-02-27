param(
    [string]$EnvFilePath = "..\\..\\economics-module\\secrets\\snowflake\\.env"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$resolvedPath = [System.IO.Path]::GetFullPath((Join-Path $scriptDir $EnvFilePath))

if (-not (Test-Path $resolvedPath)) {
    throw "Snowflake env file not found: $resolvedPath"
}

Get-Content $resolvedPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or $line -notmatch "=") {
        return
    }

    $parts = $line -split "=", 2
    $key = $parts[0].Trim()
    $value = $parts[1].Trim()

    # Drop inline comments while preserving plain values.
    if ($value -match "\s+#") {
        $value = ($value -split "\s+#", 2)[0].Trim()
    }

    [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
}

Write-Host "Loaded Snowflake env vars from $resolvedPath into current process."
