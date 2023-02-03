# winOS powershell profile ~/Documents/PowerShell/Microsoft.PowerShell_profile.ps1
oh-my-posh --init --shell pwsh --config ~/AppData/Local/Programs/oh-my-posh/themes/material.omp.json | Invoke-Expression
# Import-Module oh-my-posh
Import-Module -Name PowerColorLS

Set-Alias ls PowerColorLS

function MyPrint-Location {
    (Get-Location).Path
}
# New-Alias
Set-Alias pwd MyPrint-Location
