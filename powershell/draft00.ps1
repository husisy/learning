# file system
Set-Location #alias[cd]
Get-Location #alias[pwd]
Get-ChildItem #alias[ls] alias[dir]
Get-ChildItem -Path *.py -Recurse
New-Item -Path tbd00.txt #alias[ni]
New-Item -Path tbd00 -ItemType Directory #default -ItemType=File
New-Item -Path tbd01.txt -Value 2333
New-Item -Path tbd02.txt -Value "hello world"
Get-Content -Path tbd01.txt #alias[type]
Set-Content -Path tbd02.txt -Value "world hello"
"2333" > tbd03.txt #create file if not exist, replace if exist
Remove-Item -Path tbd03.txt #alias[del] alias[rm] alias[rmdir] alias[rd]


# Help
Get-Alias
Get-Command #alias[gcm]
help -Name Get-Children #alias[man]
Get-Command help, Get-Help #help不是Get-Help的Alias，更建议使用help

Get-ChildItem -?
Get-Help -Name Get-ChildItem
Get-Help -Name Get-ChildItem -Example #-Full -Detailed
Get-Help -Name Get-ChildItem -Parameter *


# misc
Clear-Host #shortcut[ctrl+l] alias[cls] alias[clear]


# pipeline
Get-ChildItem | Sort-Object -Descending
Get-ChildItem | Sort-Object -Descending -Property Length | Select-Object -First 1


# process system
Get-Process -Name pwsh #alias[ps]
Get-Process -Name chrome
Get-Process chrome
Get-Process pwsh, chrome

# variable
Write-Output $PSHOME #alias[echo]
$PSHOME
$home
$PSVersionTable
