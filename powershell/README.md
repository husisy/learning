# powershell

1. link
   * [github-powershell](https://github.com/PowerShell/PowerShell)
   * [MS-documentation](https://docs.microsoft.com/en-us/powershell/scripting/overview?view=powershell-6)
   * [powershell-gallery](https://www.powershellgallery.com/)
2. concept: cmdlet
3. 文档中`[]`表示可选，例如`Get-ChildItem [[-Path] <string>]`表示`-Path`是可缺省地，同时`-Path <string>`是可缺省地（默认当前路径）
4. 偏见
   * 在scripts中**禁止**使用alias
5. `$PROFILE`

```powershell
Update-Help
Get-Help
Set-ExecutionPolicy[PSITPro5_Security]
Enable-PSRemoting
```

## oh-my-posh

1. link
   * [github/oh-my-posh](https://github.com/JanDeDobbeleer/oh-my-posh)
2. remember to install Nerd Fonts

```bash
winget install JanDeDobbeleer.OhMyPosh
```

```bash
oh-my-posh --init --shell pwsh --config ~/AppData/Local/Programs/oh-my-posh/themes/zash.omp.json | Invoke-Expression
# Import-Module oh-my-posh
Import-Module -Name PowerColorLS

Set-Alias ls PowerColorLS
function MyPrint-Location {
    (Get-Location).Path
}
Set-Alias pwd MyPrint-Location
```

use

```bash
Install-Module -Name Terminal-Icons
Install-Module -Name PowerColorLS
Import-Module -Name PowerColorLS
```
