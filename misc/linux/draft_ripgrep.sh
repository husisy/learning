# tutorial https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md
# https://github.com/BurntSushi/ripgrep/archive/0.7.1.zip

# basic: single file
rg fast README.md
rg 'fast\w*' README.md
rg 'fast\w+' README.md

rg 'fast\w*' --context README.md
rg 'fast\w*' -C 10 README.md

# basic: directory
rg 'fn write\(' .
rg 'fn write\('
rg -F 'fn write('

rg 'fn write\(' src

# basic globs
rg clap -g '*.toml'
rg clap -g '!*.toml'

rg 'fn run' -g '*.rs'
rg 'fn run' --type rust
rg 'fn run' -trust

rg clap --type-not rust
rg clap -Trust

rg 'int main' -g '*.{c,h}'
rg 'int main' -tc

rg --type-list

# basic replace
rg fast README.md --replace FAST

rg 'fast\s+(\w+)' README.md -r 'fast-$1'
rg 'fast\s+(?P<word>\w+)' README.md -r 'fast-$word'
