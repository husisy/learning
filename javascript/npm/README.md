# node.js/npm

## nvm

1. link
   * [github/nvm](https://github.com/nvm-sh/nvm)
2. install
   * `brew install nvm`

```bash
# ~/.zshrc
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"  # This loads nvm
[ -s "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" ] && \. "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm"  # This loads nvm bash_completion
```

```bash
nvm install 16
nvm use 16
node -v
nvm install node #latest
nvm ls-remote
```

## node

1. 执行`node`进入交互式命令行
   * `.exit`
   * `.help`
2. install

```bash
npm install #
# npm run xxx
```
