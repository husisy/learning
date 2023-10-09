# open intepreter

1. link
   * [github](https://github.com/KillianLucas/open-interpreter)
   * [website](https://openinterpreter.com/)
   * [github/litellm](https://github.com/BerriAI/litellm)
   * [github/ollama](https://github.com/jmorganca/ollama)

```bash
mamba create -y -n openinterpreter
mamba install -n openinterpreter openai python-dotenv tiktoken ipython jupyter_core jupyter_client scipy
mamba activate openinterpreter
pip install open-interpreter
```

```bash
# https://docs.litellm.ai/docs/providers/
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly

# environment variable
OPENAI_API_KEY=...

# ctrl-c to exit
```
