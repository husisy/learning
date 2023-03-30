# gradio

1. link
   * [github](https://github.com/gradio-app/gradio)
   * [website](https://www.gradio.app/)
2. install
   * `pip instal gradio`

`python draft00.py`

```Python
# draft00.py
import gradio as gr
def greet(name):
    return "Hello " + name + "!"
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```
