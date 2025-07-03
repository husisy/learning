# https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/
import os
import dotenv
import httpx

import llama_index
import llama_index.core
import llama_index.core.llms
import llama_index.core.tools
import llama_index.llms.openai

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

llm = llama_index.llms.openai.OpenAI(api_key=ENV_CONFIG['OPENAI_API_KEY'], http_client=httpx.Client(proxy=ENV_CONFIG.get('OPENAI_PROXY', None)))

llm = llama_index.llms.openai.OpenAI() #model="gpt-4o-mini" model="gpt-3.5-turbo" (default)
x0 = llm.complete("William Shakespeare is ")
x0.text
print(x0)
llm.acomplete
llm.stream_complete
llm.astream_complete

messages = [
    llama_index.core.llms.ChatMessage(role="system", content="You are a helpful assistant."),
    llama_index.core.llms.ChatMessage(role="user", content="Tell me a joke."),
]
x0 = llm.chat(messages)
x0.message.role.value
x0.message.content

## multi-model
llm = llama_index.llms.openai.OpenAI(model="gpt-4o")
# https://www.flickr.com/photos/40143737@N02/4345417890
tmp0 = [llama_index.core.llms.ImageBlock(path="data/image.jpg"),
        llama_index.core.llms.TextBlock(text="Describe the image in a few sentences.")]
messages = [llama_index.core.llms.ChatMessage(role="user", blocks=tmp0)]
x0 = llm.chat(messages)
x0.message.content


## tool-calling
def generate_song(name: str, artist: str) -> dict[str, str]:
    """Generates a song with provided name and artist."""
    return {"name": name, "artist": artist}
tool = llama_index.core.tools.FunctionTool.from_defaults(fn=generate_song)
llm = llama_index.llms.openai.OpenAI(model="gpt-4o")
x0 = llm.predict_and_call([tool], "Pick a random song for me")
