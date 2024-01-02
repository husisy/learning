import os
import dotenv
import gradio

import litellm

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']


def inference(message, history):
    try:
        import pickle
        with open('tbd00.pkl', 'wb') as fid:
            pickle.dump(dict(history=history, message=message), fid)
        messages = [dict(role=y0, content=y1) for x in history for y0,y1 in zip(['user','assistant'], x)]
        messages.append(dict(role='user', content=message))
        partial_message = ""
        for chunk in litellm.completion(model="gpt-3.5-turbo-1106",
                    messages=messages, max_tokens=512,
                    temperature=.7, top_p=.9, stream=True):
            if chunk.choices[0].finish_reason!='stop':
                partial_message += chunk['choices'][0]['delta']['content']
                yield partial_message
    except Exception as e:
        print("Exception encountered:", str(e))
        yield f"An Error occured please 'Clear' the error and try your question again"

model_name = 'xxx'

gradio.ChatInterface(
    inference,
    chatbot=gradio.Chatbot(height=400),
    textbox=gradio.Textbox(placeholder="Enter text here...", container=False, scale=5),
    description=f"""
    CURRENT PROMPT TEMPLATE: {model_name}.
    An incorrect prompt template will cause performance to suffer.
    Check the API specifications to ensure this format matches the target LLM.""",
    title="Simple Chatbot Test Application",
    examples=["Define 'deep learning' in once sentence.", "what's the relationship between large language model and deep learning?"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).queue().launch()
