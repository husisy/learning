import os
import openai
import dotenv
import tiktoken

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class NaiveChatGPT:
    def __init__(self) -> None:
        self.message_list = [{"role": "system", "content": "You are a helpful assistant."},]
        self.response = None #for debug only

    def chat(self, message='', reset=False, tag_print=True, tag_return=False):
        if reset:
            self.message_list = self.message_list[:1]
        message = str(message)
        if message: #skip if empty
            self.message_list.append({"role": "user", "content": str(message)})
            self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list)
            tmp0 = self.response.choices[0].message.content
            self.message_list.append({"role": "assistant", "content": tmp0})
            if tag_print:
                print(tmp0)
            if tag_return:
                return tmp0

chatgpt = NaiveChatGPT()

def get_gpt35turbo_num_token(message_list):
    # https://platform.openai.com/docs/guides/chat/introduction
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
    num_tokens = 0
    for message in message_list:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


prompt_template = """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:"""
animal = 'horse'
tmp0 = prompt_template.format(animal.capitalize())
# text-davinci-003
response = openai.Completion.create(model="text-davinci-003", prompt=tmp0, temperature=0.6)
print(response.choices[0].text) #' Steed of Justice, Mighty Mare, The Noble Stallion'
chatgpt.chat(tmp0, reset=True) #Thunderhoof, Equine Avenger, Galloping Guardian



# demo a basic chatgpt
tmp0 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"},
]
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=tmp0)
response.choices[0].message.role #asisstant
response.choices[0].message.content
# 'The 2020 World Series was held at Globe Life Field in Arlington, Texas, which is the home ballpark of the Texas Rangers.'



text = 'The 2020 World Series was held at Globe Life Field in Arlington, Texas, which is the home ballpark of the Texas Rangers.'
tmp0 = f'Translate the following English text to Chinese: {text}'
chatgpt.chat(tmp0, reset=True)

chatgpt.chat('design a one-day trip in Hong Kong', reset=True)
