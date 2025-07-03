import enum
import json
import httpx
import dotenv
import openai
import typing
import pydantic

from utils import hf_text_data


OPENAI_API_KEY = dotenv.dotenv_values('.env')['OPENAI_API_KEY']
OPENAI_PROXY = dotenv.dotenv_values('.env').get('OPENAI_PROXY', None)
client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(proxy=OPENAI_PROXY))


## structured output
class SchemaCalendarEvent(pydantic.BaseModel):
    name: str
    date: str
    participants: list[str]
prompt = [{"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}]
response = client.responses.parse(
    model="gpt-4.1",
    input=prompt,
    text_format=SchemaCalendarEvent,
)
x0 = response.output_parsed
# import rich; rich.print(x0)

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "date": {"type": "string"},
        "participants": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "date", "participants"],
    "additionalProperties": False
}
response = client.responses.create(
    model="gpt-4.1",
    input=prompt,
    text={"format":{"type": "json_schema", "name": "calendar_event", "schema": schema, "strict": True}}
)
x1 = json.loads(response.output_text)


## structured output (chain of thought)
class SchemaStep(pydantic.BaseModel):
    explanation: str
    output: str
class SchemaMathReasoning(pydantic.BaseModel):
    steps: list[SchemaStep]
    final_answer: str
prompt = [{"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},]
response = client.responses.parse(model="gpt-4.1", input=prompt, text_format=SchemaMathReasoning)
x0 = response.output_parsed


# structure data extraction
class SchemaResearchPaper(pydantic.BaseModel):
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
prompt = [{"role": "system", "content": "You are an expert at structured data extraction. "
                        "You will be given unstructured text from a research paper and should convert it into the given structure."},
        {"role": "user", "content": hf_text_data('data00.txt')}] #any arxiv webpage
response = client.responses.parse(model='gpt-4.1', input=prompt, text_format=SchemaResearchPaper)
x0 = response.output_parsed


## UI generation
class SchemaUIType(str, enum.Enum):
    div = "div"
    button = "button"
    header = "header"
    section = "section"
    field = "field"
    form = "form"
class SchemaAttribute(pydantic.BaseModel):
    name: str
    value: str
class SchemaUIComponent(pydantic.BaseModel):
    type: SchemaUIType
    label: str
    children: list["SchemaUIComponent"]
    attribute: list[SchemaAttribute]
SchemaUIComponent.model_rebuild() #required to enable recursive references
prompt = [{'role':'system', 'content': 'You are a UI generator AI. Convert the user input into a UI.'},
          {'role':'user', 'content': 'Make a User Profile Form'}]
response = client.responses.parse(model="gpt-4.1", input=prompt, text_format=SchemaUIComponent)
x0 = response.output_parsed


## moderation
class SchemaContentCategory(str, enum.Enum):
    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"
class SchemaContentCompliance(pydantic.BaseModel):
    is_violating: bool
    category: typing.Optional[SchemaContentCategory]
    explanation_if_violating: typing.Optional[str]
prompt = [{"role": "system", "content": "Determine if the user input violates specific guidelines and explain if they do."},
        {"role": "user", "content": "How do I prepare for a job interview?"}]
response = client.responses.parse(model="gpt-4.1", input=prompt, text_format=SchemaContentCompliance)
x0 = response.output_parsed
