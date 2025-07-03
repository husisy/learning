import os
import random
import uuid
import httpx
import dotenv
import asyncio
import enum
import pydantic
import numpy as np
import openai
import agents
import agents.extensions.handoff_prompt

hf_data = lambda *x: os.path.join('data', *x)
np_rng = np.random.default_rng()

# dotenv.load_dotenv('.env') #OPENAI_API_KEY
OPENAI_API_KEY = dotenv.dotenv_values('.env')['OPENAI_API_KEY']
OPENAI_PROXY = dotenv.dotenv_values('.env').get('OPENAI_PROXY', None)
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=httpx.AsyncClient(proxy=OPENAI_PROXY))
agents.set_default_openai_client(client)


## https://openai.github.io/openai-agents-python/#hello-world-example
agent00 = agents.Agent(name="Assistant", instructions="You are a helpful assistant")
# result = await agents.Runner.run(agent00, "Write a haiku about recursion in programming.")
result = asyncio.run(agents.Runner.run(agent00, "Write a haiku about recursion in programming."))
# result = agents.Runner.run_sync(agent00, "Write a haiku about recursion in programming.")
result.final_output


## https://openai.github.io/openai-agents-python/quickstart/
prompt = "You provide assistance with historical queries. Explain important events and context clearly."
history_tutor_agent = agents.Agent(name="History Tutor", handoff_description="Specialist agent for historical questions", instructions=prompt)
prompt = "You provide help with math problems. Explain your reasoning at each step and include examples"
math_tutor_agent = agents.Agent(name="Math Tutor", handoff_description="Specialist agent for math questions", instructions=prompt)
prompt = "You determine which agent to use based on the user's homework question"
triage_agent = agents.Agent(name="Triage Agent", instructions=prompt, handoffs=[history_tutor_agent, math_tutor_agent])
result = asyncio.run(agents.Runner.run(triage_agent, "What is the capital of France?"))
result.final_output

class SchemaHomework(pydantic.BaseModel):
    is_homework: bool
    reasoning: str
prompt = "Check if the user is asking about homework."
guardrail_agent = agents.Agent(name="Guardrail check", instructions=prompt, output_type=SchemaHomework)

async def homework_guardrail(ctx, agent, input_data):
    result = await agents.Runner.run(guardrail_agent, input_data, context=ctx.context)
    x0 = result.final_output_as(SchemaHomework)
    ret = agents.GuardrailFunctionOutput(output_info=x0, tripwire_triggered=(not x0.is_homework))
    return ret

prompt = "You determine which agent to use based on the user's homework question"
triage_agent = agents.Agent(name="Triage Agent", instructions=prompt,
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[agents.InputGuardrail(guardrail_function=homework_guardrail)],
)
result0 = asyncio.run(agents.Runner.run(triage_agent, "who was the first president of the united states?"))
result0.final_output
try:
    result1 = asyncio.run(agents.Runner.run(triage_agent, "what is life"))
except agents.InputGuardrailTripwireTriggered as e:
    result1 = e.guardrail_result
    print(f'Guardrail tripwire triggered: "{result1.output.output_info.reasoning}"')




### https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/deterministic.py
"""This example demonstrates a deterministic flow, where each step is performed by an agent."""
class OutlineCheckerOutput(pydantic.BaseModel):
    good_quality: bool
    is_scifi: bool
prompt = "Generate a very short story outline based on the user's input."
story_outline_agent = agents.Agent(name="story_outline_agent", instructions=prompt)
prompt = "Read the given story outline, and judge the quality. Also, determine if it is a scifi story."
outline_checker_agent = agents.Agent(name="outline_checker_agent", instructions=prompt, output_type=OutlineCheckerOutput)
prompt = "Write a short story based on the given outline."
story_agent = agents.Agent(name="story_agent", instructions=prompt, output_type=str)

async def run_story_flow(prompt:str):
    # Ensure the entire workflow is a single trace
    with agents.trace("Deterministic story flow"):
        outline_result = await agents.Runner.run(story_outline_agent, prompt)
        outline_checker_result:OutlineCheckerOutput = (await agents.Runner.run(outline_checker_agent, outline_result.final_output)).final_output
        if outline_checker_result.good_quality and outline_checker_result.is_scifi:
            ret = await agents.Runner.run(story_agent, outline_result)
        else:
            ret = outline_checker_result
    return ret

prompt_list = '''
A science fiction story set in the distant future
A fantasy adventure with dragons and magic
A cozy mystery in a small town
A romantic comedy with a happy ending
A psychological thriller with a twist
A historical drama set in ancient Rome
A horror story that's not too scary
A detective noir with a clever protagonist
'''.strip().split('\n')
result0 = asyncio.run(run_story_flow(prompt_list[3])) #should fail
result1 = asyncio.run(run_story_flow(prompt_list[0]))
result1.final_output


### https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py
"""
This example shows the handoffs/routing pattern. The triage agent receives the first message, and
then hands off to the appropriate agent based on the language of the request. Responses are
streamed to the user.
"""
french_agent = agents.Agent(name="french_agent", instructions="You only speak French")
spanish_agent = agents.Agent(name="spanish_agent", instructions="You only speak Spanish")
english_agent = agents.Agent(name="english_agent", instructions="You only speak English")
prompt = "Handoff to the appropriate agent based on the language of the request."
triage_agent = agents.Agent(name="triage_agent", instructions=prompt, handoffs=[french_agent, spanish_agent, english_agent])

_conversation_history = dict()
async def hf_conversation(conversationID:str, prompt:str):
    # conversationID is linked to (conversation history, agent, agents.trace)
    if conversationID in _conversation_history:
        agent,message_list = _conversation_history[conversationID]
        message_list.append({"content": prompt, "role": "user"})
    else:
        agent = triage_agent
        message_list = [{'role': 'assistant', 'content': "Hi! We speak French, Spanish and English. How can I help?"},
            {"role": "user", "content": prompt}]

    # Each conversation turn is a single trace. Normally, each input from the user would be an
    # API request to your app, and you can wrap the request in a trace()
    with agents.trace("Routing example", group_id=conversationID):
        result = agents.Runner.run_streamed(agent, input=message_list)
        async for event in result.stream_events():
            if not isinstance(event, agents.RawResponsesStreamEvent):
                continue
            data = event.data
            if isinstance(data, openai.types.responses.ResponseTextDeltaEvent):
                print(data.delta, end="", flush=True)
            elif isinstance(data, openai.types.responses.ResponseContentPartDoneEvent):
                print("\n")
    _conversation_history[conversationID] = result.current_agent, result.to_input_list()

prompt_list = '''
Thank you! I'd prefer to continue in English, please. I need some help with a booking.
Bonjour, merci! Je préfère parler en français. J'ai une question concernant votre service.
¡Gracias! Me gustaría continuar en español. ¿Puede ayudarme con un problema que tengo?
'''.strip().split('\n')


print("Hi! We speak French, Spanish and English. How can I help? ") #begining of the conversation
conversationID = uuid.uuid4().hex[:16]
asyncio.run(hf_conversation(conversationID, prompt_list[0]))

prompt = "I'm trying to change the date of my reservation—can you help with that?"
asyncio.run(hf_conversation(conversationID, prompt))

'''
assistent: Hi! We speak French, Spanish and English. How can I help?
user: Thank you! I'd prefer to continue in English, please. I need some help with a booking.
assistent: Sure! What do you need help with regarding your booking?
user: I'm trying to change the date of my reservation—can you help with that?
assistent: I can guide you through the process. Could you let me know where you made the reservation?
'''


### https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/agents_as_tools.py
"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation agents.
"""
spanish_agent = agents.Agent(name="spanish_agent", instructions="You translate the user's message to Spanish", handoff_description="An english to spanish translator")
french_agent = agents.Agent(name="french_agent", instructions="You translate the user's message to French", handoff_description="An english to french translator")
italian_agent = agents.Agent(name="italian_agent", instructions="You translate the user's message to Italian", handoff_description="An english to italian translator")
prompt = ("You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools.")
tmp0 = [spanish_agent.as_tool(tool_name="translate_to_spanish", tool_description="Translate the user's message to Spanish"),
        french_agent.as_tool(tool_name="translate_to_french", tool_description="Translate the user's message to French"),
        italian_agent.as_tool(tool_name="translate_to_italian", tool_description="Translate the user's message to Italian")]
orchestrator_agent = agents.Agent(name="orchestrator_agent", instructions=prompt, tools=tmp0)
synthesizer_agent = agents.Agent(name="synthesizer_agent", instructions="You inspect translations, correct them if needed, and produce a final concatenated response.")

async def hf_conversation(prompt:str):
    # Run the entire orchestration in a single trace
    with agents.trace("Orchestrator evaluator"):
        result = await agents.Runner.run(orchestrator_agent, prompt)
        for x0 in result.new_items:
            if isinstance(x0, agents.MessageOutputItem):
                text = agents.ItemHelpers.text_message_output(x0)
                if text:
                    print(f"  - Translation step: {text}")
        synthesizer_result = await agents.Runner.run(synthesizer_agent, result.to_input_list())
    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")

prompt_list = '''
Can you translate "Good morning, how are you?" into Spanish?
Please translate "I would like to order a coffee" into French.
Translate the phrase "Where is the nearest train station?" into Italian.
Can you translate "Hello, my name is Anna" into French and Spanish?
I want to see how this sentence sounds in both Spanish and Italian: "Do you have any recommendations?"
'''.strip().split('\n')
print("assistent: Hi! What would you like translated, and to which languages? ") #begining of the conversation
prompt = prompt_list[4]
print(f"user: {prompt}")
asyncio.run(hf_conversation(prompt))


### https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py
"""
This example shows the LLM as a judge pattern. The first agent generates an outline for a story.
The second agent judges the outline and provides feedback. We loop until the judge is satisfied
with the outline.
"""
class SchemaScore(str, enum.Enum):
    pass_ = 'pass'
    needs_improvement = "needs_improvement"
    fail = "fail"
class SchemaEvaluationFeedback(pydantic.BaseModel):
    feedback: str
    score: SchemaScore

prompt = "You generate a very short story outline based on the user's input.If there is any feedback provided, use it to improve the outline."
story_outline_generator = agents.Agent(name="story_outline_generator", instructions=prompt)
prompt = ("You evaluate a story outline and decide if it's good enough."
        "If it's not good enough, you provide feedback on what needs to be improved. Never give it a pass on the first try."
        "If the outline is good, you give it a pass. You must pass or fail the outline within 3 tries")
evaluator = agents.Agent(name="evaluator", instructions=prompt, output_type=SchemaEvaluationFeedback)

async def hf_conversation(prompt:str, max_try:int=3):
    message_list = [{"content": prompt, "role": "user"}]
    with agents.trace("LLM as a judge"):
        for _ in range(max_try):
            result = await agents.Runner.run(story_outline_generator, message_list)
            message_list = result.to_input_list()
            evaluator_result = await agents.Runner.run(evaluator, message_list)
            result_eval = evaluator_result.final_output
            print(f"Evaluator score: {result_eval.score}")
            print('Feedback:', result_eval.feedback)
            if result_eval.score == SchemaScore.pass_:
                break
            message_list.append({"content": f"Feedback: {result_eval.feedback}", "role": "user"})
    print(f"Final story outline: {result.final_output}")
    return result

prompt_list = '''
A science fiction story set in the distant future
A fantasy adventure with dragons and magic
A cozy mystery in a small town
A romantic comedy with a happy ending
A psychological thriller with a twist
A historical drama set in ancient Rome
A horror story that's not too scary
A detective noir with a clever protagonist
'''.strip().split('\n')
print("What kind of story would you like to hear? ") #begining of the conversation
prompt = prompt_list[2]
result = asyncio.run(hf_conversation(prompt))


### https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/input_guardrails.py
"""
In this example, we'll setup an input guardrail that trips if the user is asking to do math homework.
If the guardrail trips, we'll respond with a refusal message.
"""
class SchemaMathHomeworkOutput(pydantic.BaseModel):
    reasoning: str
    is_math_homework: bool

prompt = "Check if the user is asking you to do their math homework."
guardrail_agent = agents.Agent(name="Guardrail check", instructions=prompt, output_type=SchemaMathHomeworkOutput)

@agents.input_guardrail
async def math_guardrail(context: agents.RunContextWrapper, agent:agents.Agent, message_list:str|list[agents.TResponseInputItem]) -> agents.GuardrailFunctionOutput:
    result = await agents.Runner.run(guardrail_agent, message_list, context=context.context)
    final_output = result.final_output_as(SchemaMathHomeworkOutput)
    ret = agents.GuardrailFunctionOutput(output_info=final_output, tripwire_triggered=final_output.is_math_homework)
    return ret
prompt = "You are a customer support agent. You help customers with their questions."
agent00 = agents.Agent(name="Customer support agent", instructions=prompt, input_guardrails=[math_guardrail])

async def hf_conversation(prompt:str):
    message_list = [{"role": "user", "content": prompt}]
    try:
        result = await agents.Runner.run(agent00, message_list)
        print(result.final_output)
        message_list = result.to_input_list()
    except agents.InputGuardrailTripwireTriggered as e:
        print('reasoning:', e.guardrail_result.output.output_info.reasoning)
        tmp0 = "Sorry, I can't help you with your math homework."
        print(tmp0)
        message_list.append({"role": "assistant", "content": tmp0})
    return message_list

asyncio.run(hf_conversation("What's the capital of California?"))
asyncio.run(hf_conversation("Can you help me solve for x: 2x + 5 = 11"))


### https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/output_guardrails.py
"""This example shows how to use output guardrails: use a (contrived) example where we check if the agent's response contains a phone number."""
class SchemaMessageOutput(pydantic.BaseModel):
    reasoning: str = pydantic.Field(description="Thoughts on how to respond to the user's message")
    response: str = pydantic.Field(description="The response to the user's message")
    user_name: str|None = pydantic.Field(description="The name of the user who sent the message, if known")
@agents.output_guardrail
async def sensitive_data_check(context:agents.RunContextWrapper, agent:agents.Agent, output:SchemaMessageOutput) -> agents.GuardrailFunctionOutput:
    x0 = "650" in output.response
    x1 = "650" in output.reasoning
    ret = agents.GuardrailFunctionOutput(output_info={"phone_number_in_response":x0, "phone_number_in_reasoning":x1}, tripwire_triggered=(x0 or x1))
    return ret
agent00 = agents.Agent(name="Assistant", instructions="You are a helpful assistant.", output_type=SchemaMessageOutput, output_guardrails=[sensitive_data_check])

result0 = asyncio.run(agents.Runner.run(agent00, "What's the capital of California?")) #passed
result0.final_output.response
try:
    result1 = asyncio.run(agents.Runner.run(agent00, "My phone number is 650-123-4567. Where do you think I live?")) #should fail
except agents.OutputGuardrailTripwireTriggered as e:
    result1 = e.guardrail_result
    print(f'Guardrail tripwire triggered: "{result1.output.output_info}"')


### https://github.com/openai/openai-agents-python/blob/main/examples/basic/tools.py
class SchemaWeather(pydantic.BaseModel):
    city: str
    temperature_range: str
    conditions: str
@agents.function_tool
def get_weather(city: str) -> SchemaWeather:
    return SchemaWeather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")
agent00 = agents.Agent(name="Hello world", instructions="You are a helpful agent.", tools=[get_weather])
result = asyncio.run(agents.Runner.run(agent00, input="What's the weather in Tokyo?"))
result.final_output


### https://github.com/openai/openai-agents-python/blob/main/examples/tools/file_search.py
vector_store = asyncio.run(client.vector_stores.create(name="numqi00"))
# https://github.com/numqi/numqi/blob/main/python/numqi/entangle/ppt.py
with open('ppt.py', "rb") as fid:
    file_obj = asyncio.run(client.vector_stores.files.upload_and_poll(vector_store_id=vector_store.id, file=fid))
tmp0 = agents.FileSearchTool(max_num_results=3, vector_store_ids=[vector_store.id], include_search_results=True)
agent00 = agents.Agent(name="File searcher", instructions="You are a helpful agent.", tools=[tmp0])
with agents.trace("File search example"):
    result = asyncio.run(agents.Runner.run(agent00, "Be concise, and tell me how to calculate the relative of entropy of entanglement for Bell state using the utility functions provided."))
print(result.final_output)


### https://github.com/openai/openai-agents-python/blob/main/examples/tools/code_interpreter.py
## DO NOT run this example, it takes 0.3RMB per call
# tmp0 = agents.CodeInterpreterTool(tool_config={"type": "code_interpreter", "container": {"type": "auto"}})
# agent = agents.Agent(name="Code interpreter", instructions="You love doing math.", tools=[tmp0])
# async def hf_conversation(prompt:str):
#     with agents.trace("Code interpreter conversation"):
#         print("Solving math problem...")
#         # result = agents.Runner.run(agent, prompt)
#         result = agents.Runner.run_streamed(agent, prompt)
#         async for event in result.stream_events():
#             if ((event.type == "run_item_stream_event") and (event.item.type == "tool_call_item") and (event.item.raw_item.type == "code_interpreter_call")):
#                 print(f"Code interpreter code:\n```\n{event.item.raw_item.code}\n```\n")
#             elif event.type == "run_item_stream_event":
#                 print(f"Other event: {event.item.type}")
#         print(f"Final output: {result.final_output}")
# prompt = '"What is the square root of 273 * 312821 plus 1782?"'
# asyncio.run(hf_conversation(prompt))

'''
Solving math problem...
Code interpreter code:
```
import math

# Calculate the expression
result = math.sqrt(273 * 312821 + 1782)
result
```

Other event: message_output_item
Final output: The square root of \(273 \times 312821 + 1782\) is approximately \(9241.32\).
'''


## https://github.com/openai/openai-agents-python/blob/main/examples/handoffs/message_filter.py
@agents.function_tool
def random_number_tool(max_: int) -> int:
    """Return a random integer between 0 and the given maximum."""
    return random.randint(0, max_)

def spanish_handoff_message_filter(handoff_message_data: agents.HandoffInputData) -> agents.HandoffInputData:
    # remove any tool-related messages from the message history
    x0 = agents.extensions.handoff_filters.remove_all_tools(handoff_message_data) #input history can be manipulated here
    ret = agents.HandoffInputData(input_history=x0.input_history, pre_handoff_items=tuple(x0.pre_handoff_items), new_items=tuple(x0.new_items))
    return ret

first_agent = agents.Agent(name="Assistant", instructions="Be extremely concise.", tools=[random_number_tool])
prompt = "You only speak Spanish and are extremely concise."
spanish_agent = agents.Agent(name="Spanish Assistant", instructions=prompt, handoff_description="A Spanish-speaking assistant.")
prompt = "Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant."
tmp0 = agents.handoff(spanish_agent, input_filter=spanish_handoff_message_filter)
second_agent = agents.Agent(name="Assistant", instructions=prompt, handoffs=[tmp0])

with agents.trace(workflow_name="Message filtering"):
    result = asyncio.run(agents.Runner.run(first_agent, input="Hi, my name is Sora."))
    message_list = result.to_input_list() + [{"content": "Can you generate a random number between 0 and 100?", "role": "user"}]
    result = asyncio.run(agents.Runner.run(first_agent, input=message_list))
    message_list = result.to_input_list() + [{"content": "I live in New York City. Whats the population of the city?", "role": "user"}]
    result = asyncio.run(agents.Runner.run(second_agent, input=message_list))
    # Please speak Spanish. What is my name and where do I live?
    message_list = result.to_input_list() + [{"content": "Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?", "role": "user"}]
    result = asyncio.run(agents.Runner.run(second_agent, input=message_list)) #handoff to Spanish assistant
    print(result.to_input_list())
