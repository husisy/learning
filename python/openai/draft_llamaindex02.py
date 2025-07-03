# https://docs.llamaindex.ai/en/stable/understanding/agent/
# https://docs.llamaindex.ai/en/stable/understanding/agent/tools/
import os
import dotenv
import asyncio

import llama_index
import llama_index.core
import llama_index.core.workflow
import llama_index.core.agent.workflow
import llama_index.llms.openai
import llama_index.tools.yahoo_finance
import llama_index.tools.tavily_research

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']


def hf_llama_index_agent_run(agent, prompt:str|list[str], ctx=None):
    # as `asyncio.run(agent.run(prompt))` will fail
    isone = isinstance(prompt, str)
    if isone:
        prompt:list[str] = [prompt]
    assert all(isinstance(p, str) for p in prompt), "All prompts must be strings."
    async def hf_agent_run(prompt:list[str]):
        ret = []
        for p in prompt:
            ret.append(await agent.run(p, ctx=ctx))
        return ret
    ret = asyncio.run(hf_agent_run(prompt))
    if isone:
        ret = ret[0]
    return ret


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


llm = llama_index.llms.openai.OpenAI(model="gpt-4o-mini")

workflow = llama_index.core.agent.workflow.FunctionAgent(tools=[multiply, add], llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.")
x0 = hf_llama_index_agent_run(workflow, "What is 20+(2*4)?")



finance_tools = llama_index.tools.yahoo_finance.YahooFinanceToolSpec().to_tool_list()
workflow = llama_index.core.agent.workflow.FunctionAgent(name="Agent", llm=llm,
    tools=finance_tools+[multiply,add],
    description="Useful for performing financial operations.",
    system_prompt="You are a helpful assistant.",
)
x0 = hf_llama_index_agent_run(workflow, "What's the current stock price of NVIDIA?")


tavily_tool = llama_index.tools.tavily_research.TavilyToolSpec(api_key=ENV_CONFIG['TAVILY_API_KEY'])
workflow = llama_index.core.agent.workflow.FunctionAgent(
    tools=tavily_tool.to_tool_list(), llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)
x0 = hf_llama_index_agent_run(workflow, "What's the weather like in ShenZhen?")




async def dangerous_task(ctx: llama_index.core.workflow.Context) -> str:
    """A dangerous task that requires human confirmation."""
    question = "Are you sure you want to proceed? "
    tmp0 = llama_index.core.workflow.InputRequiredEvent(prefix=question, user_name="Laurie")
    response = await ctx.wait_for_event(llama_index.core.workflow.HumanResponseEvent,
            waiter_id=question, waiter_event=tmp0, requirements={"user_name": "Laurie"})
    if response.response.strip().lower() == "yes":
        ret = "Dangerous task completed successfully."
    else:
        ret = "Dangerous task aborted."
    return ret
workflow = llama_index.core.agent.workflow.FunctionAgent(tools=[dangerous_task], llm=llm,
        system_prompt="You are a helpful assistant that can perform dangerous tasks.")
async def hf_dangerous_workflow():
    handler = workflow.run(user_msg="I want to proceed with the dangerous task.")
    async for event in handler.stream_events():
        if isinstance(event, llama_index.core.workflow.InputRequiredEvent):
            x0 = input(event.prefix)
            handler.ctx.send_event(llama_index.core.workflow.HumanResponseEvent(response=x0, user_name=event.user_name))
    print(await handler)
asyncio.run(hf_dangerous_workflow())



## set state
async def set_name(ctx: llama_index.core.workflow.Context, name: str) -> str:
    state = await ctx.store.get("state")
    state["name"] = name
    await ctx.store.set("state", state)
    return f"Name set to {name}"

workflow = llama_index.core.agent.workflow.AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"},
)


async def hf_workflow_set_state():
    ctx = llama_index.core.workflow.Context(workflow)
    response = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(response)
    response2 = await workflow.run(user_msg="My name is Laurie", ctx=ctx)
    print(response2)
    # retrieve the value from the state directly
    state = await ctx.store.get("state")
    print("Name as stored in state: ",state["name"])

asyncio.run(hf_workflow_set_state())
