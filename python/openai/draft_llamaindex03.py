# https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/
import os
import re
import dotenv
import asyncio
import xml.etree.ElementTree
import pydantic
import typing

import llama_index
import llama_index.core
import llama_index.core.workflow
import llama_index.core.agent.workflow
import llama_index.core.llms
import llama_index.llms.openai
import tavily

ENV_CONFIG = dotenv.dotenv_values('.env')
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

## https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/
llm = llama_index.llms.openai.OpenAI(model="gpt-4o-mini")

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = tavily.AsyncTavilyClient(api_key=ENV_CONFIG['TAVILY_API_KEY'])
    return str(await client.search(query))


async def record_notes(ctx: llama_index.core.workflow.Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic. Your input should be notes with a title to save the notes under."""
    x0 = await ctx.store.get("state")
    if "research_notes" not in x0:
        x0["research_notes"] = {}
    x0["research_notes"][notes_title] = notes
    await ctx.store.set("state", x0)
    return "Notes recorded."


async def write_report(ctx: llama_index.core.workflow.Context, report_content: str) -> str:
    """Useful for writing a report on a given topic. Your input should be a markdown formatted report."""
    x0 = await ctx.store.get("state")
    x0["report_content"] = report_content
    await ctx.store.set("state", x0)
    return "Report written."


async def review_report(ctx: llama_index.core.workflow.Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback. Your input should be a review of the report."""
    x0 = await ctx.store.get("state")
    x0["review"] = review
    await ctx.store.set("state", x0)
    return "Report reviewed."


tmp0 = ("You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic. "
        "You should have at least some notes on a topic before handing off control to the WriteAgent.")
research_agent = llama_index.core.agent.workflow.FunctionAgent(
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    name="ResearchAgent", system_prompt=tmp0, llm=llm, tools=[search_web, record_notes], can_handoff_to=["WriteAgent"],
)

tmp0 = ("You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent.")
write_agent = llama_index.core.agent.workflow.FunctionAgent(
    description="Useful for writing a report on a given topic.",
    name="WriteAgent", system_prompt=tmp0, llm=llm, tools=[write_report], can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

tmp0 = ("You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes for the WriteAgent to implement. "
        "If you have feedback that requires changes, you should hand off control to the WriteAgent to implement the changes after submitting the review.")
review_agent = llama_index.core.agent.workflow.FunctionAgent(
    description="Useful for reviewing a report and providing feedback.",
    name="ReviewAgent", system_prompt=tmp0, llm=llm, tools=[review_report], can_handoff_to=["WriteAgent"],
)

tmp0 = {"research_notes": {}, "report_content": "Not written yet.", "review": "Review required."}
agent_workflow = llama_index.core.agent.workflow.AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state=tmp0,
)
# TODO, is the initial_state for all sessions

async def hf_agent_workflow_run(ctx: llama_index.core.workflow.Context, msg:str):
    handler = agent_workflow.run(user_msg=msg, ctx=ctx)
    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and (event.current_agent_name != current_agent):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")

        # if isinstance(event, AgentStream):
        #     if event.delta:
        #         print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("📥 Input:", event.input)
        elif isinstance(event, llama_index.core.agent.workflow.AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print("🛠️  Planning to use tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, llama_index.core.agent.workflow.ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, llama_index.core.agent.workflow.ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")

# tmp0 = ("Write me a report on the history of the internet. "
#         "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
#         "and the development of the internet in the 21st century.")
ctx = llama_index.core.workflow.Context(agent_workflow)
tmp0 = ("Please write a report on the recent research progress in concatenated quantum error-correcting codes. "
        "The report should briefly summarize recent advances, highlight key developments in theory and practical implementation, "
        "and discuss current challenges and future directions in this area.")
asyncio.run(hf_agent_workflow_run(ctx, tmp0))
x0 = asyncio.run(ctx.store.get('state'))
print(x0['report_content'])



## https://docs.llamaindex.ai/en/stable/examples/agent/agents_as_tools/
sub_agent_llm = llama_index.llms.openai.OpenAI(model="gpt-4.1-mini")
orchestrator_llm = llama_index.llms.openai.OpenAI(model="o3-mini")

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = tavily.AsyncTavilyClient(api_key=ENV_CONFIG['TAVILY_API_KEY'])
    return str(await client.search(query))

tmp0 = ("You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "You should output notes on the topic in a structured format.")
research_agent = llama_index.core.agent.workflow.FunctionAgent(system_prompt=tmp0, llm=sub_agent_llm, tools=[search_web])

tmp0 = ("You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Return your markdown report surrounded by ... tags.")
write_agent = llama_index.core.agent.workflow.FunctionAgent(system_prompt=tmp0, llm=sub_agent_llm, tools=[])

tmp0 = ("You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes to be implemented.")
review_agent = llama_index.core.agent.workflow.FunctionAgent(system_prompt=tmp0, llm=sub_agent_llm, tools=[])

async def call_research_agent(ctx: llama_index.core.workflow.Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(user_msg=f"Write some notes about the following: {prompt}")
    state = await ctx.store.get("state")
    state["research_notes"].append(str(result))
    await ctx.store.set("state", state)
    return str(result)

async def call_write_agent(ctx: llama_index.core.workflow.Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    state = await ctx.store.get("state")
    notes = state.get("research_notes", None)
    if not notes:
        return "No research notes to write from."
    user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: ...:\n\n"
    feedback = state.get("review", None)
    if feedback:
        user_msg += f"{feedback}\n\n"
    user_msg += "\n\n".join(notes) + "\n\n"
    result = await write_agent.run(user_msg=user_msg)
    report = re.search(r"(.*)", str(result), re.DOTALL).group(1)
    state["report_content"] = str(report)
    await ctx.store.set("state", state)
    return str(report)

async def call_review_agent(ctx: llama_index.core.workflow.Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    state = await ctx.store.get("state")
    report = state.get("report_content", None)
    if not report:
        return "No report content to review."
    result = await review_agent.run(user_msg=f"Review the following report: {report}")
    state["review"] = result
    await ctx.store.set("state", state)
    return result

tmp0 = ("You are an expert in the field of report writing. "
        "You are given a user request and a list of tools that can help with the request. "
        "You are to orchestrate the tools to research, write, and review a report on the given topic. "
        "Once the review is positive, you should notify the user that the report is ready to be accessed.")
tmp1 = {"research_notes": [], "report_content": None, "review": None}
orchestrator = llama_index.core.agent.workflow.FunctionAgent(
    tools=[call_research_agent, call_write_agent, call_review_agent,],
    system_prompt=tmp0, llm=orchestrator_llm, initial_state=tmp1,
)

async def run_orchestrator(ctx: llama_index.core.workflow.Context, user_msg: str):
    handler = orchestrator.run(user_msg=user_msg, ctx=ctx)
    async for event in handler.stream_events():
        if isinstance(event, llama_index.core.agent.workflow.AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("📥 Input:", event.input)
        elif isinstance(event, llama_index.core.agent.workflow.AgentOutput):
            # Skip printing the output since we are streaming above
            # if event.response.content:
            #     print("📤 Output:", event.response.content)
            if event.tool_calls:
                print("🛠️  Planning to use tools:", [x.tool_name for x in event.tool_calls])
        elif isinstance(event, llama_index.core.agent.workflow.ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, llama_index.core.agent.workflow.ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")

ctx = llama_index.core.workflow.Context(orchestrator)
# tmp0 = ("Please write a report on the recent research progress in concatenated quantum error-correcting codes. "
#         "The report should briefly summarize recent advances, highlight key developments in theory and practical implementation, "
#         "and discuss current challenges and future directions in this area.")
tmp0 = ("Please write a report on recent progress in open-source agentic Python libraries that are similar to, "
        "or serve as alternatives to, LlamaIndex. The report should provide an overview of notable libraries, "
        "highlight recent developments and features, compare their capabilities, and discuss their current status and potential applications.")
asyncio.run(run_orchestrator(ctx, tmp0))
x0 = asyncio.run(ctx.store.get("state"))
print(x0['report_content'])


## fail https://docs.llamaindex.ai/en/stable/examples/agent/custom_multi_agent/
sub_agent_llm = llama_index.llms.openai.OpenAI(model="gpt-4.1-mini")

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = tavily.AsyncTavilyClient(api_key=ENV_CONFIG['TAVILY_API_KEY'])
    return str(await client.search(query))

tmp0 = ("You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "You should output notes on the topic in a structured format.")
research_agent = llama_index.core.agent.workflow.FunctionAgent(
    description="Useful for recording research notes based on a specific prompt.",
    name="ResearchAgent", system_prompt=tmp0, llm=sub_agent_llm, tools=[search_web])
tmp0 = ("You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Return your markdown report surrounded by ... tags.")
write_agent = llama_index.core.agent.workflow.FunctionAgent(
    description="Useful for writing a report based on the research notes or revising the report based on feedback.",
    name="WriteAgent", system_prompt=tmp0, llm=sub_agent_llm, tools=[])
tmp0 = ("You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes to be implemented.")
review_agent = llama_index.core.agent.workflow.FunctionAgent(
    description="Useful for reviewing a report and providing feedback.",
    name="ReviewAgent", system_prompt=tmp0, llm=sub_agent_llm, tools=[])

async def call_research_agent(ctx: llama_index.core.workflow.Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(user_msg=f"Write some notes about the following: {prompt}")
    state = await ctx.store.get("state")
    state["research_notes"].append(str(result))
    await ctx.store.set("state", state)
    return str(result)

async def call_write_agent(ctx: llama_index.core.workflow.Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    state = await ctx.store.get("state")
    notes = state.get("research_notes", None)
    if not notes:
        return "No research notes to write from."
    user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: ...:\n\n"
    feedback = state.get("review", None)
    if feedback:
        user_msg += f"{feedback}\n\n"
    user_msg += "\n\n".join(notes) + "\n\n"
    result = await write_agent.run(user_msg=user_msg)
    report = re.search(r"(.*)", str(result), re.DOTALL).group(1)
    state["report_content"] = str(report)
    await ctx.store.set("state", state)
    return str(report)

async def call_review_agent(ctx: llama_index.core.workflow.Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    state = await ctx.store.get("state")
    report = state.get("report_content", None)
    if not report:
        return "No report content to review."
    result = await review_agent.run(user_msg=f"Review the following report: {report}")
    state["review"] = result
    await ctx.store.set("state", state)
    return result

PLANNER_PROMPT = """You are a planner chatbot.

Given a user request and the current state, break the solution into ordered  blocks.  Each step must specify the agent to call and the message to send, e.g.

search for …
draft a report …
  ...


{state}


{available_agents}


The general flow should be:
- Record research notes
- Write a report
- Review the report
- Write the report again if the review is not positive enough

If the user request does not require any steps, you can skip the  block and respond directly.
"""

class InputEvent(llama_index.core.workflow.StartEvent):
    user_msg: typing.Optional[str] = pydantic.Field(default=None)
    chat_history: list[llama_index.core.llms.ChatMessage]
    state: typing.Optional[dict[str, typing.Any]] = pydantic.Field(default=None)

class OutputEvent(llama_index.core.workflow.StopEvent):
    response: str
    chat_history: list[llama_index.core.llms.ChatMessage]
    state: dict[str, typing.Any]

class StreamEvent(llama_index.core.workflow.Event):
    delta: str

class PlanEvent(llama_index.core.workflow.Event):
    step_info: str

# Modelling the plan
class PlanStep(pydantic.BaseModel):
    agent_name: str
    agent_input: str

class Plan(pydantic.BaseModel):
    steps: list[PlanStep]

class ExecuteEvent(llama_index.core.workflow.Event):
    plan: Plan
    chat_history: list[llama_index.core.llms.ChatMessage]

class PlannerWorkflow(llama_index.core.workflow.Workflow):
    llm: llama_index.llms.openai.OpenAI = llama_index.llms.openai.OpenAI(model="o3-mini")
    agents: dict[str, llama_index.core.agent.workflow.FunctionAgent] = {"ResearchAgent": research_agent, "WriteAgent": write_agent, "ReviewAgent": review_agent}

    @llama_index.core.workflow.step
    async def plan(self, ctx: llama_index.core.workflow.Context, ev: InputEvent) -> ExecuteEvent | OutputEvent:
        # Set initial state if it exists
        if ev.state:
            await ctx.store.set("state", ev.state)

        chat_history = ev.chat_history

        if ev.user_msg:
            user_msg = llama_index.core.llms.ChatMessage(role="user", content=ev.user_msg)
            chat_history.append(user_msg)

        # Inject the system prompt with state and available agents
        state = await ctx.store.get("state")
        available_agents_str = "\n".join([f'{agent.description}' for agent in self.agents.values()])
        system_prompt = llama_index.core.llms.ChatMessage(role="system", content=PLANNER_PROMPT.format(state=str(state), available_agents=available_agents_str))

        # Stream the response from the llm
        response = await self.llm.astream_chat(messages=[system_prompt] + chat_history)
        full_response = ""
        async for chunk in response:
            full_response += chunk.delta or ""
            if chunk.delta:
                ctx.write_event_to_stream(StreamEvent(delta=chunk.delta))

        # Parse the response into a plan and decide whether to execute or output
        xml_match = re.search(r"(.*)", full_response, re.DOTALL)

        if not xml_match:
            chat_history.append(llama_index.core.llms.ChatMessage(role="assistant", content=full_response))
            return OutputEvent(response=full_response, chat_history=chat_history, state=state)
        else:
            xml_str = xml_match.group(1)
            root = xml.etree.ElementTree.fromstring(xml_str)
            plan = Plan(steps=[])
            for step in root.findall("step"):
                plan.steps.append(PlanStep(agent_name=step.attrib["agent"], agent_input=step.text.strip() if step.text else ""))

            return ExecuteEvent(plan=plan, chat_history=chat_history)

    @llama_index.core.workflow.step
    async def execute(self, ctx: llama_index.core.workflow.Context, ev: ExecuteEvent) -> InputEvent:
        chat_history = ev.chat_history
        plan = ev.plan

        for step in plan.steps:
            agent = self.agents[step.agent_name]
            agent_input = step.agent_input
            ctx.write_event_to_stream(PlanEvent(step_info=f'{step.agent_input}'))

            if step.agent_name == "ResearchAgent":
                await call_research_agent(ctx, agent_input)
            elif step.agent_name == "WriteAgent":
                # Note: we aren't passing the input from the plan since
                # we're using the state to drive the write agent
                await call_write_agent(ctx)
            elif step.agent_name == "ReviewAgent":
                await call_review_agent(ctx)

        state = await ctx.store.get("state")
        tmp0 = f"I've completed the previous steps, here's the updated state:\n\n\n{state}\n\n\nDo you need to continue and plan more steps?, If not, write a final response."
        chat_history.append(llama_index.core.llms.ChatMessage(role="user", content=tmp0))

        return InputEvent(chat_history=chat_history)


async def run_planner_workflow(user_msg:str):
    planner_workflow = PlannerWorkflow(timeout=None)
    tmp0 = {"research_notes": [], "report_content": "Not written yet.", "review": "Review required."}
    handler = planner_workflow.run(user_msg=x0, chat_history=[], state=tmp0)

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if isinstance(event, PlanEvent):
            print("Executing plan step: ", event.step_info)
        elif isinstance(event, ExecuteEvent):
            print("Executing plan: ", event.plan)

    result = await handler
    return result

x0 = ("Write me a report on the history of the internet. "
        "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
        "and the development of the internet in the 21st century.")
x1 = asyncio.run(run_planner_workflow(x0))
