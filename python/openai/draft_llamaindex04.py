# https://docs.llamaindex.ai/en/stable/understanding/workflows/branches_and_loops/
# https://docs.llamaindex.ai/en/stable/understanding/workflows/state/
import os
import dotenv
import pydantic
import random
import asyncio

import llama_index
import llama_index.core
import llama_index.core.workflow
import llama_index.utils.workflow
import llama_index.llms.openai

ENV_CONFIG = dotenv.dotenv_values('.env')
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

def get_openai_model(model:str|None=None):
    import httpx
    ret = llama_index.llms.openai.OpenAI(model=model, api_key=ENV_CONFIG['OPENAI_API_KEY'], http_client=httpx.Client(proxy=ENV_CONFIG.get('OPENAI_PROXY', None)))
    return ret

async def hf_dummy_run(hf0, *args, **kwargs):
    ret = await hf0(*args, **kwargs)
    return ret

class MyWorkflow(llama_index.core.workflow.Workflow):
    @llama_index.core.workflow.step
    async def my_step(self, ev: llama_index.core.workflow.StartEvent) -> llama_index.core.workflow.StopEvent:
        return llama_index.core.workflow.StopEvent(result="Hello, world!")
# llama_index.utils.workflow.draw_all_possible_flows(MyWorkflow, filename="tbd00.html") # max_label_length=10
x0 = MyWorkflow(timeout=10, verbose=False)
result = asyncio.run(hf_dummy_run(x0.run))


## multi-step, loop, branch
class LoopEvent(llama_index.core.workflow.Event):
    loop_output: str

class FirstEvent(llama_index.core.workflow.Event):
    first_output: str

class SecondA1Event(llama_index.core.workflow.Event):
    payload: str

class SecondA2Event(llama_index.core.workflow.Event):
    payload: str

class MyWorkflow01(llama_index.core.workflow.Workflow):
    @llama_index.core.workflow.step
    async def step_one(self, ev: llama_index.core.workflow.StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            ret = LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            ret = FirstEvent(first_output="First step complete.")
        return ret

    @llama_index.core.workflow.step
    async def step_two(self, ev: FirstEvent) -> SecondA1Event | SecondA2Event:
        if random.randint(0, 1) == 0:
            print("Go to branch A1")
            ret = SecondA1Event(payload="Branch A1")
        else:
            print("Go to branch A2")
            ret = SecondA2Event(payload="Branch A2")
        return ret

    @llama_index.core.workflow.step
    async def step_a1(self, ev: SecondA1Event) -> llama_index.core.workflow.StopEvent:
        print(ev.payload)
        return llama_index.core.workflow.StopEvent(result="Branch A1 complete.")

    @llama_index.core.workflow.step
    async def step_a2(self, ev: SecondA2Event) -> llama_index.core.workflow.StopEvent:
        print(ev.payload)
        return llama_index.core.workflow.StopEvent(result="Branch A2 complete.")
# llama_index.utils.workflow.draw_all_possible_flows(MyWorkflow01, filename="tbd00.html") # max_label_length=10

x0 = MyWorkflow01(timeout=10, verbose=False)
result = asyncio.run(hf_dummy_run(x0.run, first_input="Start the workflow."))
print(result)


## maintain state
class SetupEvent(llama_index.core.workflow.Event):
    query: str

class StepTwoEvent(llama_index.core.workflow.Event):
    query: str

class StatefulFlow(llama_index.core.workflow.Workflow):
    @llama_index.core.workflow.step
    async def start(self, ctx: llama_index.core.workflow.Context, ev: llama_index.core.workflow.StartEvent) -> SetupEvent | StepTwoEvent:
        db = await ctx.store.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            ret = SetupEvent(query=ev.query)
        else:
            ret = StepTwoEvent(query=ev.query)
        return ret

    @llama_index.core.workflow.step
    async def setup(self, ctx: llama_index.core.workflow.Context, ev: SetupEvent) -> llama_index.core.workflow.StartEvent:
        await ctx.store.set("some_database", [1, 2, 3])
        return llama_index.core.workflow.StartEvent(query=ev.query)

    @llama_index.core.workflow.step
    async def step_two(self, ctx: llama_index.core.workflow.Context, ev: StepTwoEvent) -> llama_index.core.workflow.StopEvent:
        print("Data is ", await ctx.store.get("some_database"))
        return llama_index.core.workflow.StopEvent(result=await ctx.store.get("some_database"))

x0 = StatefulFlow(timeout=10, verbose=False)
result = asyncio.run(hf_dummy_run(x0.run, query="Some query"))
print(result)


class MyRandomObject:
    def __init__(self, name: str = "default"):
        self.name = name
# NOTE: all fields must have defaults
class MyState(pydantic.BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    my_obj: MyRandomObject = pydantic.Field(default_factory=MyRandomObject)
    some_key: str = pydantic.Field(default="some_value")

    # optional, but can be useful if you want to control the serialization of your state!
    @pydantic.field_serializer("my_obj", when_used="always")
    def serialize_my_obj(self, my_obj: MyRandomObject) -> str:
        return my_obj.name

    @pydantic.field_validator("my_obj", mode="before")
    @classmethod
    def deserialize_my_obj(cls, v: str | MyRandomObject) -> MyRandomObject:
        if isinstance(v, MyRandomObject):
            return v
        if isinstance(v, str):
            return MyRandomObject(v)
        raise ValueError(f"Invalid type for my_obj: {type(v)}")

class MyStatefulFlow(llama_index.core.workflow.Workflow):
    @llama_index.core.workflow.step
    async def start(self, ctx: llama_index.core.workflow.Context[MyState], ev: llama_index.core.workflow.StartEvent) -> llama_index.core.workflow.StopEvent:
        state = await ctx.store.get_state()
        state.my_obj.name = "new_name"
        await ctx.store.set_state(state)
        ## Can also access fields directly if needed
        # name = await ctx.store.get("my_obj.name")
        await ctx.store.set("my_obj.name", "newer_name")
        return llama_index.core.workflow.StopEvent(result="Done!")

x0 = MyStatefulFlow(timeout=10, verbose=False)
ctx = llama_index.core.workflow.Context(x0)
result = asyncio.run(hf_dummy_run(x0.run, ctx=ctx))
state:MyState = asyncio.run(ctx.store.get_state())
print(state, state.my_obj.name)


## stream events
# https://docs.llamaindex.ai/en/stable/understanding/workflows/stream/
class FirstEvent(llama_index.core.workflow.Event):
    first_output: str

class SecondEvent(llama_index.core.workflow.Event):
    second_output: str
    response: str

class ProgressEvent(llama_index.core.workflow.Event):
    msg: str

class MyWorkflow(llama_index.core.workflow.Workflow):
    @llama_index.core.workflow.step
    async def step_one(self, ctx: llama_index.core.workflow.Context, ev: llama_index.core.workflow.StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")

    @llama_index.core.workflow.step
    async def step_two(self, ctx: llama_index.core.workflow.Context, ev: FirstEvent) -> SecondEvent:
        # llm = get_openai_model('gpt-4o-mini')
        llm = llama_index.llms.openai.OpenAI(model="gpt-4o-mini")
        generator = await llm.astream_complete("Please give me the first 3 paragraphs of Moby Dick, a book in the public domain.")
        async for response in generator:
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
        return SecondEvent(second_output="Second step complete, full response attached", response=str(response))

    @llama_index.core.workflow.step
    async def step_three(self, ctx: llama_index.core.workflow.Context, ev: SecondEvent) -> llama_index.core.workflow.StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return llama_index.core.workflow.StopEvent(result="Workflow complete.")
# llama_index.utils.workflow.draw_all_possible_flows(MyWorkflow, filename="tbd00.html")

async def hf_myworkflow_run():
    w = MyWorkflow(timeout=30, verbose=True)
    handler = w.run(first_input="Start the workflow.")
    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg, end='', flush=True)
    ret = await handler
    return ret
x0 = asyncio.run(hf_myworkflow_run())


## Concurrent execution of workflows
## https://docs.llamaindex.ai/en/stable/understanding/workflows/concurrent_execution/
class StepTwoEvent(llama_index.core.workflow.Event):
    query: str

class StepThreeEvent(llama_index.core.workflow.Event):
    result: str

class ConcurrentFlow(llama_index.core.workflow.Workflow):
    @llama_index.core.workflow.step
    async def start(self, ctx: llama_index.core.workflow.Context, ev: llama_index.core.workflow.StartEvent) -> StepTwoEvent:
        ctx.send_event(StepTwoEvent(query="Query 1"))
        ctx.send_event(StepTwoEvent(query="Query 2"))
        ctx.send_event(StepTwoEvent(query="Query 3"))

    @llama_index.core.workflow.step(num_workers=4)
    async def step_two(self, ctx: llama_index.core.workflow.Context, ev: StepTwoEvent) -> StepThreeEvent:
        print("Running query ", ev.query)
        await asyncio.sleep(random.randint(1, 5))
        return StepThreeEvent(result=ev.query)

    @llama_index.core.workflow.step
    async def step_three(self, ctx: llama_index.core.workflow.Context, ev: StepThreeEvent) -> llama_index.core.workflow.StopEvent:
        # wait until we receive 3 events
        result = ctx.collect_events(ev, [StepThreeEvent] * 3)
        if result is None:
            return None
        # do something with all 3 results together
        print(result)
        return llama_index.core.workflow.StopEvent(result="Done")
