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


## https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py
class SchemaAirlineAgentContext(pydantic.BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None

@agents.function_tool(name_override="faq_lookup_tool", description_override="Lookup frequently asked questions.")
async def faq_lookup_tool(question:str) -> str:
    if "bag" in question or "baggage" in question:
        ret = "You are allowed to bring one bag on the plane. It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
    elif "seats" in question or "plane" in question:
        ret = ("There are 120 seats on the plane. There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. Rows 5-8 are Economy Plus, with extra legroom. ")
    elif "wifi" in question:
        ret = "We have free wifi on the plane, join Airline-Wifi"
    else:
        ret = "I'm sorry, I don't know the answer to that question."
    return ret


@agents.function_tool
async def update_seat(context:agents.RunContextWrapper[SchemaAirlineAgentContext], confirmation_number:str, new_seat:str) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    # Update the context based on the customer's input
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    # Ensure that the flight number has been set by the incoming handoff
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


async def on_seat_booking_handoff(context: agents.RunContextWrapper[SchemaAirlineAgentContext]) -> None:
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number

prompt = f"""{agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX}
You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
Use the following routine to support the customer.
# Routine
1. Identify the last question asked by the customer.
2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
3. If you cannot answer the question, transfer back to the triage agent."""
tmp0 = "A helpful agent that can answer questions about the airline."
faq_agent = agents.Agent[SchemaAirlineAgentContext](name="FAQ Agent", handoff_description=tmp0, instructions=prompt, tools=[faq_lookup_tool])

prompt = f"""{agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX}
You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
Use the following routine to support the customer.
# Routine
1. Ask for their confirmation number.
2. Ask the customer what their desired seat number is.
3. Use the update seat tool to update the seat on the flight.
If the customer asks a question that is not related to the routine, transfer back to the triage agent. """
tmp0 = "A helpful agent that can update a seat on a flight."
seat_booking_agent = agents.Agent[SchemaAirlineAgentContext](name="Seat Booking Agent", handoff_description=tmp0, instructions=prompt, tools=[update_seat])

prompt = (f"{agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents.")
tmp0 = "A triage agent that can delegate a customer's request to the appropriate agent."
tmp1 = agents.handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff)
triage_agent = agents.Agent[SchemaAirlineAgentContext](name="Triage Agent", handoff_description=tmp0, instructions=prompt, handoffs=[faq_agent, tmp1])

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)

current_agent: agents.Agent[SchemaAirlineAgentContext] = triage_agent
input_items: list[agents.TResponseInputItem] = []
context = SchemaAirlineAgentContext()

# Normally, each input from the user would be an API request to your app, and you can wrap the request in a trace()
# Here, we'll just use a random UUID for the conversation ID
conversation_id = uuid.uuid4().hex[:16]

while True:
    user_input = input("Enter your message: ")
    with agents.trace("Customer service", group_id=conversation_id):
        input_items.append({"content": user_input, "role": "user"})
        result = asyncio.run(agents.Runner.run(current_agent, input_items, context=context))

        for new_item in result.new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, agents.MessageOutputItem):
                print(f"{agent_name}: {agents.ItemHelpers.text_message_output(new_item)}")
            elif isinstance(new_item, agents.HandoffOutputItem):
                print(f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")
            elif isinstance(new_item, agents.ToolCallItem):
                print(f"{agent_name}: Calling a tool")
            elif isinstance(new_item, agents.ToolCallOutputItem):
                print(f"{agent_name}: Tool call output: {new_item.output}")
            else:
                print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
        input_items = result.to_input_list()
        current_agent = result.last_agent
