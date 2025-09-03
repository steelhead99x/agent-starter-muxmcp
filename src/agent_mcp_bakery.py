import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
import os

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

# from livekit.plugins import noise_cancellation

# This example demonstrates a multi-agent system where tasks are delegated to sub-agents
# based on the user's request.
#
# The user is initially connected to a greeter, and depending on their need, the call is
# handed off to other agents that could help with the more specific tasks.
# This helps to keep each agent focused on the task at hand, and also reduces costs
# since only a subset of the tools are used at any given time.


logger = logging.getLogger("bakery-example")
logger.setLevel(logging.INFO)

load_dotenv('.env.local')

voices = {
    "greeter": "565510e8-6b45-45de-8758-13588fbaec73",
    "appointment": "156fb8d2-335b-4950-9cb3-a2d33befec77",
    "pickup": "996a8b96-4804-46f0-8e05-3fd4ef1a87cd",
    "checkout": "39b376fc-488e-4d0c-8b37-e00b72059fdd",
    "message": "156fb8d2-335b-4950-9cb3-a2d33befec77",
}


@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    pickup_time: Optional[str] = None

    order: Optional[list[str]] = None

    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    message: Optional[str] = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "pickup_time": self.pickup_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_cvv or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


# common functions


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        chat_ctx.add_message(
            role="system",
            content=(
                "Speak in plain, natural sentences for voice. Do not use any markdown or special characters. "
                "Do not output asterisks '*', underscores '_', bullets '-', numbered lists, code blocks, or emojis. "
                "Never say the words 'asterisk' or 'star'; just speak the content plainly."
            ),
        )
        await self.update_chat_ctx(chat_ctx)
        # Allow the model to call tools (e.g., update_name) on agent entry
        self.session.generate_reply(tool_choice="auto")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."


class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a friendly employee at a roadside bakery stand. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "place a pickup order, leave a message, or proceed to checkout. "
                "Guide them to the right agent using tools. "
                "Speak plainly without markdown, bullets, or asterisks."
            ),
            llm=openai.LLM(
                model=os.getenv("LLM_MODEL", "gpt-oss:20b"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.88.16:11434/v1"),
                parallel_tool_calls=False,
            ),
            tts=cartesia.TTS(voice=voices["greeter"]),
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a pickup appointment time.
        This function handles transitioning to the appointment agent
        who will collect the necessary details like pickup time,
        customer name and phone number."""
        return await self._transfer_to_agent("appointment", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a pickup order.
        This includes handling orders for pickup, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("pickup", context)

    @function_tool()
    async def to_message(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to leave a message for the bakery."""
        return await self._transfer_to_agent("message", context)


class Appointment(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are an appointment agent for a bakery pickup. Ask for the pickup time, "
            "then the customer's name and phone number. Then confirm the appointment details with the customer. "
            "Use plain sentences only; no markdown or asterisks.",
            tools=[update_name, update_phone, to_greeter],
            tts=cartesia.TTS(voice=voices["appointment"]),
        )

    @function_tool()
    async def update_pickup_time(
        self,
        time: Annotated[str, Field(description="The pickup appointment time")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their pickup time.
        Confirm the time with the user before calling the function."""
        userdata = context.userdata
        userdata.pickup_time = time
        return f"The pickup time is updated to {time}"

    @function_tool()
    async def confirm_appointment(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the pickup appointment."""
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."

        if not userdata.pickup_time:
            return "Please provide pickup time first."

        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a pickup order agent for the bakery. "
                f"Our menu is: {menu}\n"
                "Clarify quantities and special requests, and confirm the order with the customer. "
                "Respond in plain text with no asterisks or markdown."
            ),
            tools=[to_greeter],
            tts=cartesia.TTS(voice=voices["pickup"]),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user create or update their order."""
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        userdata = context.userdata
        if not userdata.order:
            return "No pickup order found. Please make an order first."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a checkout agent at a bakery. The menu is: {menu}\n"
                "You are responsible for confirming the total cost of the order and then collecting the customer's "
                "name, phone number and credit card information step by step (card number, expiry, CVV). "
                "Speak plainly; do not use markdown, bullets, or asterisks."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=cartesia.TTS(voice=voices["checkout"]),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("pickup", context)


class Message(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a message-taking agent for the bakery. "
                "Ask the caller for their name, phone number, and the message they would like to leave. "
                "Confirm the message back to them and then return to the greeter. "
                "Keep language plain; avoid markdown and asterisks."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=cartesia.TTS(voice=voices["message"]),
        )

    @function_tool()
    async def update_message(
        self,
        message: Annotated[str, Field(description="The message the caller wants to leave")],
        context: RunContext_T,
    ) -> str:
        """Called when the user dictates their message."""
        userdata = context.userdata
        userdata.message = message
        return "Message recorded."

    @function_tool()
    async def confirm_message(self, context: RunContext_T) -> tuple[Agent, str] | str:
        """Called when the user confirms the message is correct."""
        userdata = context.userdata
        if not userdata.message:
            return "Please provide the message you'd like to leave."
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number so we can reach you back."
        return await self._transfer_to_agent("greeter", context)


async def entrypoint(ctx: JobContext):
    menu = "Home Made Muffins: $10, Sourdough Bread: $8, Fresh Baked Cinnamon Rolls: $15, Hot Coffee: $3, Fresh Goat Milk: $10"
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "appointment": Appointment(),
            "pickup": Takeaway(menu),
            "checkout": Checkout(menu),
            "message": Message(),
        }
    )
    session = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(
                    model=os.getenv("LLM_MODEL", "gpt-oss:20b"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.88.16:11434/v1"),
                ),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        max_tool_steps=5,
        # to use realtime model, replace the stt, llm, tts and vad with the following
        # llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # await agent.say("Welcome to our bakery stand! How may I assist you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))