import logging
import os
import base64

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import mcp

logger = logging.getLogger("mcp-agent")

# load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
load_dotenv(".env.local")

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You can retrieve data via the MCP server. The interface is voice-based: "
                "accept spoken user queries and respond with synthesized speech."
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()

token_id = os.getenv("MUX_TOKEN_ID", "")
token_secret = os.getenv("MUX_TOKEN_SECRET", "")
if not token_id or not token_secret:
  raise RuntimeError("MUX_TOKEN_ID and MUX_TOKEN_SECRET must be set")

basic = base64.b64encode(f"{token_id}:{token_secret}".encode()).decode()
mux_mcp_url = os.getenv("MUX_MCP_URL", "https://mcp.mux.com?tools=dynamic")
logging.getLogger("agent").info(f"Using Mux MCP URL: {mux_mcp_url}")


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "multi"),
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(
            voice=os.getenv("CARTESIA_VOICE", "6f84f4b8-58a2-430c-8c79-688dad597532")
        ),
        llm=openai.LLM(
            model=os.getenv("LLM_MODEL", "gpt-oss:20b"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.88.16:11434/v1"),
        ),
        turn_detection=MultilingualModel(),
        mcp_servers=[
            mcp.MCPServerHTTP(
                url=mux_mcp_url,
                headers={"Authorization": f"Basic {basic}"},
                timeout=30,
                client_session_timeout_seconds=30,
            ),
        ],
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))