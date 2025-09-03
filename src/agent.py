import logging
import os
from datetime import datetime, timezone

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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.
        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
        Args:
            location: The location to look up weather information for (e.g. city name)
        """
        logger.info(f"Looking up weather for {location}")

        try:
            import aiohttp

            # Identify your application per NWS and Nominatim policies
            app_name = os.getenv("WEATHER_APP_NAME", "LiveKit-Weather-Agent")
            contact = os.getenv("WEATHER_CONTACT_EMAIL", "someone@example.com")
            user_agent = f"{app_name} (contact: {contact})"

            # 1) Geocode the free-form location to lat/lon via Nominatim
            async def _geocode_location(q: str):
                url = "https://nominatim.openstreetmap.org/search"
                params = {"q": q, "format": "json", "limit": 1}
                headers = {"User-Agent": user_agent}
                async with aiohttp.ClientSession(headers=headers) as s:
                    async with s.get(url, params=params, timeout=15) as r:
                        if r.status != 200:
                            logger.warning(f"Geocoding failed HTTP {r.status}")
                            return None
                        data = await r.json()
                        if not data:
                            return None
                        try:
                            lat = float(data[0]["lat"])
                            lon = float(data[0]["lon"])
                            return lat, lon
                        except Exception:
                            return None

            coords = await _geocode_location(location)
            if not coords:
                logger.info("No geocoding result; returning UNSUPPORTED_LOCATION")
                return "UNSUPPORTED_LOCATION"

            lat, lon = coords

            # 2) Resolve NWS grid for that coordinate
            nws_headers = {
                "User-Agent": user_agent,
                "Accept": "application/geo+json",
            }
            async with aiohttp.ClientSession(headers=nws_headers) as s:
                points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
                async with s.get(points_url, timeout=15) as r:
                    if r.status != 200:
                        logger.info(f"NWS points lookup failed HTTP {r.status}; location likely unsupported")
                        return "UNSUPPORTED_LOCATION"
                    points = await r.json()

                props = points.get("properties") or {}
                office = props.get("gridId")
                grid_x = props.get("gridX")
                grid_y = props.get("gridY")
                if not all([office, isinstance(grid_x, int), isinstance(grid_y, int)]):
                    logger.info("NWS points response missing grid data; returning UNSUPPORTED_LOCATION")
                    return "UNSUPPORTED_LOCATION"

                # 3) Fetch the forecast for that grid
                forecast_url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast"
                async with s.get(forecast_url, timeout=15) as r:
                    if r.status != 200:
                        logger.warning(f"NWS forecast failed HTTP {r.status}")
                        raise RuntimeError("Weather service is unavailable")
                    forecast = await r.json()

            fprops = (forecast.get("properties") or {})
            periods = fprops.get("periods") or []
            if not periods:
                logger.info("NWS forecast has no periods; returning UNSUPPORTED_LOCATION")
                return "UNSUPPORTED_LOCATION"

            # Choose the most relevant period relative to now
            now = datetime.now(timezone.utc)

            def _parse_iso(ts: str):
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    return None

            chosen = None
            for p in periods:
                st = _parse_iso(p.get("startTime") or "")
                en = _parse_iso(p.get("endTime") or "")
                if st and en and st <= now <= en:
                    chosen = p
                    break
            if not chosen:
                upcoming = [p for p in periods if _parse_iso(p.get("startTime") or "") and _parse_iso(p["startTime"]) >= now]
                chosen = upcoming[0] if upcoming else periods[0]

            short = (chosen.get("shortForecast") or "").strip()
            temp = chosen.get("temperature")
            unit = (chosen.get("temperatureUnit") or "F").upper()
            st = _parse_iso(chosen.get("startTime") or "")
            # Include the date the forecast period starts
            date_str = st.date().isoformat() if st else now.date().isoformat()

            if short and isinstance(temp, (int, float)):
                return f"{short.lower()} with a temperature of {int(round(temp))} degrees on {date_str}."
            elif short:
                return f"{short.lower()} on {date_str}."
            else:
                return f"Weather information is currently unavailable on {date_str}."
        except Exception as e:
            logger.exception("lookup_weather failed")
            # Signal a service issue so the agent can inform the user
            raise RuntimeError("Weather service is unavailable") from e


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        llm=openai.LLM(
            model=os.getenv("LLM_MODEL", "gpt-oss:20b"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.88.16:11434/v1"),
        ),
        stt=deepgram.STT(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "multi"),
        ),
        tts=cartesia.TTS(
            voice=os.getenv("CARTESIA_VOICE", "6f84f4b8-58a2-430c-8c79-688dad597532")
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=os.getenv("PREEMPTIVE_GENERATION", "true").lower() == "true",
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))