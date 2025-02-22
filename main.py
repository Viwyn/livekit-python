import asyncio
from os import getenv
from dotenv import load_dotenv
import json

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import google, silero

load_dotenv()

async def entrypoint(ctx: JobContext):

    try:
        with open(getenv("GOOGLE_APPLICATION_CREDENTIALS"), "r") as f:
            google_credentials = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load Google credentials: {e}")

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by Riaru. Your interface with users will be voice. "
            "You should use short and concise responses whilst speaking like an anime girl, avoid using unpronouncable punctuation"
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=google.STT(
            model="latest_short",
            spoken_punctuation=True,
            credentials_info=google_credentials
        ),
        llm=google.LLM(
            model="gemini-2.0-flash-exp",
            temperature="0.8",
        ),
        tts=google.TTS(
            gender="female",
            voice_name="en-US-Standard-H",
        ),
        chat_ctx=initial_ctx
    )
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, how may I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
