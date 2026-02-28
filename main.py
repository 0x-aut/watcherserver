from dotenv import load_dotenv
import logging

from vision_agents.core import Agent, AgentLauncher, User, Runner, ServeOptions
from vision_agents.plugins import getstream, gemini, elevenlabs, smart_turn

from posture_processor import PostureProcessor, PostureEvent

from api import router as posture_router, update_session_from_event

load_dotenv()
logging.basicConfig(level=logging.INFO)

async def create_agent(**kwargs) -> Agent:
  processor = PostureProcessor(fps=5, model_path="yolo26n-pose.pt")
  return Agent(
    edge=getstream.Edge(),
    agent_user=User(name="watcher", id="watcher-agent"),
    instructions=open("WatcherInstructions.md").read(),
    llm=gemini.Realtime(fps=3),
    stt=elevenlabs.STT(),
    tts=elevenlabs.TTS(),
    turn_detection=smart_turn.TurnDetection(),
    processors=[
      processor
    ],
  )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
  call = await agent.create_call(call_type, call_id)
  async with agent.join(call):
    # We want to update the in-memory store and event on each frame captured
    async def on_posture_event(event: PostureEvent) -> None:
      update_session_from_event(call_id, event)
    
    agent.events.subscribe(on_posture_event)
    await agent.simple_response(
      "Greet the user warmly, tell them you are watching their posture "
      "and will check in if you notice anything. Keep it to two sentences."
    )
    await agent.wait_for_participant()
    

runner = Runner(
  AgentLauncher(
    create_agent=create_agent,
    join_call=join_call,
    max_sessions_per_call=1,
    agent_idle_timeout=120.0,
    max_session_duration_seconds=7200,
  ),
  serve_options=ServeOptions(
    cors_allow_origins=["*"],
    cors_allow_methods=["GET", "POST", "DELETE"],
    cors_allow_headers=["*"],
    cors_allow_credentials=True,
  ),
)

runner.fast_api.include_router(posture_router)


if __name__ == "__main__":
  runner.cli()