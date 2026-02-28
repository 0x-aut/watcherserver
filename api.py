from fastapi import APIRouter, HTTPException
from posture_processor import PostureEvent

# An in-memory store for sessions
posture_sessions: dict[str, dict] = {}

router = APIRouter(prefix="/posture", tags=["posture"])

def update_session_from_event(call_id: str, event: PostureEvent) -> None:
  """
  Called from main.py whenever a PostureEvent fires.
  Upserts the in-memory session dict for this call_id.
  """
  posture_sessions[call_id] = {
    "score": event.session_score,
    "good_seconds": event.good_seconds,
    "bad_seconds": event.bad_seconds,
    "duration_seconds": event.duration_seconds,
    "total_frames": 0,  # not tracked here — processor owns that
    "last_posture_ok": event.posture_ok,
    "last_issues": event.issues,
    "last_neck_angle": event.neck_angle,
    "last_shoulder_slope": event.shoulder_slope,
    "keypoints_visible": event.keypoints_visible,
    "ended": posture_sessions.get(call_id, {}).get("ended", False),
  }
  
@router.get("/{call_id}/stats")
async def get_posture_stats(call_id: str):
  """
  Polled by Next.js every 2 seconds during an active session.
  Returns live posture score and breakdown.
  """
  session = posture_sessions.get(call_id)
  if not session:
    return {
      "call_id": call_id,
      "score": 100,
      "good_seconds": 0,
      "bad_seconds": 0,
      "duration_seconds": 0,
      "last_posture_ok": True,
      "last_issues": [],
      "keypoints_visible": False,
      "ended": False,
    }
  return {
    "call_id": call_id,
    "score": session["score"],
    "good_seconds": session["good_seconds"],
    "bad_seconds": session["bad_seconds"],
    "duration_seconds": session["duration_seconds"],
    "last_posture_ok": session["last_posture_ok"],
    "last_issues": session["last_issues"],
    "last_neck_angle": session.get("last_neck_angle", 0.0),
    "last_shoulder_slope": session.get("last_shoulder_slope", 0.0),
    "keypoints_visible": session.get("keypoints_visible", False),
    "ended": session["ended"],
  }
  
@router.post("/{call_id}/end")
async def end_posture_session(call_id: str):
  """
  Called by the frontend when the user clicks End Session.
  Freezes the session stats and returns the final summary.
  The caller should also DELETE /sessions/{session_id} to stop the agent.
  """
  session = posture_sessions.get(call_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")
    
  session["ended"] = True
  score = session["score"]
  
  if score >= 90:
    grade = "Excellent"
    message = "Outstanding session. Your posture was consistently great."
  elif score >= 70:
    grade = "Good"
    message = "Solid session. A few moments to watch, but mostly on point."
  elif score >= 50:
    grade = "Fair"
    message = "Room to improve. Keep an eye on your neck and shoulders."
  elif score >= 30:
    grade = "Poor"
    message = "Quite a few posture alerts this session. Worth focusing on tomorrow."
  else:
    grade = "Needs work"
    message = "Your posture struggled today. Try setting a 30-minute posture reminder."
    
  return {
    "call_id": call_id,
    "score": score,
    "grade": grade,
    "message": message,
    "good_seconds": session["good_seconds"],
    "bad_seconds": session["bad_seconds"],
    "duration_seconds": session["duration_seconds"],
    "top_issues": session["last_issues"],
  }
  
@router.delete("/{call_id}")
async def clear_posture_session(call_id: str):
  """Clears session data from memory. Call after the summary page is shown."""
  posture_sessions.pop(call_id, None)
  return {"cleared": True}
  
