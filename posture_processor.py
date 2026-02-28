# This file is to calculate the correct pose using the YOLO points gotten from the model

import math, time, aiortc, av, cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from vision_agents.core.events import BaseEvent
from vision_agents.core.processors import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins import ultralytics

# KEYPOINT INDICES
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# THRESHOLDS
CONFIDENCE_MIN = 0.5          # ignore keypoints below this visibility score

NECK_ANGLE_WARN = 15.0        # degrees from vertical — mild warning
NECK_ANGLE_BAD = 25.0         # degrees — flag as bad posture

SHOULDER_SLOPE_WARN = 5.0     # degrees of tilt between shoulders — mild
SHOULDER_SLOPE_BAD = 10.0     # degrees — flag as bad

HEAD_OFFSET_WARN = 0.15       # normalised ratio of shoulder width — mild
HEAD_OFFSET_BAD = 0.25        # ratio — flag as bad

SPINE_LEAN_WARN = 0.10        # normalised ratio of shoulder width — mild
SPINE_LEAN_BAD = 0.20         # ratio — flag as bad

# At fps=5, 25 frames = 5 seconds of sustained bad posture before flagging.
# This prevents false positives from single-frame noise or momentary shifts.
BAD_FRAME_STREAK_THRESHOLD = 25

# DATA STRUCTURES
@dataclass
class PostureEvent(BaseEvent):
  """
  Emitted after every processed frame
  For subscription in main.py to feed the FastAPI stat endpoint
  """
  type: str = "PostureEvent"
  posture_ok: bool = True
  neck_angle: float = 0.0
  shoulder_slope: float = 0.0
  head_offset: float = 0.0
  spine_lean: float = 0.0
  issues: list = field(default_factory=list)
  session_score: int = 100
  good_seconds: int = 0
  bad_seconds: int = 0
  duration_seconds: int = 0
  keypoints_visible: bool = True
  llm_context: str = ""
  
  
@dataclass
class PostureMetrics:
  """Raw computed metrics from a single frame."""
  neck_angle: float = 0.0
  shoulder_slope: float = 0.0
  head_offset: float = 0.0
  spine_lean: float = 0.0
  posture_ok: bool = True
  issues: list = field(default_factory=list)
  confidence: float = 1.0
  keypoints_visible: bool = True
  
  
@dataclass
class SessionStats:
  """Running totals for the current session"""
  start_time: float = field(default_factory=time.time)
  good_frames: int = 0
  bad_frames: int = 0
  total_frames: int = 0
  
  def record(self, posture_ok: bool) -> None:
    self.total_frames += 1
    if posture_ok:
      self.good_frames += 1
    else:
      self.bad_frames += 1
      
  @property
  def score(self) -> int:
    if self.total_frames == 0:
      return 100
    return round((self.good_frames / self.total_frames) * 100)
    
  @property
  def good_seconds(self) -> int:
    return self.good_frames // 5 # Here tweak figure to be equal to FPS, Hence fps here is 5
    
  @property
  def bad_seconds(self) -> int:
    return self.bad_frames // 5 # Here tweak figure to be equal to FPS, Hence fps here is 5
    
  @property
  def duration_seconds(self) -> int:
    return int(time.time() - self.start_time)
    
# MATH HELPERS
# All calculations work on actual pixel coords
# Always denormalise (multiply by frame w/h) before passing into these
def _get_point(
  x_list: list,
  y_list: list,
  vis_list: list,
  idx: int,
  frame_w: int,
  frame_h: int,
) -> Optional[tuple[float, float]]:
  """
  Extract a keypoint as actual pixel coordinates.
  Returns None if confidence is below CONFIDENCE_MIN
  
  YOLO gives normalised x/y (0.0–1.0) in separate lists.
  We denormalise here so all downstream angle math uses real pixel
  distances — required so arctan2 / arccos aren't distorted by aspect ratio.
  """
  if idx >= len(x_list):
    return None
  if float(vis_list[idx]) < CONFIDENCE_MIN:
    return None
  return (float(x_list[idx]) * frame_w, float(y_list[idx]) * frame_h)

def calculate_angle_three_points(
  a: tuple[float, float],
  b: tuple[float, float],
  c: tuple[float, float],
) -> float:
  """
  Returns the angle in degrees at point B formed by vectors B -> A and B -> C.
  Uses dot product with arccos. Range: 0 to 180 degrees.
  """
  ax, ay = a[0] - b[0], a[1] - b[1]
  cx, cy = c[0] - b[0], c[1] - b[1]
  dot = ax * cx + ay * cy
  mag_a = math.sqrt(ax ** 2 + ay ** 2)
  mag_c = math.sqrt(cx ** 2 + cy ** 2)
  if mag_a == 0 or mag_c == 0:
    return 0.0
  cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
  return math.degrees(math.acos(cos_angle))
    
    
def calculate_neck_angle(
  x: list, y: list, vis: list, fw: int, fh: int,
) -> Optional[float]:
  """
  How far the head is tilting forward — ear relative to shoulder.
  
  Method: For each side, create a virtual point 100px directly above
  the shoulder. The angle at the shoulder between
  [vertical_above → shoulder → ear] is the neck tilt.
  Average both sides for a cleaner reading.
  
  0°    = ear directly above shoulder (ideal)
  15–25° = mild forward tilt
  25°+  = bad (tech neck)
  """
  results = []
  for ear_idx, shoulder_idx in [
    (KP_LEFT_EAR, KP_LEFT_SHOULDER),
    (KP_RIGHT_EAR, KP_RIGHT_SHOULDER),
  ]:
    ear = _get_point(x, y, vis, ear_idx, fw, fh)
    shoulder = _get_point(x, y, vis, shoulder_idx, fw, fh)
    if ear is None or shoulder is None:
      continue
    # Virtual point straight above the shoulder in pixel space
    vertical_above = (shoulder[0], shoulder[1] - 100)
    angle = calculate_angle_three_points(vertical_above, shoulder, ear)
    results.append(angle)
  
  return sum(results) / len(results) if results else None
  
def calculate_shoulder_slope(
    x: list, y: list, vis: list, fw: int, fh: int
) -> Optional[float]:
  """
  Tilt of the shoulder line — one shoulder higher than the other.

  Method: arctan2(y-delta, x-delta) between left and right shoulder
  pixel positions. Absolute value — magnitude not direction.

  Screen y increases downward, so a lower shoulder = higher y value.

  0°    = perfectly level
  5–10° = mild asymmetry / slouch to one side
  10°+  = significant tilt
  """
  left = _get_point(x, y, vis, KP_LEFT_SHOULDER, fw, fh)
  right = _get_point(x, y, vis, KP_RIGHT_SHOULDER, fw, fh)
  if left is None or right is None:
    return None
  dy = right[1] - left[1]
  dx = right[0] - left[0]
  if dx == 0:
    return 90.0
  return abs(math.degrees(math.atan2(dy, dx)))
  
def calculate_head_forward_offset(
    x: list, y: list, vis: list, fw: int, fh: int
) -> Optional[float]:
  """
  How far the head has pushed forward horizontally — "tech neck" translation.
  This is NOT tilt/rotation — it's the whole head moving forward of the torso.

  Method: average |ear_x - shoulder_x| on each side, normalised by
  shoulder width for resolution independence.

  0.0       = ear above shoulder (ideal)
  0.15–0.25 = mild forward head posture
  0.25+     = bad forward head posture
  """
  left_ear      = _get_point(x, y, vis, KP_LEFT_EAR, fw, fh)
  right_ear     = _get_point(x, y, vis, KP_RIGHT_EAR, fw, fh)
  left_shoulder = _get_point(x, y, vis, KP_LEFT_SHOULDER, fw, fh)
  right_shoulder = _get_point(x, y, vis, KP_RIGHT_SHOULDER, fw, fh)

  if left_shoulder is None or right_shoulder is None:
    return None

  shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
  if shoulder_width < 1:
    return None

  offsets = []
  if left_ear:
    offsets.append(abs(left_ear[0] - left_shoulder[0]))
  if right_ear:
    offsets.append(abs(right_ear[0] - right_shoulder[0]))

  if not offsets:
    return None

  return (sum(offsets) / len(offsets)) / shoulder_width
  
  
def calculate_spine_lean(
    x: list, y: list, vis: list, fw: int, fh: int
) -> Optional[float]:
  """
  Whether the whole torso is leaning forward — not just head but full upper body.

  Method: horizontal distance between shoulder midpoint and hip midpoint,
  normalised by shoulder width.

  0.0       = shoulders above hips (ideal)
  0.10–0.20 = mild forward lean
  0.20+     = significant lean

  Supporting signal — combine with neck angle for high-confidence verdict.
  """
  ls = _get_point(x, y, vis, KP_LEFT_SHOULDER, fw, fh)
  rs = _get_point(x, y, vis, KP_RIGHT_SHOULDER, fw, fh)
  lh = _get_point(x, y, vis, KP_LEFT_HIP, fw, fh)
  rh = _get_point(x, y, vis, KP_RIGHT_HIP, fw, fh)

  # if None in (ls, rs, lh, rh):
  #   return None // This isnt working for some reason
    
  if ls is None:
    return None
  if rs is None:
    return None
  if lh is None:
    return None
  if rh is None:
    return None
  
  shoulder_mid_x = (ls[0] + rs[0]) / 2
  hip_mid_x      = (lh[0] + rh[0]) / 2
  shoulder_width = abs(rs[0] - ls[0])

  if shoulder_width < 1:
    return None

  return abs(shoulder_mid_x - hip_mid_x) / shoulder_width
  

def evaluate_posture(
  neck:   Optional[float],
  slope:  Optional[float],
  offset: Optional[float],
  lean:   Optional[float],
) -> tuple[bool, list[str]]:
  """
  Combines all metrics into a posture_ok verdict + list of issue labels.

  Weighting:
  - Neck angle is the primary signal (counts double).
    Bad neck alone → bad posture.
  - Two or more secondary signals bad simultaneously → bad posture.
  - One secondary signal bad → warning only, not yet flagged.
  """
  issues = []
  bad_count = 0

  if neck is not None:
    if neck >= NECK_ANGLE_BAD:
      issues.append("neck_forward_severe")
      bad_count += 2          # primary signal — double weight
    elif neck >= NECK_ANGLE_WARN:
      issues.append("neck_forward_mild")
      bad_count += 1

  if slope is not None:
    if slope >= SHOULDER_SLOPE_BAD:
      issues.append("shoulder_tilt_severe")
      bad_count += 1
    elif slope >= SHOULDER_SLOPE_WARN:
      issues.append("shoulder_tilt_mild")

  if offset is not None:
    if offset >= HEAD_OFFSET_BAD:
      issues.append("head_forward_severe")
      bad_count += 1
    elif offset >= HEAD_OFFSET_WARN:
      issues.append("head_forward_mild")

  if lean is not None:
    if lean >= SPINE_LEAN_BAD:
      issues.append("spine_lean_severe")
      bad_count += 1
    elif lean >= SPINE_LEAN_WARN:
      issues.append("spine_lean_mild")

  posture_ok = bad_count < 2
  return posture_ok, issues
  
# MAIN PROCESSING CLASS
class PostureProcessor(VideoProcessorPublisher):
  """
  Vision Agents custom video processor for posture analysis.
  
  Extends VideoProcessorPublisher so the annotated frame (YOLO skeleton
  + posture overlay) is published back into the call — Gemini receives
  this annotated video, not the raw webcam feed.
  
  Emits PostureEvent after every processed frame so the FastAPI layer
  can serve live session stats without coupling to the processor directly.
  """
  
  name = "posture_processor"
  
  def __init__(self, fps:int = 5, model_path:str = "yolo26n-post.pt"):
    self.fps = fps
    self.model_path = model_path
    
    self._forwarder: Optional[VideoForwarder] = None
    self._video_track = QueuedVideoTrack()
    self._events = None
    
    # YOLO Model - Lazy loaded on first init to prevent blocking init
    self._model = None
    
    # Session analytics
    self.session = SessionStats()
    
    # Streak filter - Only report bad posture after sustained bad frames
    self._bad_streak = 0
    self._good_streak = 0
    
    # Smoothing buffers - Rolling average over the last 5 readings
    self._neck_buf: deque = deque(maxlen=5)
    self._slope_buf: deque = deque(maxlen=5)
    self._offset_buf: deque = deque(maxlen=5)
    self._lean_buf: deque = deque(maxlen=5)
    
  ## Vision agents lifecycle
  def attach_agent(self, agent) -> None:
    """
    Called by vision agents when the processor is attached to an agent.
    """
    self._events = agent.events
    self._events.register(PostureEvent)
    
  async def process_video(
    self,
    track: aiortc.VideoStreamTrack,
    participant_id: Optional[str],
    shared_forwarder: Optional[VideoForwarder] = None,
  ) -> None:
    """Called when a participant starts publishing video"""
    if self._forwarder:
      await self._forwarder.remove_frame_handler(self._process_frame)
      
    self._forwarder = shared_forwarder
    self._forwarder.add_frame_handler(
      self._process_frame,
      fps=float(self.fps),
      name="posture_processor"
    )
    
  def publish_video_track(self) -> aiortc.VideoStreamTrack:
    """Returns the annotated video track vision agent publishes into the call"""
    return self._video_track
  
  async def stop_processing(self) -> None:
    """Called when the participants video track is removed"""
    if self._forwarder:
      await self._forwarder.remove_frame_handler(self._process_frame)
      self._forwarder = None
      
  async def close(self) -> None:
    """Called when the agent shuts down"""
    await self.stop_processing()
    self._video_track.stop()
    
  ## Frame processing functions
  def _load_model(self) -> None:
    """Lazy load YOLO model on first frame to avoid blocking the event loop"""
    if self._model is None:
      from ultralytics import YOLO
      self._model = YOLO(self.model_path)
  
  async def _process_frame(self, frame: av.VideoFrame) -> None:
    """
    Main per-frame callback. Called by VideoForwarder at self.fps rate.
    
    Pipeline:
      av.VideoFrame → numpy RGB → YOLO pose → compute metrics →
      draw overlay → av.VideoFrame → QueuedVideoTrack → Gemini sees it
    """
    self._load_model()
    
    # Convert av.VideoFrame to Numpy RGB
    img_rgb = frame.to_ndarray(format="rgb24")
    fh, fw = img_rgb.shape[:2]
    
    # YOLO expects BGR (OpenCV convention)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if self._model is None:
      return None
    results = self._model(img_bgr, verbose=False, conf=0.5)
    result = results[0]
    
    # Extract keypoints for detected person
    kp_x, kp_y, kp_vis = None, None, None
    metrics = PostureMetrics()
    
    if result.keypoints is not None and len(result.keypoints) > 0:
      kps_obj = result.keypoints
      num_persons = len(kps_obj)
      
      if num_persons > 0:
        # conf. shape: (num_persons, 17)
        confs = kps_obj.conf.cpu().numpy()
        best_idx = int(np.argmax(confs.mean(axis=1)))
        
        # .xyn shape: (num_persons, 17, 2) - normalized 0.0-1.0
        xyn = kps_obj.xyn.cpu().numpy()[best_idx]
        conf = confs[best_idx]
        
        kp_x = xyn[:, 0].tolist()
        kp_y = xyn[:, 1].tolist()
        kp_vis = conf.tolist()
        
    # YOLO annotated frame
    annotated_bgr = result.plot(kpt_line=True, kpt_radius=4)
    
    # No person detected
    if kp_x is None:
      metrics.keypoints_visible = False
      metrics.posture_ok = True ## No penalties for being away from screen
      self._draw_overlay(annotated_bgr, metrics, fw, fh, no_person=True)
    elif kp_y is None:
      metrics.keypoints_visible = False
      metrics.posture_ok = True ## No penalties for being away from screen
      self._draw_overlay(annotated_bgr, metrics, fw, fh, no_person=True)
    elif kp_vis is None:
      metrics.keypoints_visible = False
      metrics.posture_ok = True ## No penalties for being away from screen
      self._draw_overlay(annotated_bgr, metrics, fw, fh, no_person=True)
    else:
      # Raw metrics
      raw_neck = calculate_neck_angle(kp_x, kp_y, kp_vis, fw, fh)
      raw_slope  = calculate_shoulder_slope(kp_x, kp_y, kp_vis, fw, fh)
      raw_offset = calculate_head_forward_offset(kp_x, kp_y, kp_vis, fw, fh)
      raw_lean   = calculate_spine_lean(kp_x, kp_y, kp_vis, fw, fh)
      
      # Smooth over rolling buffer
      neck = self._smooth(self._neck_buf,   raw_neck)
      slope = self._smooth(self._slope_buf,  raw_slope)
      offset = self._smooth(self._offset_buf, raw_offset)
      lean = self._smooth(self._lean_buf,   raw_lean)
      
      # Verdict for posture
      posture_ok, issues = evaluate_posture(neck, slope, offset, lean)
      
      # Streak filter
      if not posture_ok:
        self._bad_streak += 1
        self._good_streak = 0
      else:
        self._good_streak += 1
        self._bad_streak  = 0
      
      sustained_bad = self._bad_streak >= BAD_FRAME_STREAK_THRESHOLD
      
      # Keypoint visibility confidence
      key_indices = [
        KP_LEFT_EAR, KP_RIGHT_EAR,
        KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
        KP_LEFT_HIP, KP_RIGHT_HIP,
      ]
      visible = sum(
        1 for i in key_indices
        if float(kp_vis[i]) >= CONFIDENCE_MIN
      )
      
      # Populate metrics
      metrics.neck_angle = round(neck,   1) if neck   is not None else 0.0
      metrics.shoulder_slope = round(slope,  1) if slope  is not None else 0.0
      metrics.head_offset = round(offset, 3) if offset is not None else 0.0
      metrics.spine_lean = round(lean,   3) if lean   is not None else 0.0
      metrics.posture_ok = not sustained_bad
      metrics.issues = issues if sustained_bad else []
      metrics.confidence = visible / len(key_indices)
      
      # Update session analytics 
      self.session.record(metrics.posture_ok)
      
      # Draw overlay on annotated frame 
      self._draw_overlay(annotated_bgr, metrics, fw, fh)
      
    # Emit PostureEvent for FastAPI stats layer
    if self._events:
      await self._events.emit(PostureEvent(
        posture_ok = metrics.posture_ok,
        neck_angle = metrics.neck_angle,
        shoulder_slope = metrics.shoulder_slope,
        head_offset = metrics.head_offset,
        spine_lean = metrics.spine_lean,
        issues = metrics.issues,
        session_score = self.session.score,
        good_seconds = self.session.good_seconds,
        bad_seconds = self.session.bad_seconds,
        duration_seconds = self.session.duration_seconds,
        keypoints_visible = metrics.keypoints_visible,
        llm_context = self._format_for_llm(metrics),
      ))
      
    # Publish annotated frame back into the call 
    # Convert back to RGB for av.VideoFrame
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    out_frame = av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")
    await self._video_track.add_frame(out_frame)
    
  # Helpers
  def _smooth(self, buf: deque, value: Optional[float]) -> Optional[float]:
    """Rolling average. Eliminates single-frame jitter from YOLO noise."""
    if value is not None:
      buf.append(value)
    return (sum(buf) / len(buf)) if buf else None
  
  def _draw_overlay(
    self,
    frame: np.ndarray,
    metrics: PostureMetrics,
    fw: int,
    fh: int,
    no_person: bool = False,
  ) -> None:
    """
    Draws a clean status banner + metric readouts on the annotated frame.
    Green = good. Red = bad. Grey = no person.
    This is what Gemini sees AND what judges see in the demo.
    """
    if no_person:
      color = (128, 128, 128)
      label = "No person detected"
    elif metrics.posture_ok:
      color = (0, 180, 60)       # green (BGR)
      label = "Good posture"
    else:
      color = (30, 30, 210)      # red (BGR)
      label = "Posture alert"
      
    # Status banner
    cv2.rectangle(frame, (0, 0), (fw, 40), color, -1)
    cv2.putText(frame, label,
      (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
      (255, 255, 255), 2, cv2.LINE_AA)
    
    # Score top-right
    cv2.putText(frame, f"Score: {self.session.score}",
      (fw - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
      (255, 255, 255), 2, cv2.LINE_AA)
    
    
    # Metric readouts bottom-left
    if not no_person:
      lines = [
        f"Neck: {metrics.neck_angle:.1f}deg  (ok < {NECK_ANGLE_BAD})",
        f"Shoulders: {metrics.shoulder_slope:.1f}deg  (ok < {SHOULDER_SLOPE_BAD})",
        f"Head fwd: {metrics.head_offset:.2f}  (ok < {HEAD_OFFSET_BAD})",
        f"Spine: {metrics.spine_lean:.2f}  (ok < {SPINE_LEAN_BAD})",
      ]
      y0 = fh - 110
      for i, line in enumerate(lines):
        cv2.putText(frame, line,
          (10, y0 + i * 24),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
          (230, 230, 230), 1, cv2.LINE_AA)
        
  def _format_for_llm(self, metrics: PostureMetrics) -> str:
    """
    Concise string injected into Gemini's context alongside the video.
    Gives Gemini actual numbers so coaching is specific and credible.
    """
    if not metrics.keypoints_visible:
      return "[PostureData: no person visible]"
  
    status = "GOOD" if metrics.posture_ok else "BAD"
    issues = ", ".join(metrics.issues) if metrics.issues else "none"
  
    return (
      f"[PostureData] status={status} | "
      f"neck={metrics.neck_angle}deg | "
      f"shoulders={metrics.shoulder_slope}deg | "
      f"head_offset={metrics.head_offset:.2f} | "
      f"spine_lean={metrics.spine_lean:.2f} | "
      f"issues={issues} | "
      f"score={self.session.score} | "
      f"bad_seconds={self.session.bad_seconds}"
    )
  def get_session_stats(self) -> dict:
    """Plain dict for FastAPI — fallback if event subscription isn't wired."""
    return {
      "score": self.session.score,
      "good_seconds": self.session.good_seconds,
      "bad_seconds": self.session.bad_seconds,
      "duration_seconds": self.session.duration_seconds,
      "total_frames": self.session.total_frames,
    }
    
  def reset_session(self) -> None:
    """Reset all analytics. Call at the start of each new session."""
    self.session = SessionStats()
    self._bad_streak = 0
    self._good_streak = 0
    self._neck_buf.clear()
    self._slope_buf.clear()
    self._offset_buf.clear()
    self._lean_buf.clear()