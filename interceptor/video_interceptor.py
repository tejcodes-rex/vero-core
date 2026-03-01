import threading
import time
import cv2
import mss
import numpy as np
import logging
import mediapipe as mp
from interceptor.buffer_manager import BufferManager

logger = logging.getLogger("VideoInterceptor")

class VideoInterceptor(threading.Thread):
    def __init__(self, fps=15, frame_size=(224, 224), window_size=10):
        super().__init__(daemon=True)
        self.fps = fps
        self.frame_size = frame_size
        self.buffer = BufferManager(max_size=window_size)
        self.running = False
        self.roi = None
        self.cap = None
        self.sync_event = threading.Event()
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def run(self):
        self.running = True
        logger.info("Using video source: Desktop Screen Capture (MSS)")

        try:
            with mss.mss() as sct:
                # Get the primary monitor
                monitor = sct.monitors[1]
                logger.info(f"Screen resolution detected: {monitor['width']}x{monitor['height']}")
                
                while self.running:
                    start_time = time.perf_counter()
                    try:
                        # Grab the targeted region (or entire screen)
                        capture_region = self.roi if self.roi else monitor
                        sct_img = sct.grab(capture_region)
                        frame = np.array(sct_img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    except Exception as e:
                        logger.error(f"Frame Capture Error: {e}")
                        time.sleep(0.1)
                        continue

                    # Preprocess: Smart Autonomous Tracking (EMA Smoothed)
                    # We ALWAYS run face detection to track the face perfectly.
                    # If ROI is set, we are tracking the face INSIDE the manual square.
                    # If ROI is not set, we are tracking the face across the entire monitor.
                    results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ih, iw, _ = frame.shape
                    
                    target_detection = None
                    if results.detections:
                        # 1. Target Priority: Largest face usually means the primary speaker
                        # Sort by area descending
                        detections = sorted(results.detections, 
                                         key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, 
                                         reverse=True)
                        
                        # 2. Identity Lock: If we were already tracking a face, stay near it
                        if getattr(self, "smoothed_bbox", None) is not None:
                            prev_center = self.smoothed_bbox[:2]
                            best_dist = float('inf')
                            for d in detections:
                                bbox = d.location_data.relative_bounding_box
                                cx = (bbox.xmin + bbox.width/2) * iw
                                cy = (bbox.ymin + bbox.height/2) * ih
                                dist = np.linalg.norm(np.array([cx, cy]) - prev_center)
                                if dist < best_dist and dist < (iw * 0.25):
                                    best_dist = dist
                                    target_detection = d
                        
                        # Fallback to largest face
                        if target_detection is None:
                            target_detection = detections[0]

                        bbox = target_detection.location_data.relative_bounding_box
                        x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
                        w, h = int(bbox.width * iw), int(bbox.height * ih)
                        
                        center_x, center_y = x + w // 2, y + h // 2
                        size = max(w, h)
                        size += int(size * 0.35) # Stable margin
                        
                        target_bbox = np.array([center_x, center_y, size], dtype=np.float32)
                        
                        if getattr(self, "smoothed_bbox", None) is None:
                            self.smoothed_bbox = target_bbox
                        else:
                            dist = np.linalg.norm(self.smoothed_bbox[:2] - target_bbox[:2])
                            alpha = 0.6 if dist > int(iw * 0.15) else 0.10
                            self.smoothed_bbox = self.smoothed_bbox * (1 - alpha) + target_bbox * alpha
                            
                        s_cx, s_cy, s_sz = [int(v) for v in self.smoothed_bbox]
                        half_sz = s_sz // 2
                        x1, y1 = max(0, s_cx - half_sz), max(0, s_cy - half_sz)
                        x2, y2 = min(iw, s_cx + half_sz), min(ih, s_cy + half_sz)
                        
                        # Aspect-Ratio Correct Square Crop (Letterboxing)
                        # Prevents squashing faces near screen edges which triggers AI_V: 1.0
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            ch, cw, _ = cropped.shape
                            if ch != cw:
                                # Create a black square base
                                side = max(ch, cw)
                                square_crop = np.zeros((side, side, 3), dtype=np.uint8)
                                # Center the original crop in the square
                                dy = (side - ch) // 2
                                dx = (side - cw) // 2
                                square_crop[dy:dy+ch, dx:dx+cw] = cropped
                                frame = cv2.resize(square_crop, self.frame_size)
                            else:
                                frame = cv2.resize(cropped, self.frame_size)
                        else:
                            frame = cv2.resize(frame, self.frame_size)
                    else:
                        frame = cv2.resize(frame, self.frame_size)
                        self.smoothed_bbox = None 
                    
                    self.buffer.push(frame)
                    
                    if len(self.buffer.buffer) % 5 == 0:
                        logger.info(f"Video Flow Active: {len(self.buffer.buffer)}/10 frames in buffer")

                    # Maintain FPS with high-precision Event wait
                    elapsed = time.perf_counter() - start_time
                    wait_time = max(0, (1.0 / self.fps) - elapsed)
                    if self.sync_event.wait(timeout=wait_time):
                        break
        except Exception as e:
            logger.error(f"Global Video Interceptor Failure: {e}")
        finally:
            if self.cap: self.cap.release()

    def stop(self):
        self.running = False
        self.sync_event.set()

    def set_roi(self, x, y, w, h):
        self.roi = {"top": y, "left": x, "width": w, "height": h}
        logger.info(f"Video ROI updated: {self.roi}")

    def get_window(self) -> np.ndarray:
        return self.buffer.get_window()

    def is_ready(self) -> bool:
        return self.buffer.is_ready()
