"""
Multi-Person Right Leg Detector — Working Version
Uses segmentation + connected components to find multiple people,
then runs MediaPipe Pose on each person's region.
"""

import cv2
import numpy as np
import math
import time
from typing import Tuple, Optional, Dict, Any, List

try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import selfie_segmentation as mp_selfie
except ImportError:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_selfie = mp.solutions.selfie_segmentation

# ─── MediaPipe landmark indices ────────────────────────────────────
_R_HIP, _R_KNEE, _R_ANKLE, _R_HEEL, _R_FOOT = 24, 26, 28, 30, 32
_L_HIP, _L_KNEE, _L_ANKLE = 23, 25, 27

_RIGHT_LEG_INDICES = [_R_HIP, _R_KNEE, _R_ANKLE, _R_HEEL, _R_FOOT]

# EMA smoothing factor
_ALPHA_EMA = 0.35

# Minimum pixel area for a refined mask to be considered valid
_MIN_REFINED_MASK_PX = 200

# Reference hip-to-ankle distance (normalised) for scale export
_REF_HIP_ANKLE_DIST = 0.45


class MultiPersonRightLegDetector:
    """
    Detects and highlights right legs of multiple people.
    Same color for all right legs.
    """

    # ── colour palette (BGR) ───────────────────────────────────────
    LEG_COLOR   = (255, 180, 100)   # highlighted leg - SAME for all people
    BODY_COLOR  = (60, 60, 60)      # rest-of-body silhouette
    BG_COLOR    = (0, 0, 0)         # black background

    def __init__(self, static_mode: bool = False):
        self.static_mode = static_mode

        # Single pose detector (will be reused for each person)
        self.pose = mp_pose.Pose(
            static_image_mode=True,  # Use static mode for better detection on cropped regions
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)

        # ── temporal state for tracking people ─────────────────────
        self._person_states = {}  # person_id -> state dict
        self._next_person_id = 0
        self._start_time = time.time()
        self._frame_index = 0

        # ── cached segmentation result ─────────────────────────────
        self._cached_seg_mask: Optional[np.ndarray] = None
        self._cached_seg_frame_id: int = -1

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, debug: bool = False
                      ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process one BGR frame and detect all people.

        Returns
        -------
        rendered : np.ndarray        — visualisation frame
        gait_data_list : list[dict]  — list of gait data, one per person
        """
        if frame is None:
            return frame, []

        self._frame_index += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 1. Get body segmentation mask ──────────────────────────
        seg_mask = self._get_seg_mask(rgb)

        if seg_mask is None:
            # No segmentation available
            return self._render_empty(h, w), []

        # ── 2. Find individual people using connected components ───
        person_regions = self._find_people_regions(seg_mask, w, h)

        if not person_regions:
            # No people detected
            rendered = self._render_no_detection(frame, seg_mask, w, h)
            if debug:
                cv2.putText(rendered, "No people detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return rendered, []

        # ── 3. Process each person ─────────────────────────────────
        all_leg_masks = []
        all_gait_data = []

        for person_idx, (bbox, person_mask) in enumerate(person_regions):
            x1, y1, x2, y2 = bbox
            
            # Extract person's region from original frame
            person_crop = rgb[y1:y2, x1:x2].copy()
            
            if person_crop.size == 0:
                continue

            # Run pose detection on this person's region
            pose_res = self.pose.process(person_crop)

            if not pose_res.pose_landmarks:
                continue

            lms = pose_res.pose_landmarks.landmark
            
            # Get or create person ID
            person_id = self._match_or_create_person(lms, bbox)

            # Detect side and extract landmarks
            side = self._detect_and_lock_side(lms, person_id)
            raw = self._extract_landmarks(lms, side)
            smoothed = self._smooth_landmarks(raw, person_id)

            # Convert normalized coords to full frame coords
            smoothed_full = [
                ((x * (x2 - x1) + x1) / w, (y * (y2 - y1) + y1) / h, v)
                for x, y, v in smoothed
            ]

            # Visibility check
            visible = [(x, y) for x, y, v in smoothed_full if v > 0.3]
            if len(visible) < 3:
                continue

            # Build leg mask in full frame coordinates
            leg_mask = self._build_leg_mask(smoothed_full, w, h)
            
            # Refine with person's segmentation mask
            leg_mask = self._refine_mask_with_person(leg_mask, person_mask)

            # Calculate metrics
            confidence = self._calc_confidence(smoothed_full, leg_mask)
            gait = self._calc_gait_data(smoothed_full, confidence, w, h, person_id)

            all_leg_masks.append(leg_mask)
            all_gait_data.append(gait)

        # ── 4. Render all people ───────────────────────────────────
        rendered = self._render_multi_person(frame, all_leg_masks, seg_mask, w, h)

        if debug:
            num_people = len(all_gait_data)
            avg_conf = sum(g['confidence'] for g in all_gait_data) / max(num_people, 1) if all_gait_data else 0.0
            cv2.putText(rendered, f"People: {num_people}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(rendered, f"Avg Confidence: {avg_conf:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rendered, "RIGHT LEGS", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.LEG_COLOR, 2)

        return rendered, all_gait_data

    # ──────────────────────────────────────────────────────────────
    # PERSON DETECTION using Connected Components
    # ──────────────────────────────────────────────────────────────

    def _find_people_regions(self, seg_mask: np.ndarray, w: int, h: int
                            ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Find individual people in the segmentation mask using connected components.
        
        Returns: List of (bbox, person_mask) tuples
        """
        # Find connected components in the segmentation mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_mask, connectivity=8
        )

        person_regions = []

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area (minimum 2000 pixels to be considered a person)
            if area < 2000:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Add padding around bbox
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + width + padding)
            y2 = min(h, y + height + padding)

            # Create mask for this person only
            person_mask = (labels == i).astype(np.uint8) * 255

            bbox = (x1, y1, x2, y2)
            person_regions.append((bbox, person_mask))

        return person_regions

    def _match_or_create_person(self, lms, bbox: Tuple[int, int, int, int]) -> int:
        """Match detected person to existing tracked person or create new ID."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Try to match with existing people based on bbox center
        min_distance = float('inf')
        matched_id = None

        for person_id, state in self._person_states.items():
            if 'last_center' in state:
                last_x, last_y = state['last_center']
                distance = math.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                if distance < 100 and distance < min_distance:  # 100 pixel threshold
                    min_distance = distance
                    matched_id = person_id

        if matched_id is not None:
            self._person_states[matched_id]['last_center'] = (center_x, center_y)
            return matched_id

        # Create new person
        new_id = self._next_person_id
        self._next_person_id += 1
        self._person_states[new_id] = {
            'last_center': (center_x, center_y),
            'prev_landmarks': None,
            'locked_side': None,
            'stride_count': 0,
            'prev_ankle_y': None,
        }
        return new_id

    # ──────────────────────────────────────────────────────────────
    # SIDE DETECTION & LANDMARK EXTRACTION
    # ──────────────────────────────────────────────────────────────

    def _detect_and_lock_side(self, lms, person_id: int) -> str:
        """Detect which side is the right leg for this person."""
        state = self._person_states[person_id]

        r_hip_x = lms[_R_HIP].x
        l_hip_x = lms[_L_HIP].x

        if state['locked_side'] is None:
            # MediaPipe RIGHT hip has smaller x when person faces camera
            state['locked_side'] = 'right' if r_hip_x < l_hip_x else 'left'

        return state['locked_side']

    def _extract_landmarks(self, lms, side: str) -> List[Tuple[float, float, float]]:
        """Extract landmarks for the specified side."""
        if side == 'right':
            indices = _RIGHT_LEG_INDICES
        else:
            indices = [_L_HIP, _L_KNEE, _L_ANKLE, _R_HEEL, _R_FOOT]

        out = []
        for idx in indices:
            lm = lms[idx]
            out.append((lm.x, lm.y, lm.visibility))
        return out

    # ──────────────────────────────────────────────────────────────
    # TEMPORAL SMOOTHING
    # ──────────────────────────────────────────────────────────────

    def _smooth_landmarks(self, raw: List[Tuple[float, float, float]],
                          person_id: int) -> List[Tuple[float, float, float]]:
        """Apply EMA smoothing per person."""
        state = self._person_states[person_id]

        if state['prev_landmarks'] is None or len(state['prev_landmarks']) != len(raw):
            state['prev_landmarks'] = list(raw)
            return list(raw)

        smoothed = []
        for (rx, ry, rv), (px, py, _pv) in zip(raw, state['prev_landmarks']):
            sx = px + _ALPHA_EMA * (rx - px)
            sy = py + _ALPHA_EMA * (ry - py)
            smoothed.append((sx, sy, rv))

        state['prev_landmarks'] = smoothed
        return smoothed

    # ──────────────────────────────────────────────────────────────
    # SEGMENTATION
    # ──────────────────────────────────────────────────────────────

    def _get_seg_mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Run segmentation once per frame; cache result."""
        if self._cached_seg_frame_id == self._frame_index:
            return self._cached_seg_mask

        res = self.selfie_seg.process(rgb)
        if res.segmentation_mask is not None:
            self._cached_seg_mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
        else:
            self._cached_seg_mask = None
        self._cached_seg_frame_id = self._frame_index
        return self._cached_seg_mask

    # ──────────────────────────────────────────────────────────────
    # MASK CONSTRUCTION
    # ──────────────────────────────────────────────────────────────

    def _build_leg_mask(self, landmarks: List[Tuple[float, float, float]],
                        w: int, h: int) -> np.ndarray:
        """Build ordered polygon leg mask."""
        mask = np.zeros((h, w), dtype=np.uint8)

        pts = []
        for x, y, v in landmarks:
            if v > 0.2:
                pts.append(np.array([x * w, y * h]))
            else:
                pts.append(None)

        pts = self._interpolate_missing(pts)
        if pts is None or len(pts) < 3:
            return mask

        hip, ankle = pts[0], pts[2]
        hip_ankle_dist = np.linalg.norm(ankle - hip)
        leg_half_w = max(int(hip_ankle_dist * 0.18), 6)

        normal_offsets = []
        for i in range(len(pts)):
            nxt = pts[min(i + 1, len(pts) - 1)]
            prv = pts[max(i - 1, 0)]
            tangent = nxt - prv
            length = np.linalg.norm(tangent)
            if length < 1e-6:
                normal_offsets.append(np.array([leg_half_w, 0]))
            else:
                tangent /= length
                normal = np.array([-tangent[1], tangent[0]])
                normal_offsets.append(normal * leg_half_w)

        right_side = [pts[i] + normal_offsets[i] for i in range(len(pts))]
        left_side  = [pts[i] - normal_offsets[i] for i in reversed(range(len(pts)))]

        polygon = np.array(right_side + left_side, dtype=np.int32)
        polygon = polygon.reshape(-1, 1, 2)

        cv2.fillPoly(mask, [polygon], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    @staticmethod
    def _interpolate_missing(pts: list) -> Optional[list]:
        """Linear interpolation for None entries."""
        n = len(pts)
        first = next((i for i in range(n) if pts[i] is not None), None)
        last  = next((i for i in range(n - 1, -1, -1) if pts[i] is not None), None)
        if first is None or last is None:
            return None

        for i in range(first):
            pts[i] = pts[first].copy()
        for i in range(last + 1, n):
            pts[i] = pts[last].copy()

        i = first
        while i <= last:
            if pts[i] is None:
                j = i
                while j <= last and pts[j] is None:
                    j += 1
                for k in range(i, j):
                    t = (k - (i - 1)) / (j - (i - 1))
                    pts[k] = pts[i - 1] * (1 - t) + pts[j] * t
                i = j
            else:
                i += 1
        return pts

    def _refine_mask_with_person(self, leg_mask: np.ndarray,
                                  person_mask: np.ndarray) -> np.ndarray:
        """Intersect leg mask with this person's segmentation mask."""
        refined = cv2.bitwise_and(leg_mask, person_mask)
        if np.count_nonzero(refined) < _MIN_REFINED_MASK_PX:
            return leg_mask
        return refined

    # ──────────────────────────────────────────────────────────────
    # RENDERING
    # ──────────────────────────────────────────────────────────────

    def _render_multi_person(self, frame: np.ndarray, leg_masks: List[np.ndarray],
                             seg_mask: np.ndarray, w: int, h: int) -> np.ndarray:
        """Render all people with SAME COLOR for all right legs."""
        output = np.full((h, w, 3), self.BG_COLOR, dtype=np.uint8)

        # Combine all leg masks into one
        combined_leg_mask = np.zeros((h, w), dtype=np.uint8)
        for leg_mask in leg_masks:
            combined_leg_mask = cv2.bitwise_or(combined_leg_mask, leg_mask)

        # Body silhouette (dark grey), excluding all legs
        body_no_legs = cv2.bitwise_and(seg_mask, cv2.bitwise_not(combined_leg_mask))
        output[body_no_legs > 0] = list(self.BODY_COLOR)

        # All right legs in SAME color with smooth blending
        blurred_mask = cv2.GaussianBlur(combined_leg_mask, (11, 11), 0).astype(np.float32) / 255.0
        alpha = blurred_mask[..., None]
        leg_layer = np.full((h, w, 3), self.LEG_COLOR, dtype=np.float32)
        output = (output.astype(np.float32) * (1.0 - alpha) +
                  leg_layer * alpha).astype(np.uint8)

        return output

    def _render_no_detection(self, frame: np.ndarray, seg_mask: np.ndarray,
                             w: int, h: int) -> np.ndarray:
        """Render when no legs detected."""
        output = np.full((h, w, 3), self.BG_COLOR, dtype=np.uint8)
        if seg_mask is not None:
            output[seg_mask > 0] = list(self.BODY_COLOR)
        return output

    def _render_empty(self, h: int, w: int) -> np.ndarray:
        """Render empty black frame."""
        return np.full((h, w, 3), self.BG_COLOR, dtype=np.uint8)

    # ──────────────────────────────────────────────────────────────
    # GAIT METRICS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at point B."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def _calc_confidence(self, landmarks: List[Tuple[float, float, float]],
                         mask: np.ndarray) -> float:
        """Calculate confidence score."""
        visibilities = [v for _, _, v in landmarks if v > 0.2]
        if not visibilities:
            return 0.0
        vis_score = np.mean(visibilities)
        mask_score = min(np.count_nonzero(mask) / 5000.0, 1.0)
        return float(round(vis_score * 0.7 + mask_score * 0.3, 3))

    def _calc_gait_data(self, landmarks: List[Tuple[float, float, float]],
                        confidence: float, w: int, h: int, person_id: int) -> Dict[str, Any]:
        """Compute gait data for one person."""
        state = self._person_states[person_id]

        hip    = np.array([landmarks[0][0] * w, landmarks[0][1] * h])
        knee   = np.array([landmarks[1][0] * w, landmarks[1][1] * h])
        ankle  = np.array([landmarks[2][0] * w, landmarks[2][1] * h])

        knee_angle = self._angle_between(hip, knee, ankle)

        hip_ankle_dist = float(np.linalg.norm(ankle - hip))
        scale_factor = hip_ankle_dist / (_REF_HIP_ANKLE_DIST * max(w, h)) if max(w, h) > 0 else 1.0

        # Stride detection
        ankle_y_norm = landmarks[2][1]
        if state['prev_ankle_y'] is not None:
            if state['prev_ankle_y'] < 0.5 and ankle_y_norm >= 0.5:
                state['stride_count'] += 1
        state['prev_ankle_y'] = ankle_y_norm

        timestamp = round(time.time() - self._start_time, 3)

        return {
            "person_id":         person_id,
            "status":            "right leg detected",
            "confidence":        confidence,
            "frame_index":       self._frame_index,
            "timestamp_s":       timestamp,
            "knee_angle_deg":    round(knee_angle, 2),
            "hip_ankle_dist_px": round(hip_ankle_dist, 2),
            "scale_factor":      round(scale_factor, 3),
            "stride_count":      state['stride_count'],
            "landmarks_px": {
                "hip":   [round(hip[0], 1),   round(hip[1], 1)],
                "knee":  [round(knee[0], 1),  round(knee[1], 1)],
                "ankle": [round(ankle[0], 1), round(ankle[1], 1)],
            },
        }

    # ──────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────

    def close(self):
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
            self.pose = None
        if hasattr(self, 'selfie_seg') and self.selfie_seg:
            self.selfie_seg.close()
            self.selfie_seg = None

    def __del__(self):
        self.close()