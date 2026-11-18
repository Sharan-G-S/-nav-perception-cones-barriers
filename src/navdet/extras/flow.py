"""Sparse optical flow (Lucas-Kanade) used to carry detections across frames and keep track identities."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from navdet.detect import COLORS, Detection

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}


def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def flow_points(
    prev_gray: np.ndarray, gray: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moves each (x, y) point from the previous frame into the current one; ok is False where tracking failed."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
    return moved.reshape(-1, 2), status.reshape(-1).astype(bool)


def dense_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


@dataclass
class Track:
    track_id: int
    label: str
    points: list[tuple[float, float]] = field(default_factory=list)
    lost: int = 0
    last_detection: Detection | None = None

    @property
    def position(self) -> tuple[float, float]:
        return self.points[-1]


class FlowTracker:
    """Predicts every track with optical flow, then snaps it to the nearest new detection of the same label."""

    def __init__(self, max_lost: int = 5, match_dist_px: float = 40.0):
        self.max_lost = max_lost
        self.match_dist_px = match_dist_px
        self.tracks: list[Track] = []
        self.next_id = 1
        self.prev_gray: np.ndarray | None = None

    def predict(self, gray: np.ndarray) -> list[tuple[float, float]]:
        if self.prev_gray is None or not self.tracks:
            return [t.position for t in self.tracks]
        moved, ok = flow_points(self.prev_gray, gray, np.float32([t.position for t in self.tracks]))
        return [
            tuple(map(float, m)) if good else t.position
            for t, m, good in zip(self.tracks, moved, ok)
        ]

    def update(self, frame: np.ndarray, detections: list[Detection]) -> list[Track]:
        gray = to_gray(frame)
        predicted = self.predict(gray)
        unmatched = list(range(len(detections)))
        for track, guess in zip(self.tracks, predicted):
            best, best_dist = None, self.match_dist_px
            for i in unmatched:
                det = detections[i]
                if det.label != track.label:
                    continue
                dist = float(np.hypot(det.box.center[0] - guess[0], det.box.center[1] - guess[1]))
                if dist < best_dist:
                    best, best_dist = i, dist
            if best is None:
                track.points.append(guess)
                track.lost += 1
            else:
                unmatched.remove(best)
                track.points.append(detections[best].box.center)
                track.last_detection = detections[best]
                track.lost = 0
        for i in unmatched:
            det = detections[i]
            self.tracks.append(Track(self.next_id, det.label, [det.box.center], 0, det))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        self.prev_gray = gray
        return list(self.tracks)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        color = COLORS.get(track.label, (255, 255, 255))
        pts = np.int32(np.round(track.points)).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
        x, y = (round(v) for v in track.position)
        cv2.circle(out, (x, y), 4, color, -1)
        caption = track.last_detection.caption() if track.last_detection else track.label
        cv2.putText(
            out,
            f"#{track.track_id} {caption}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out
