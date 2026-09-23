"""
OC-SORT multi-object tracking algorithm.

Observation-Centric SORT improves on ByteTrack in two ways:
  1. Velocity direction consistency — lost tracks are penalised when the
     direction from their last observation to a candidate detection
     conflicts with their estimated motion direction.
  2. Observation-Centric Re-Update (ORU) — on re-association the Kalman
     state is re-initialised from the last *actual* observation and
     re-predicted forward, correcting drift accumulated during the gap.

Reference:
  Jiangmiao Pang et al.: OC-SORT: Observation-Centric SORT on Video
  Highlights Detection and Beyond. CVPR 2023.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

try:
    import lap  # type: ignore

    LAP_AVAILABLE = True
except ImportError:
    LAP_AVAILABLE = False
    lap = None

try:
    from cython_bbox import bbox_overlaps as bbox_ious  # type: ignore

    CYTHON_BBOX_AVAILABLE = True
except ImportError:
    CYTHON_BBOX_AVAILABLE = False
    bbox_ious = None

from .tracker_kalman import KalmanFilter, TrackState, _numpy_bbox_ious

logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    """Represents a tracked object with Kalman state and observation history."""

    track_id: int = 0
    _tlwh: np.ndarray = field(default_factory=lambda: np.zeros(4))
    det_score: float = 0.0
    state: TrackState = TrackState.New
    is_activated: bool = False
    frame_id: int = 0
    start_frame: int = 0

    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    kalman_filter: KalmanFilter | None = None
    min_hits: int = 2

    # OC-SORT: last two actual observations for velocity estimation and ORU
    _last_obs_tlbr: np.ndarray | None = None
    _last_obs_frame: int = -1
    _prev_obs_tlbr: np.ndarray | None = None
    _prev_obs_frame: int = -1
    # Estimated motion direction (unit vector, cx/cy), None until 2 observations exist
    velocity: np.ndarray | None = None

    _count = 0

    @staticmethod
    def next_id() -> int:
        Tracklet._count += 1
        return Tracklet._count

    @staticmethod
    def reset_id_counter() -> None:
        Tracklet._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def _record_observation(self, tlbr: np.ndarray, frame_id: int) -> None:
        """Store current detection as observation and refresh velocity."""
        self._prev_obs_tlbr = self._last_obs_tlbr
        self._prev_obs_frame = self._last_obs_frame
        self._last_obs_tlbr = tlbr.copy()
        self._last_obs_frame = frame_id

        if self._prev_obs_tlbr is not None and self._prev_obs_frame >= 0:
            dt = self._last_obs_frame - self._prev_obs_frame
            if dt > 0:
                prev_c = np.array(
                    [
                        (self._prev_obs_tlbr[0] + self._prev_obs_tlbr[2]) / 2,
                        (self._prev_obs_tlbr[1] + self._prev_obs_tlbr[3]) / 2,
                    ]
                )
                curr_c = np.array(
                    [
                        (self._last_obs_tlbr[0] + self._last_obs_tlbr[2]) / 2,
                        (self._last_obs_tlbr[1] + self._last_obs_tlbr[3]) / 2,
                    ]
                )
                vel = (curr_c - prev_c) / dt
                norm = np.linalg.norm(vel)
                self.velocity = vel / norm if norm > 1e-6 else None

    def activate(
        self,
        kalman_filter: KalmanFilter,
        frame_id: int,
        min_hits: int = 2,
    ) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = TrackState.New
        self.is_activated = True
        self.min_hits = min_hits
        self._record_observation(self.tlbr, frame_id)

    def predict(self) -> None:
        """Predict next state (prior) using the Kalman filter."""
        assert self.mean is not None, "Tracklet not initialized: mean is None"
        assert self.covariance is not None, "Tracklet not initialized: covariance is None"
        assert self.kalman_filter is not None, "Tracklet not initialized: kalman_filter is None"

        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def update(self, new_track: "Tracklet", frame_id: int) -> None:
        """Update tracklet state with a new measurement."""
        assert self.mean is not None, "Tracklet not initialized: mean is None"
        assert self.covariance is not None, "Tracklet not initialized: covariance is None"
        assert self.kalman_filter is not None, "Tracklet not initialized: kalman_filter is None"

        self.frame_id = frame_id
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        if (frame_id - self.start_frame) > self.min_hits:
            self.state = TrackState.Tracked
        else:
            self.state = TrackState.New
        self.is_activated = True
        self.det_score = new_track.det_score
        self._record_observation(new_track.tlbr, frame_id)

    def re_activate(
        self,
        new_track: "Tracklet",
        frame_id: int,
        new_id: bool = False,
    ) -> None:
        """Re-activate with Observation-Centric Re-Update (ORU).

        Instead of updating the drifted predicted state, we re-initialise
        the Kalman filter from the last actual observation, predict forward
        to the current frame, then apply the new measurement. This corrects
        velocity estimates that drifted during the lost gap.
        """
        assert self.kalman_filter is not None, "Tracklet not initialized: kalman_filter is None"

        if self._last_obs_tlbr is not None and self._last_obs_frame >= 0:
            # Re-init from last real observation
            self.mean, self.covariance = self.kalman_filter.initiate(
                self.tlwh_to_xyah(self.tlbr_to_tlwh(self._last_obs_tlbr))
            )
            # Predict forward through the gap (excluding the current frame)
            gap = frame_id - self._last_obs_frame - 1
            for _ in range(max(0, gap)):
                assert self.mean is not None and self.covariance is not None
                mean_state = self.mean.copy()
                self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        assert self.mean is not None and self.covariance is not None
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.det_score = new_track.det_score
        self._record_observation(new_track.tlbr, frame_id)

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


@dataclass
class TrackingParams:
    """Tracking parameters."""

    score_threshold: float = 0.3
    min_hits: int = 2
    max_time_lost: int = 10
    # OC-SORT: weight applied to velocity-direction cost for lost track matching
    inertia: float = 0.2


class PersonTracker:
    """OC-SORT multi-object tracker.

    Reference:
    Jiangmiao Pang et al.: OC-SORT: Observation-Centric SORT on Video
    Highlights Detection and Beyond. CVPR 2023.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self.tracked_tracklets: list[Tracklet] = []
        self.lost_tracklets: list[Tracklet] = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.is_initialized = False

    def init_tracker(self) -> None:
        self._logger.debug("Initializing person tracker...")
        self.tracked_tracklets = []
        self.lost_tracklets = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        Tracklet.reset_id_counter()
        self.is_initialized = True

    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        cost_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.array(list(range(cost_matrix.shape[0])), dtype=int),
                np.array(list(range(cost_matrix.shape[1])), dtype=int),
            )
        if not LAP_AVAILABLE:
            return self._greedy_assignment(cost_matrix, cost_threshold)

        assert lap is not None, "LAP should be available"
        matches = []
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=cost_threshold)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        return (
            np.asarray(matches) if matches else np.empty((0, 2), dtype=int),
            unmatched_a,
            unmatched_b,
        )

    def _greedy_assignment(
        self,
        cost_matrix: np.ndarray,
        cost_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matches = []
        matched_a, matched_b = set(), set()
        indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        for i, j in zip(indices[0], indices[1], strict=True):
            if cost_matrix[i, j] > cost_threshold:
                break
            if i not in matched_a and j not in matched_b:
                matches.append([i, j])
                matched_a.add(i)
                matched_b.add(j)
        unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_a])
        unmatched_b = np.array([j for j in range(cost_matrix.shape[1]) if j not in matched_b])
        return (
            np.asarray(matches) if matches else np.empty((0, 2), dtype=int),
            unmatched_a,
            unmatched_b,
        )

    def _iou_cost(
        self,
        tracks: list[Tracklet],
        detections: list[Tracklet],
    ) -> np.ndarray:
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes = np.array([d.tlbr for d in detections])
        if CYTHON_BBOX_AVAILABLE:
            assert bbox_ious is not None, "Cython bbox should be available"
            ious = bbox_ious(
                np.ascontiguousarray(track_boxes, dtype=np.float64),
                np.ascontiguousarray(det_boxes, dtype=np.float64),
            )
        else:
            ious = _numpy_bbox_ious(track_boxes, det_boxes)
        return 1 - ious

    def _direction_cost(
        self,
        tracks: list[Tracklet],
        detections: list[Tracklet],
        inertia: float,
    ) -> np.ndarray:
        """Velocity-direction consistency penalty for lost track matching.

        For each (track, detection) pair where the track has an estimated
        velocity, penalise assignments where the direction from the last
        observation to the detection candidate opposes the track velocity.
        Cost is in [0, inertia]; 0 = perfectly aligned, inertia = opposite.
        """
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        if inertia == 0:
            return cost
        for i, t in enumerate(tracks):
            if t.velocity is None or t._last_obs_tlbr is None:
                continue
            last_c = np.array(
                [
                    (t._last_obs_tlbr[0] + t._last_obs_tlbr[2]) / 2,
                    (t._last_obs_tlbr[1] + t._last_obs_tlbr[3]) / 2,
                ]
            )
            for j, d in enumerate(detections):
                det_box = d.tlbr
                det_c = np.array(
                    [
                        (det_box[0] + det_box[2]) / 2,
                        (det_box[1] + det_box[3]) / 2,
                    ]
                )
                delta = det_c - last_c
                norm = np.linalg.norm(delta)
                if norm < 1e-6:
                    continue
                cos_sim = np.dot(t.velocity, delta / norm)
                # (1 - cos_sim) / 2 maps [-1,1] -> [1,0], i.e. high when opposing
                cost[i, j] = inertia * (1.0 - cos_sim) / 2.0
        return cost

    def _make_detections(
        self, det_bboxes: list[list[float]], det_scores: list[float]
    ) -> list[Tracklet]:
        """Wrap raw detections as New tracklets."""
        return [
            Tracklet(
                _tlwh=Tracklet.tlbr_to_tlwh(np.array(bbox)),
                det_score=score,
            )
            for bbox, score in zip(det_bboxes, det_scores, strict=True)
        ]

    def _match_active_tracks(
        self,
        tracked: list[Tracklet],
        detections: list[Tracklet],
        threshold: float,
    ) -> tuple[list[Tracklet], list[Tracklet], list[int]]:
        """Stage 1: match currently-tracked tracks to detections.

        Returns ``(activated, newly_lost, unmatched_detection_indices)``.
        """
        cost_matrix = self._iou_cost(tracked, detections)
        matches, u_track, u_det = self._linear_assignment(cost_matrix, 1 - threshold)

        activated: list[Tracklet] = []
        for itracked, idet in matches:
            track = tracked[itracked]
            track.update(detections[idet], self.frame_id)
            activated.append(track)

        newly_lost: list[Tracklet] = []
        for it in u_track:
            track = tracked[it]
            track.mark_lost()
            newly_lost.append(track)
        return activated, newly_lost, [int(i) for i in u_det]

    def _match_lost_tracks(
        self,
        remaining_dets: list[Tracklet],
        params: TrackingParams,
    ) -> tuple[list[Tracklet], list[Tracklet]]:
        """Stage 2: match lost tracks (velocity-direction cost) to remaining detections.

        Returns ``(refound, still_unmatched_detections)``.
        """
        # OC-SORT adds velocity-direction cost for lost tracks.
        iou_cost = self._iou_cost(self.lost_tracklets, remaining_dets)
        dir_cost = self._direction_cost(self.lost_tracklets, remaining_dets, params.inertia)
        matches, _u_lost, u_det = self._linear_assignment(
            iou_cost + dir_cost, 1 - params.score_threshold
        )
        refound: list[Tracklet] = []
        for ilost, idet in matches:
            track = self.lost_tracklets[ilost]
            # OC-SORT ORU: corrects Kalman drift during gap
            track.re_activate(remaining_dets[idet], self.frame_id)
            refound.append(track)
        return refound, [remaining_dets[i] for i in u_det]

    def _match_unconfirmed(
        self,
        unconfirmed: list[Tracklet],
        remaining_dets: list[Tracklet],
        params: TrackingParams,
    ) -> tuple[list[Tracklet], list[Tracklet]]:
        """Stage 3: match unconfirmed tracks and activate genuinely new ones.

        Returns ``(activated, brand_new_tracklets)``.
        """
        cost = self._iou_cost(unconfirmed, remaining_dets)
        matches, u_unconf, u_det = self._linear_assignment(cost, 1 - params.score_threshold)
        activated: list[Tracklet] = []
        for itracked, idet in matches:
            unconfirmed[itracked].update(remaining_dets[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconf:
            unconfirmed[it].mark_removed()

        brand_new: list[Tracklet] = []
        for inew in u_det:
            track = remaining_dets[inew]
            track.activate(self.kalman_filter, self.frame_id, params.min_hits)
            brand_new.append(track)
        return activated, brand_new

    def _prune_lost_tracks(self, params: TrackingParams) -> list[Tracklet]:
        """Mark and return lost tracks that exceeded max_time_lost."""
        removed: list[Tracklet] = []
        for track in self.lost_tracklets:
            if self.frame_id - track.frame_id > params.max_time_lost:
                track.mark_removed()
                removed.append(track)
        return removed

    def _reconcile_track_lists(
        self,
        activated: list[Tracklet],
        refound: list[Tracklet],
        newly_lost: list[Tracklet],
        removed: list[Tracklet],
    ) -> None:
        """Rebuild the tracked/lost/removed lists after matching."""
        self.tracked_tracklets = [
            t for t in self.tracked_tracklets if t.state == TrackState.Tracked and t.is_activated
        ]
        self.tracked_tracklets = self._merge_lists(self.tracked_tracklets, activated)
        self.tracked_tracklets = self._merge_lists(self.tracked_tracklets, refound)
        self.lost_tracklets = self._subtract_lists(self.lost_tracklets, refound)
        self.lost_tracklets = self._subtract_lists(self.lost_tracklets, removed)
        self.lost_tracklets.extend(newly_lost)

    def update(
        self,
        det_bboxes: list[list[float]],
        det_scores: list[float],
        params: TrackingParams | None = None,
    ) -> list[Tracklet]:
        """Update tracker with new detections.

        Args:
            det_bboxes: Detection boxes in [x1, y1, x2, y2] format.
            det_scores: Detection confidence scores.
            params: Tracking parameters.

        Returns:
            List of active tracklets.
        """
        if not self.is_initialized:
            self.init_tracker()
        if params is None:
            params = TrackingParams()

        self.frame_id += 1
        detections = self._make_detections(det_bboxes, det_scores)

        unconfirmed = [t for t in self.tracked_tracklets if not t.is_activated]
        tracked = [t for t in self.tracked_tracklets if t.is_activated]
        for t in tracked + self.lost_tracklets:
            t.predict()

        activated, newly_lost, u_det = self._match_active_tracks(
            tracked, detections, params.score_threshold
        )
        refound, remaining_dets = self._match_lost_tracks([detections[i] for i in u_det], params)
        unconf_activated, brand_new = self._match_unconfirmed(unconfirmed, remaining_dets, params)
        activated.extend(unconf_activated)
        removed = self._prune_lost_tracks(params)

        self._reconcile_track_lists(activated + brand_new, refound, newly_lost, removed)

        return [
            t
            for t in self.tracked_tracklets
            if t.is_activated
            and t.state == TrackState.Tracked
            and (self.frame_id - t.frame_id) < params.max_time_lost
        ]

    @staticmethod
    def _merge_lists(
        list_a: list[Tracklet],
        list_b: list[Tracklet],
    ) -> list[Tracklet]:
        exists = {t.track_id for t in list_a}
        result = list(list_a)
        for t in list_b:
            if t.track_id not in exists:
                exists.add(t.track_id)
                result.append(t)
        return result

    @staticmethod
    def _subtract_lists(
        list_a: list[Tracklet],
        list_b: list[Tracklet],
    ) -> list[Tracklet]:
        ids_to_remove = {t.track_id for t in list_b}
        return [t for t in list_a if t.track_id not in ids_to_remove]
