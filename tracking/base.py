import uuid


def create_track(detect_result):
    return {
        'box': detect_result['box'],
        'max_score': detect_result['score'],
        'score': detect_result['score'],
        'class_id': detect_result['class_id'],
        'track_id': uuid.uuid4().hex[:6],
        'frames': 1,
        'active': 1
    }


def update_track(track, detect_result):
    return {
        'box': detect_result['box'],
        'max_score': max(track['max_score'], detect_result['score']),
        'score': detect_result['score'],
        'class_id': detect_result['class_id'],
        'track_id': track['track_id'],
        'frames': track['frames'] + 1,
        'active': 1
    }


class BaseTracker():

    def __init__(self):
        self.tracks_active = []
        self.tracks_finished = []
        self.frame_counter = 0

    def track(self, detections):
        return NotImplementedError

    def get_tracks(self):
        """Return list of valid tracks, including finished and active tracks"""
        return NotImplementedError

    def create_track(self, det):
        track = create_track(det)
        track['start_frame'] = self.frame_counter
        return track
