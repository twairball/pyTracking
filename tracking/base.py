import uuid
import numpy as np


class Detection():
    """Detection Result"""

    def __init__(self, box, score):
        """
        Args:
            box: bounding box as [x, y, w, h]
            score: detection confidence in [0, 1]
        """
        self.box = np.asarray(box, dtype=np.float)
        self.score = score

    def to_array(self):
        return np.append(self.box, self.score)

    def get_box_xyxy(self):
        box = self.box.copy()
        box[2:] += box[:2]
        return box

    def to_array_xyxy(self):
        return np.append(self.get_box_xyxy(), self.score)


class Track():
    """ Single target tracking """
    NEW = 0
    ACTIVE = 1
    FINISHED = 2

    def __init__(self, det, track_id=None):
        """
        Returns a single target tracker. 

        Args:
            det: Detection result object
            track_id: unique track id. Assigns random 6 digit hex string if None. Default=None. 
        """
        self.box = det.get_box_xyxy()
        self.score = det.score
        self.status = Track.NEW

        self.track_id = track_id if track_id is not None else uuid.uuid4().hex[:6]

        self.frames = 1
        self.status = Track.NEW
    
    def __str__(self):
        """print str"""
        return "%s" % self.to_dict()

    def to_dict(self):
        """Returns dict representation"""
        return {
            'track_id': self.track_id,
            'box': self.box,
            'score': self.score,
            'status': self.status,
            'frames': self.frames
        }

    def update(self, det):
        """Update tracker. 
        Args:
            det: Detection result object
        """
        self.box = det.get_box_xyxy()
        self.score = det.score
        self.frames += 1

    def active(self):
        """Set status to active"""
        self.status = Track.ACTIVE

    def finish(self):
        """Set status to finished; can be marked for deletion"""
        self.status = Track.FINISHED

    def is_finished(self):
        return self.status == Track.FINISHED

class Tracker():
    """ Multi-target tracker """

    def __init__(self):
        self.tracks = []

    def track(self, detections):
        return NotImplementedError

    def get_tracks(self):
        """Returns new or active tacks"""
        tracks = [track for track in self.tracks if not track.is_finished()]
        tracks = np.asarray(tracks)
        return tracks
