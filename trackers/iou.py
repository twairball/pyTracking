import uuid
from .box_utils import calc_iou
from .base import BaseTracker

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

class IOUTracker(BaseTracker):
    """IOU based tracker
    High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora
    http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf

    track is a dict with keys:
        boxes: list of box [x1, y1, x2, y2]
        max_score: float

    """
    
    def __init__(self, sigma_l=0.0, sigma_h=0.5, sigma_iou=0.3, t_min=3):
        """Initialize IOU tracker. 
        Args:
            sigma_l: Threshold [0 - 1.] for detections. 
            sigma_h: Threshold [0 - 1.] for finished tracks. 
            sigma_iou: Threshold for track matching.
            t_min: minimum frames for finishing tracks. 
        Returns:
            tracks_active: List of active tracks. 
        """
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.tracks_active = []
        self.tracks_finished = []
    
    def track(self, detections):
        updated_tracks = []

        # filter detections below threshold
        dets = [det for det in detections if det['score'] >= self.sigma_l]

        for track in self.tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: calc_iou(track['box'], x['box']))
                if calc_iou(track['box'], best_match['box']) >= self.sigma_iou:
                    track = update_track(track, best_match)
                    updated_tracks.append(track)
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= self.sigma_h and track['frames'] >= self.t_min:
                    track['active'] = 0 # deactivate
                    self.tracks_finished.append(track)

        # create new tracks
        new_tracks = [create_track(det) for det in dets]
        
        self.tracks_active = updated_tracks + new_tracks
        return self.tracks_active
