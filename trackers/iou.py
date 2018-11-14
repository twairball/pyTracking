from .box_utils import calc_iou
from .base import BaseTracker, update_track


class IOUTracker(BaseTracker):
    """IOU based tracker
    High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora
    http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf

    track is a dict with keys:
        boxes:     list of box [x1, y1, x2, y2]
        max_score: float, max confidence tracker has seen. 
        score:     float, detector confidence.
        class_id:  int, unique id for object detection class. 
        track_id:  str, UUID for tracker.
        frames:    int, number of frames tracker active. 
        active:    int, denotes if tracker is still active. 
    """

    def __init__(self, sigma_l=0.0, sigma_h=0.5, sigma_iou=0.3, t_min=3):
        """Initialize IOU tracker. 
        Args:
            sigma_l: Threshold [-1, 1.] for detections. 
            sigma_h: Threshold [-1, 1.] for finished tracks. 
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
        self.frame_counter = 0

    def track(self, detections):
        """TODO: 
        """
        updated_tracks = []
        new_tracks = []

        # filter detections below threshold
        dets = [det for det in detections if det['score'] >= self.sigma_l]

        for track in self.tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                ind, best_match = max(
                    enumerate(dets), key=lambda x: calc_iou(track['box'], x[1]['box']))

                if calc_iou(track['box'], best_match['box']) >= self.sigma_iou:
                    print('track: ', track['box'], ', best_match:', best_match['box'], 'iou: ', calc_iou(
                        track['box'], best_match['box']))

                    # TODO:
                    # track['bboxes'].append(best_match['bbox'])
                    track = update_track(track, best_match)
                    updated_tracks.append(track)
                    del dets[ind]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= self.sigma_h and track['frames'] >= self.t_min:
                    track['active'] = 0  # deactivate
                    self.tracks_finished.append(track)
                    print("finished track: ", track)

        if len(dets) > 0:
            print("create new tracks: ", len(dets))
            # create new tracks
            new_tracks = [self.create_track(det) for det in dets]

        self.tracks_active = updated_tracks + new_tracks

        # update counter
        self.frame_counter += 1
        return self.tracks_active

    def get_tracks(self):
        tracks = self.tracks_finished
        # finish remaining active tracks
        tracks += [track for track in self.tracks_active
                   if track['max_score'] >= self.sigma_h and track['frames'] >= self.t_min]
        return tracks
