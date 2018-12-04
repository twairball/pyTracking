from .box_utils import calc_iou
from .base import Tracker, Track

import logging
logger = logging.getLogger()

class MaxScoreTrack(Track):
    """Single box tracker, maintains max score"""

    def __init__(self, det):
        """
        Args:
            det: Detection object
        """
        super(MaxScoreTrack, self).__init__(det)
        self.max_score = self.score

    def update(self, det):
        super(MaxScoreTrack, self).update(det)
        self.max_score = max(self.score, det.score)


class IOUTracker(Tracker):
    """IOU based tracker
    High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora
    http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf. 

    Code originally from https://github.com/bochinski/iou-tracker, modified by Jerry Liu. 
    """

    def __init__(self, sigma_l=0.0, sigma_h=0.5, sigma_iou=0.3, t_min=3):
        """Initialize IOU tracker. 
        Args:
            sigma_l: Threshold [-1, 1.] for detections. 
            sigma_h: Threshold [-1, 1.] for finished tracks. 
            sigma_iou: Threshold for track matching.
            t_min: minimum frames for marking active tracks
        """
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h # TODO: deprecated
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.tracks = []

    def track(self, detections):
        """Update tracks for detections. 
        Args:
            detections: list of Detection objects
        Returns:
            list of all tracks
        """
        next_tracks = []

        # filter detections below threshold
        dets = [det for det in detections if det.score >= self.sigma_l]

        # get new or active tracks
        tracks = self.get_tracks()

        for track in tracks:
            if len(dets) > 0:
                # get det with highest iou
                ind, best_match = max(
                    enumerate(dets), key=lambda x: calc_iou(track.box, x[1].box))

                if calc_iou(track.box, best_match.box) >= self.sigma_iou:
                    logger.info("track: %s, best_match: %s, iou: %s" % (track.box, best_match.box, calc_iou(track.box, best_match.box)))
                    track.update(best_match)
                    if track.frames >= self.t_min:
                        track.active()
                    del dets[ind]

            # mark finished if track was not updated
            if track not in next_tracks:
                track.finish()
                logger.info("finished track: ", track)

            # add to list
            next_tracks.append(track)

        if len(dets) > 0:
            logger.info("create new tracks: ", len(dets))
            # create new tracks
            next_tracks += [MaxScoreTrack(det) for det in dets]
            
        self.tracks = next_tracks
        return self.tracks

