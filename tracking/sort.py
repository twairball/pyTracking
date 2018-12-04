from filterpy.kalman import KalmanFilter
import numpy as np


from .linear_assignment_ import linear_assignment

from .box_utils import calc_iou
from .base import Tracker, Track


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


# TODO:
# this is a single track
class KalmanTrack(Track):
    """
    This class represents the internel state of individual tracked objects observed as bbox.

    Code originally from https://github.com/abewley/sort, modified by Jerry Liu 2018. 

    """
    # count = 0
    def __init__(self,det):
        """
        Initialises a tracker using initial bounding box.
        """
        super(KalmanTrack, self).__init__(det)
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(det.get_box_xyxy())
        self.time_since_update = 0

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,det):
        """
        Updates the state vector with observed box.
        """
        super(KalmanTrack, self).update(det)
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(self.box))
        # self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_tracks(detections, tracks, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Args:
        detections: list of Detection objects
        tracks: list of Track objects
        iou_threshold: minimum iou score threshold for matching criteria. 
    Returns:    
        Returns 3 lists of matches, unmatched_detections and unmatched_tracks. 

    Code from https://github.com/abewley/sort

    """
    if(len(tracks)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(tracks)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(tracks):
            iou_matrix[d,t] = calc_iou(det[0:4],trk[0:4])
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_tracks = []
    for t,trk in enumerate(tracks):
        if(t not in matched_indices[:,1]):
            unmatched_tracks.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)



class SortTracker(Tracker):
    """SORT based tracker. 
    Simple online and real-time tracking by Bewley, et al. 
    http://arxiv.org/abs/1602.00763
    
    Code originally from https://github.com/abewley/sort, modified by Jerry Liu
    
    """

    def __init__(self,max_age=1,min_hits=3):
        """Initialize SORT tracker. 
        Args:
            max_age: int, number of frames before finishing tracks
            min_hits: int, number of frames before activating tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0

    def track(self, detections):
        """Update tracks for detections. 
        Args:
            detections: list of Detection objects
        Returns:
            list of all tracks
        """
        self.frame_count += 1
        tracks = self.get_tracks()

        predicted_pos = np.asarray([t.predict()[0] for t in tracks])
        predicted_pos = np.reshape(predicted_pos, [len(tracks), 4])
        predicted_pos = np.ma.compress_rows(np.ma.masked_invalid(predicted_pos))
        det_pos = np.asarray([d.get_box_xyxy() for d in detections])

        # remove tracks that have nan predictions
        invalid_pos = np.any(np.isnan(predicted_pos), axis=1)
        if len(invalid_pos) > 0:
            tracks = tracks[~invalid_pos]

        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            det_pos, predicted_pos)

        # update matched trackers with assigned detections
        for t, track in enumerate(tracks):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0][0]
                track.update(detections[d])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            track = KalmanTrack(detections[i])
            tracks = np.append(tracks, [track])
        
        for t in tracks:
            d = track.get_state()[0]
            if((t.time_since_update < 1) and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                t.active()
            if(t.time_since_update > self.max_age):
                t.finish()

        self.tracks = tracks
        return self.tracks
