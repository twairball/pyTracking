import numpy as np
import os
import configparser
import csv

def get_sequences(data_dir="./data/mot17/train"):
    sequences = os.listdir(data_dir)
    if '.DS_Store' in sequences:
        sequences.remove('.DS_Store')
    return sequences


def get_dets(seq, train_dir):
    det_filepath = "%s/det/det.txt" % seq
    path = os.path.join(train_dir, det_filepath)
    seq_dets = np.loadtxt(path, delimiter=',')  # load detections
    return seq_dets


def get_labels(seq, train_dir):
    det_filepath = "%s/gt/gt.txt" % seq
    path = os.path.join(train_dir, det_filepath)
    seq_gts = np.loadtxt(path, delimiter=',')  # load detections
    return seq_gts


def get_image_shape(seq, train_dir):
    config = configparser.ConfigParser()
    path = os.path.join(train_dir, '%s/seqinfo.ini' % seq)
    config.read(path)
    image_shape = config['Sequence']['imHeight'], config['Sequence']['imWidth']
    return image_shape


def mot_to_dict(mot_detection):
    """Return detection result as dict.
    Args:
        mot_detection: [x, y, w, h, score] in np.array
    Returns:
        detection object
            box: [x1, y1, x2, y2] 
            score: float
    """
    box = mot_detection[0:4]
    box[2:4] += box[0:2]
    return dict(box=box, score=mot_detection[4], class_id=-1)


def tracks_to_mot(tracks):
    """
    Format tracks to MOT16 format
    """
    pass


def save_to_csv(out_path, tracks):
    """Save tracks to CSV file. 
    Args:
        out_path: str, path to output csv file. 
        tracks: list of tracks to store.  
    """
    field_names = ['frame', 'id', 'x', 'y', 'w',
                   'h', 'score', 'class', 'visibility']

    with open(out_path, 'w') as f:
        writer = csv.DictWriter(f, field_names)
        _id = 1
        for track in tracks:
            for i, box in enumerate(track['box']):
                # TODO:
                row = {'id': _id,
                       'frame': track['start_frame'] + i,
                       'x': box[0],
                       'y': box[1],
                       'w': box[2] - box[0],
                       'h': box[3] - box[1],
                       'score': track['max_score'],
                       'wx': -1,
                       'wy': -1,
                       'wz': -1}
