#main file for Vehicle Detection using SSD
'''
    @author: Arvind C S
'''


import numpy as np
import matplotlib.pyplot as plt
from glob import glob
#from moviepy.editor import VideoFileClip
import cv2
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

from helper import *
from detector import *
from tracker import *

#global variables used

frame_count = 0 # frame counter

max_age = 4 # no of consecutive unmatched detection before a track is deleted

min_hits = 1 # no of consecutive matches needed to establish a track

tracker_list = [] # list for trackers

# list fro track ID

track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = True

def assign_detections_to_trackers(trackers,detections,iou_thrd=0.3):

    IOU_mat = np.zeros((len(trackers),len(detections)),dtype=np.float32)

    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = box_iou2(trk,det)

    #matches
    #solve the maximizing the sum of IOU assignment problem using hungarian algorithm

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers,unmatched_detections = [],[]

    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d,det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifly the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches,np.array(unmatched_detections),np.array(unmatched_trackers)





def detectionAlgo(image):

    #function for detection and tracking

    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global tracker_id_list
    global debug

    frame_count+=1

    img_dim = (image.shape[1],image.shape[0])
    z_box = det.get_localization(image) # measurement
    if debug:
        print('frame:',frame_count)

    x_box = []
    if debug:
        for i in range(len(z_box)):
            img1 = draw_box_label(image,z_box[i],box_color=(255,0,0))
            plt.imshow(img1)
        plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched,unmatched_dets,unmatched_trks = assign_detections_to_trackers(x_box,z_box,iou_thrd = 0.3)

    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)


    # Deal with matched detections

    if matched.size > 0:
        for trk_idx,det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z,axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0],xx[2],xx[4],xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits +=1

    # deal with unmatched detections

    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z,axis=0).T
            tmp_trk = Tracker()
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0],xx[2],xx[4],xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)


    # Deal with unmatched tracks

    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses +=1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0],xx[2],xx[4],xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # the list of tracks to be annotated

    good_tracker_list = []
    for trk in tracker_list:
        if((trk.hits >= min_hits)and(trk.no_losses<=max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box

            if debug:
                print('updated box: ',x_cv2)
                print()
            img = draw_box_label(image,x_cv2)

    # book keeping

    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    return image



if __name__ == "__main__":

    det = CarDetector()

    if debug:
        images = [plt.imread(file) for file in glob('./test_images/*.jpg')]

        for i in range(len(images)): #[0:7]:
            image = images[i]
            #image = cv2.resize(image,(1280,720))
            image_box = detectionAlgo(image)
            plt.imshow(image_box)
            plt.show()