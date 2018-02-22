Vehicle Detection and Tracking

Arvind C S 


Detection and tracking of multiple vehicles using a mounted camera inside a self-driving car.Detector localizes the vehicles in each frame.
The tracker is then updated with the detection results. We use a lightweight model: ssd_mobilenet_v1_coco that is based on Single Shot Multibox Detection (SSD) framework with minimal modification

main.py - Is the main file which calls detector function and tracking function

Detector.py - ssd_mobilenet coco pre trained weights are used for detector of vehicle 
	      first 14 classes all related to transportation, including bicycle, car, and bus, etc. The ID for car is 3.
	      category_index =
		{1: {'id': 1, 'name': u'person'},
		 2: {'id': 2, 'name': u'bicycle'},
 		 3: {'id': 3, 'name': u'car'}, 
 		 4: {'id': 4, 'name': u'motorcycle'},
 		 5: {'id': 5, 'name': u'airplane'}, 
 		 6: {'id': 6, 'name': u'bus'}, 
 		 7: {'id': 7, 'name': u'train'},
 		 8: {'id': 8, 'name': u'truck'}, 
 		 9: {'id': 9, 'name': u'boat'}, 
 		10: {'id': 10, 'name': u'traffic light'}, 
 		11: {'id': 11, 'name': u'fire hydrant'}, 
 		13: {'id': 13, 'name': u'stop sign'},
  		14: {'id': 14, 'name': u'parking meter'}} 

Tracker.py - kalaman filter is used for tracking the detected object
	     Kalman filter for tracking objects. Kalman filter has the following important features that tracking can benefit from:
		Prediction of object's future location
		Correction of the prediction based on new measurements
		Reduction of noise introduced by inaccurate detections
		Facilitating the process of association of multiple objects to their tracks

		Kalman filter consists of two steps: prediction and update. The first step uses previous states to predict the current state.
		The second step uses the current measurement, such as detection bounding box location , to correct the state.

Multiple Vehicle Detection and Tracking

If there are multiple detections, we need to match (assign) each of them to a tracker. 
We use intersection over union (IOU) of a tracker bounding box and detection bounding box as a metric.
We solve the maximizing the sum of IOU assignment problem using the Hungarian algorithm.



		







