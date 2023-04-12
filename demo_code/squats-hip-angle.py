# squats-hip-angle.py

# COMPUTE AND DISPLAY RIGHT HIP ANGLE FROM RIGHT SHOULDER TO RIGHT KNEE

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

training_dir = './training_set'
files = ['1-sit-to-stand-a_h.jpg','2-sit-to-stand-b_h.jpg','3-sit-to-stand-c_h.jpg','7-kb-goblet-squat-a_h.jpg','8-kb-goblet-squat-b_h.jpg']
filelist = [training_dir+'/'+filename for filename in files]

for file in filelist: 
	with mp_pose.Pose(static_image_mode=True, model_complexity=2,enable_segmentation=True,min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		# Read the file in and get dims
		image = cv2.imread(file)
		dims = image.shape

		# Convert the BGR image to RGB and then process with the `Pose` object.
		results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		# Extract landmarks
		try:
			landmarks = results.pose_landmarks.landmark

			# Get coordinates
			shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
			hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
			knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

			# Calculate angle
			angle = float("{:.2f}".format(calculate_angle(shoulder, hip, knee)))
			print("file: "+str(file)+", hip angle: "+str(angle))

			# Visualize angle
			cv2.putText(image, str(angle),tuple(np.multiply(hip, [dims[1]+20, dims[0]-50]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
		except:
			pass



		# segment person out of image
		red_img = np.zeros_like(image,dtype=np.uint8)
		red_img[:,:] = (255,255,255)
		segm_2class = 0.2 + 0.8*results.segmentation_mask
		segm_2class = np.repeat(segm_2class[...,np.newaxis],3,axis=2)
		image = image * segm_2class + red_img * (1 - segm_2class)

		# Render detections
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=10, circle_radius=4),mp_drawing.DrawingSpec(color=(245,66,230), thickness=10, circle_radius=4))

		# # Save image with drawing
		filename = "./output/"+file.split('/')[-1]
		print(filename)
		cv2.imwrite(filename, image)
