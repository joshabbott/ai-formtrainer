# utils.py
"""
Main ProcessPlank Class and some useful helper functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import spatial

# main clas for processing a plank through mediapipe and analyzing results
class ProcessPlank:
    # init object with a mp_pose instance, a pretrained plank model, and the input video filename
    def __init__(self, mp_pose, model, input_filename):
        
        self.mp_pose = mp_pose
        self.model = model
        self.input_filename = input_filename
        self.data = None
        self.angles_data = None
        self.feat_df = None
        self.y_hat = None
        
        # globals
        # create dictionary of the landmarks we care about in planks (for each side)
        self.left_features = {
            'shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,  # 11
            'elbow'   : self.mp_pose.PoseLandmark.LEFT_ELBOW.value,     # 13
            'wrist'   : self.mp_pose.PoseLandmark.LEFT_WRIST.value,     # 15                    
            'hip'     : self.mp_pose.PoseLandmark.LEFT_HIP.value,       # 23
            'knee'    : self.mp_pose.PoseLandmark.LEFT_KNEE.value,      # 25
            'ankle'   : self.mp_pose.PoseLandmark.LEFT_ANKLE.value,     # 27
            'foot'    : self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value # 31
        }

        self.right_features = {
            'shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,  # 12
            'elbow'   : self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,     # 14
            'wrist'   : self.mp_pose.PoseLandmark.RIGHT_WRIST.value,     # 16,
            'hip'     : self.mp_pose.PoseLandmark.RIGHT_HIP.value,       # 24,
            'knee'    : self.mp_pose.PoseLandmark.RIGHT_KNEE.value,      # 26,
            'ankle'   : self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,     # 28,
            'foot'    : self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value # 32
        }

        self.dict_features = {}
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        # font type
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in RGB
        self.COLORS = {
            'blue'       : (255, 127, 0), 
            'yellow'     : (0, 255, 255), 
            'white'      : (255,255,255),
            'light_blue' : (255, 204, 102)
        }

        
    # mediapipe processing code
    # process a frame from a video 
    def process_frame(self,frame,pose,frame_num,videowriter=None,saveframe=False,output_filename="output.jpg"):
        frame_height, frame_width, _ = frame.shape

        # Process the image.
        # recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # process frame through mediapipe's main process function and get the keypoints
        keypoints = pose.process(frame)

        # recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        crit_angles_list = [0,0,0,0]

        # if keypoints were found, extract landmark coordinates and compute critical angles
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)


            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)   

            camera_view = 'right'
            # determine camera view (left or right) - front will default to right for now
            if offset_angle > 35.0:
                # hard-coded for now - determined from squat code author, seems good
                camera_view = 'front'
                # print("camera view from front")

            # compute distance for both sides shoulder to foot    
            dist_l_sh_hip = abs(left_foot_coord[0] - left_shldr_coord[0])
            dist_r_sh_hip = abs(right_foot_coord[0] - right_shldr_coord)[0]        
            if dist_l_sh_hip > dist_r_sh_hip:
                camera_view = 'left'
                # print("camera view from left")


            # assign side specific coords
            if camera_view == 'left':
                shldr_coord = left_shldr_coord
                elbow_coord = left_elbow_coord
                wrist_coord = left_wrist_coord
                hip_coord = left_hip_coord
                knee_coord = left_knee_coord
                ankle_coord = left_ankle_coord
                foot_coord = left_foot_coord

                multiplier = -1

            elif camera_view == 'right':
                shldr_coord = right_shldr_coord
                elbow_coord = right_elbow_coord
                wrist_coord = right_wrist_coord
                hip_coord = right_hip_coord
                knee_coord = right_knee_coord
                ankle_coord = right_ankle_coord
                foot_coord = right_foot_coord

                multiplier = 1
            else:
                # just default to right
                shldr_coord = right_shldr_coord
                elbow_coord = right_elbow_coord
                wrist_coord = right_wrist_coord
                hip_coord = right_hip_coord
                knee_coord = right_knee_coord
                ankle_coord = right_ankle_coord
                foot_coord = right_foot_coord

                multiplier = 1


            # compute critical angles to a vertical line
            shoulder_vertical_angle = find_angle(hip_coord, np.array([shldr_coord[0], 0]), shldr_coord)
            cv2.ellipse(frame, shldr_coord, (30, 30), 
                        angle = 0, startAngle = -90, endAngle = -90-multiplier*shoulder_vertical_angle, 
                        color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

            draw_dotted_line(frame, shldr_coord, start=shldr_coord[1]-80, end=shldr_coord[1]+20, line_color=self.COLORS['blue'])
            cv2.putText(frame, str(shoulder_vertical_angle),shldr_coord+[-40,-40], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


            hip_vertical_angle = find_angle(knee_coord, np.array([hip_coord[0], 0]), hip_coord)
            cv2.ellipse(frame, hip_coord, (30, 30), 
                        angle = 0, startAngle = -90, endAngle = -90-multiplier*hip_vertical_angle, 
                        color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

            draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])
            cv2.putText(frame, str(hip_vertical_angle),hip_coord+[-40,-40], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


            knee_vertical_angle = find_angle(ankle_coord, np.array([knee_coord[0], 0]), knee_coord)
            cv2.ellipse(frame, knee_coord, (30, 30), 
                        angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                        color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

            draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])
            cv2.putText(frame, str(knee_vertical_angle),knee_coord+[-40,-40], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


            ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
            cv2.ellipse(frame, ankle_coord, (30, 30),
                        angle = 0, startAngle = -90, endAngle = -90+multiplier*ankle_vertical_angle,
                        color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

            draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])
            cv2.putText(frame, str(ankle_vertical_angle),ankle_coord+15, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)



            # draw a line between landmarks.
            cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
            cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
            cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
            cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
            cv2.line(frame, ankle_coord, knee_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
            cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)

            # plot landmark points
            cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
            cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
            cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
            cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
            cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
            cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
            cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

            # write information to video frame
            if videowriter:
                videowriter.write(frame)

            # save frame as image
            if saveframe:
                if output_filename:
                    cv2.imwrite(output_filename, frame)
                else:
                    cv2.imwrite("./average_plank.jpg", frame)

            # save frame angles for later analyses
            crit_angles_list = [shoulder_vertical_angle, hip_vertical_angle, knee_vertical_angle, ankle_vertical_angle]
        
        # return keypoint results and critical angles
        return keypoints, crit_angles_list

    
    # process a whole video and store results
    def process_video(self,output_filename="output.mp4"):
        with self.mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=1,smooth_landmarks=True) as pose:
            # Create VideoCapture object
            cap = cv2.VideoCapture(self.input_filename)

            # Raise error if file cannot be opened
            if cap.isOpened() == False:
                print("Error opening video stream or file")
                raise TypeError

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = (frame_width, frame_height)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # Get the number of frames in the video
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # create videowriter object to create new video with pose estimation overlayed
            videowriter = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'mp4v'),fps,size) # saves as .mp4
            # videowriter = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'MJPG'),fps,size) # saves as .avi


            # Create a NumPy array to store the pose data 
            # The shape is lengthx33x3 - 3D XYZ data for 33 landmarks across 'length' number of frames
            data = np.empty((length, 33, 3))
            angles_data = np.empty((length, 4)) 
            frame_num = 0

            while cap.isOpened():
                # read current frame
                ret, frame = cap.read()
                if not ret:
                    break

                # process current frame and get list of pose estimation results and list of critical angles
                results, angles_list = self.process_frame(frame,pose,frame_num,videowriter)

                # save landmark coordinates for this frame
                landmarks = results.pose_world_landmarks.landmark
                for i in range(len(self.mp_pose.PoseLandmark)):
                    data[frame_num, i, :] = (landmarks[i].x, landmarks[i].y, landmarks[i].z)

                # save the critical angles for this frame
                for i in range(len(angles_list)):
                    angles_data[frame_num,i] = angles_list[i]

                frame_num += 1

            cap.release()
            videowriter.release()
            cv2.destroyAllWindows()
            print("The video was successfully processed as: "+output_filename)

        # save pose estimation landmark coordinates and computed critical angles data
        self.data = data
        self.angles_data = angles_data
        # analyze plank data to find when in/out plank
        self.analyze_plank()

        
    # process a video but only process and save the mean frame
    def process_video_for_meanframe(self,mean_framenum,output_filename="average_plank.jpg"):
        with self.mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=1,smooth_landmarks=True) as pose:
            # Create VideoCapture object
            cap = cv2.VideoCapture(self.input_filename)

            # Raise error if file cannot be opened
            if cap.isOpened() == False:
                print("Error opening video stream or file")
                raise TypeError

            frame_num = 0

            while cap.isOpened():
                # read current frame
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num == mean_framenum:
                    # process mean frame and save
                    results, angles_list = self.process_frame(frame,pose,frame_num,saveframe=True,output_filename=output_filename)
                frame_num += 1

            cap.release()
            cv2.destroyAllWindows()
            print("The mean frame was successfully processed as: "+output_filename)

    # create a feature matrix df from saved plank data and use pretrained model to predict frames when in/out of plank
    def analyze_plank(self):
        # run through trained model
        feat_df = make_default_feature_matrix(self.data, self.angles_data)
        X = feat_df.drop(0,axis=1) # features
        # predict when in (1) or out (0) of a plank
        y_hat = self.model.predict(X)
        # save feature matrix df and model predictions
        self.feat_df = feat_df
        self.y_hat = y_hat
           
    # plot the "in plank" critical angles over time
    def display_critical_angles(self):
        plot_crit_angles(self.angles_data[self.y_hat == 1])

    # compute and process average plank
    def display_average_plank(self, output_filename="average_plank.jpg"):
        # compute mean vector over features of frames in plank
        mean_features = np.array(self.feat_df[self.y_hat == 1].mean())
        # search for nearest frame to mean_features with a KDTree
        feat_matrix = np.array(self.feat_df)
        tree = spatial.KDTree(feat_matrix)
        mean_framenum = tree.query(mean_features)[1]
        # process just this frame and save it
        self.process_video_for_meanframe(mean_framenum,output_filename)


# some useful helper functions

# get angle between 2 points
def find_angle(p1, p2, ref_pt = np.array([0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    # note degree will always be less than 180      
    degree = int(180 / np.pi) * theta

    return int(degree)

# get actual landmark coordinates given frame size
def get_landmark_array(pose_landmark, key, frame_width, frame_height):

    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])

# return coordinates for landmarks useful for planks (for one side only)
def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord   = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord   = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord   = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord   = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord   = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    
    else:
       raise ValueError("feature needs to be either 'nose', 'left' or 'right")

# draws a dotted vertical line
def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame

# places a 1 class in 0th column for every row
def make_default_feature_matrix(data, angles_data):
    features = []
    for i in range(len(data)):
        row = [1]+list(data[i].flatten())+list(angles_data[i]) # 0th column is init'd to 1
        features.append(row)

    features_df = pd.DataFrame(features)
    
    return features_df

# plot the angles data over time (in frames)
def plot_crit_angles(angles_data):
    col_names = ['Shoulder', 'Hip', 'Knee', 'Ankle']
    df = pd.DataFrame(data=angles_data,columns=col_names)
    df.plot(figsize=(10,4))
    plt.title('Plank angles over time')
    plt.xlabel('Time (frames)')
    plt.ylabel('Angle (degrees)')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()