import cv2
import numpy as np
import ast


def load_txt_dict(dictionary):
    with open(dictionary, 'r') as file:
        # Read the file
        dict_str = file.read()

        # Convert the string to a dictionary
        return ast.literal_eval(dict_str)


def load_video_with_ROI(video_path, ROI_path):
    cap = cv2.VideoCapture(video_path)
    my_dict = load_txt_dict(ROI_path)
    posy = my_dict["posy"]
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # frame[posy,:,0] = 255
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_CUBIC)
            # frame_resized = frame
            original_width = frame.shape[0]
            frames.append(frame_resized)
        else:
            break
    new_posy = 300 * posy / original_width
    # Convert the list of frames into a numpy array
    return np.array(frames), new_posy


def load_video_with_ROI_with_separate_label(video_path, ROI_path, label_path):
    vid_roi, posy = load_video_with_ROI(video_path, ROI_path)
    label_dict = load_txt_dict(label_path)
    keys = list(label_dict.keys())

    label_array = []

    for i in range(vid_roi.shape[0]):
        if f"{i}" in keys:
            label_array = np.append(label_array, label_dict[f"{i}"])
        else:
            label_array = np.append(label_array, 0)

    return label_array, vid_roi, posy


def concat_vid_rois_and_labels(vid_paths, roi_paths, lab_paths):
    labels, vid_rois, _ = load_video_with_ROI_with_separate_label(vid_paths[0], roi_paths[0], lab_paths[0])

    for i in range(1, vid_paths.shape[0]):
        labels_temp, vid_rois_temp, _ = load_video_with_ROI_with_separate_label(vid_paths[i], roi_paths[i],
                                                                                lab_paths[i])

        labels = np.concatenate((labels, labels_temp))
        vid_rois = np.concatenate((vid_rois, vid_rois_temp))

    return labels, vid_rois


class KalmanFilter(object):
    """docstring for KalmanFilter"""

    def __init__(self, dt=1, stateVariance=1, measurementVariance=1,
                 method="Accerelation"):
        super(KalmanFilter, self).__init__()
        self.method = method
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt
        self.initModel()

    """init function to initialise the model"""

    def initModel(self):
        if self.method == "Accerelation":
            self.U = 1
        else:
            self.U = 0
        self.A = np.matrix([[1, self.dt, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, self.dt], [0, 0, 0, 1]])

        self.B = np.matrix([[self.dt ** 2 / 2], [self.dt], [self.dt ** 2 / 2],
                            [self.dt]])

        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.P = np.matrix(self.stateVariance * np.identity(self.A.shape[0]))
        self.R = np.matrix(self.measurementVariance * np.identity(
            self.H.shape[0]))

        self.Q = np.matrix([[self.dt ** 4 / 4, self.dt ** 3 / 2, 0, 0],
                            [self.dt ** 3 / 2, self.dt ** 2, 0, 0],
                            [0, 0, self.dt ** 4 / 4, self.dt ** 3 / 2],
                            [0, 0, self.dt ** 3 / 2, self.dt ** 2]])

        self.erroCov = self.P
        self.state = np.matrix([[200], [1], [200], [1]])

    """Predict function which predicst next state based on previous state"""

    def predict(self):
        self.predictedState = self.A * self.state + self.B * self.U
        self.predictedErrorCov = self.A * self.erroCov * self.A.T + self.Q
        temp = np.asarray(self.predictedState)
        return temp[0], temp[2]

    """Correct function which correct the states based on measurements"""

    def correct(self, currentMeasurement):
        self.kalmanGain = self.predictedErrorCov * self.H.T * np.linalg.pinv(
            self.H * self.predictedErrorCov * self.H.T + self.R)
        self.state = self.predictedState + self.kalmanGain * (currentMeasurement
                                                              - (self.H * self.predictedState))

        self.erroCov = (np.identity(self.P.shape[0]) -
                        self.kalmanGain * self.H) * self.predictedErrorCov


class CandidateOrFish:
    def __init__(self, ID, initial_position, diameter):
        self.ID = ID
        self.initial_position = initial_position
        self.position = initial_position
        self.appearance_number = 1
        self.diameter = diameter
        self.counted = 0
        self.consecutive_appearance = 1

        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.01
        self.kalman.statePre = np.array([[initial_position[0]], [initial_position[1]], [0], [15]], np.float32)
        self.kalman.statePost = np.array([[initial_position[0]], [initial_position[1]], [0], [15]], np.float32)


    def update_and_predict(self):
        self.kalman.correct(np.reshape(self.position, (2, 1)))
        temp = self.kalman.predict()
        return temp

    def update_position_diameter_appearance(self, position, diameter):
        self.position = position
        self.diameter = diameter
        self.appearance_number = self.appearance_number + 1

    def update_consecutive_appearance(self):
        self.consecutive_appearance = self.consecutive_appearance + 1


def create_candidates(keypoints, candidates, IDs, posy):
    keypoints = [k for k in keypoints if k.pt[1] < posy]
    for k in keypoints:
        candidates.append(CandidateOrFish(len(IDs) + 1, np.array(k.pt, dtype=np.float32), np.array(k.size)))
        IDs.append(len(IDs) + 1)
    return keypoints, candidates, IDs


def promote_candidates_to_fish(fishes, candidates, required_appearances, IDs):
    to_remove = []
    for obj in candidates:
        if getattr(obj, 'appearance_number') == required_appearances and getattr(obj, 'counted') == 0:
            fishes.append(obj)
            to_remove.append(obj)
    for jesm in to_remove:
        candidates.remove(jesm)
    return fishes, candidates


def tracking(keypoints, fishes, candidates, posy):
    # t = 4
    # keypoints = [k for k in keypoints if k.pt[1] < posy+70]
    for jesm in [obj for obj in fishes+candidates if getattr(obj, 'counted') == 0]:
        jesm.update_consecutive_appearance()
        if jesm.position[1] > posy:
            candidates.remove(jesm)
    if [obj for obj in fishes+candidates if getattr(obj, 'counted') == 0] is None or len(keypoints) == 0:
        for jesm in candidates:
            if jesm.consecutive_appearance > jesm.appearance_number:
                candidates.remove(jesm)
        return keypoints, fishes, candidates
    for jesm in [obj for obj in fishes+candidates if getattr(obj, 'counted') == 0]:
        shortest_distance = 10000000
        predicted_pos = jesm.update_and_predict()
        to_remove = []
        for k in keypoints:
            k_distance = np.linalg.norm(predicted_pos[0:2, 0] - np.array(k.pt))

            if k_distance < shortest_distance:
                shortest_distance = k_distance
                nearest_keypoint = k
                to_remove = [k]
        # if shortest_distance > 50:
        #     continue
        for obj in to_remove:
            keypoints.remove(obj)
        jesm.update_position_diameter_appearance(np.array(nearest_keypoint.pt, dtype=np.float32), nearest_keypoint.size)
        if len(keypoints) == 0:
            break
    for jesm in [obj for obj in candidates if getattr(obj, 'counted') == 0]:
        if jesm.consecutive_appearance > jesm.appearance_number:
            candidates.remove(jesm)
    return keypoints, fishes, candidates


def counting(fishes, posy, count, counted_fish):
    median_dm = 0.2323
    if len(counted_fish) != 0:
        median_dm = []
        for jesm in counted_fish:
            median_dm.append(jesm.diameter)
        median_dm = np.median(median_dm)

    if count == 1:
        t = 3
    for jesm in fishes:
        if jesm.counted == 0 and jesm.position[1] > posy:
            if jesm.diameter / median_dm <= 2 or median_dm == 0.2323:
                count = count + 1
            elif 2 < (jesm.diameter / median_dm) < 3:
                count = count + 3
            elif 3 < (jesm.diameter / median_dm) < 3.5:
                count = count + 4
            elif 3.5 < (jesm.diameter / median_dm):
                count = count + 5
            jesm.counted = 1
            counted_fish.append(jesm)
            # fishes.remove(jesm)
    return count, counted_fish
