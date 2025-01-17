import numpy as np 
# np.set_printoptions(threshold=np.inf)
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1450

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, K,
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]
		self.K = K


class VisualOdometry:
	def __init__(self, cam):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=8, nonmaxSuppression=True)
		self.points_3D = None
		self.px_ref_window = None
		self.px_cur_window = None

		# with open(annotations) as f:
		# 	self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

		self.px_ref_window = np.zeros(shape=(0,2))
		self.px_cur_window = np.zeros(shape=(0,2))

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		# TODO: Try out mask
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)

		self.px_ref_window = np.zeros(shape=(0,2))
		self.px_cur_window = np.zeros(shape=(0,2))

		for i in range(0, len(self.px_ref)):
			if(self.px_ref[i][1] <= 720 and self.px_ref[i][1] >= 400):
				self.px_ref_window = np.append(self.px_ref_window, np.array([[self.px_ref[i][0],self.px_ref[i][1]]]), axis=0)
				self.px_cur_window = np.append(self.px_cur_window, np.array([[self.px_cur[i][0],self.px_cur[i][1]]]), axis=0)

		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		# absolute_scale = self.getAbsoluteScale(frame_id)
		# if(absolute_scale > 0.1):

		# Inserted Code
		M_1 = np.hstack((R, t))
		M_2 = np.hstack((np.eye(3,3), np.zeros((3,1))))

		P_1 = np.dot(self.cam.K, M_1)
		P_2 = np.dot(self.cam.K, M_2)

		points_4d_homogeneous = cv2.triangulatePoints(P_1, P_2, self.px_ref_window.T, self.px_cur_window.T)
		points_4d = points_4d_homogeneous / np.tile(points_4d_homogeneous[-1, :], (4,1))
		points_3d = points_4d[:3, :].T
		############################################

		# Pass the 3D points up to the object to later be plotted.
		# points_3d = np.array(points_3d).astype(int)
		self.points_3D = set()
		for point in points_3d:
			point = self.cur_R.dot(point)
			my_tuple = (int(point[0]), int(point[2]))
			self.points_3D.add(my_tuple)


		# Update location
		self.cur_t = self.cur_t + 0.5*self.cur_R.dot(t) 
		self.cur_R = R.dot(self.cur_R)

		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame