import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

# RMS: 1.936188835506069
# camera matrix:
K = np.array([[1029.15033, 0.00000000, 623.657540],
 [0.00000000, 1034.83203, 331.378673],
 [0.00000000, 0.00000000, 1.00000000]])
# distortion coefficients:  [ 0.06349627 -0.44519871 -0.00914446 -0.0082998   0.49303731]
dist_coeffs = [ 0.06349627, -0.44519871, -0.00914446, -0.0082998,   0.49303731]

cap = cv2.VideoCapture('webcam_vid5.avi')


# cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cam_w = 1280
cam_h = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH,cam_w);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_h);

print(cam_w , cam_h)


cam = PinholeCamera(cam_w, cam_h, 1029.15, 1034.83, cam_w/2, cam_h/2, K, dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4])
vo = VisualOdometry(cam)

# cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
# vo = VisualOdometry(cam)

traj = np.zeros((cam_h,cam_w,3), dtype=np.uint8)

size = (int(cam_w*2),int(cam_h))
videoWriter = cv2.VideoWriter(
    'ECEn631_Visual_Odometry.avi', cv2.VideoWriter_fourcc('M','P','E','G'), 20.0, size, isColor=True)

img_id = 0
FORGET_FACTOR = 1024
prev_points = [None] * FORGET_FACTOR
prev_t = [None] * FORGET_FACTOR

# for img_id in xrange(4541):
while(1):

	# img = cv2.imread('/home/jared/Documents/ECEn631/Odometry/monoVO-python/00/'+str(img_id).zfill(6)+'.png', 0)

	ret_val, img = cap.read()

	# print(img[0][0])
	# print(traj[0][0])
	# print(img.shape)
	# print(traj.shape)
	gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	vo.update(gray_img,img_id)
	# print(vo.points_3D)

	cur_t = vo.cur_t
	px_ref = vo.px_ref
	if(img_id > 2):
		# Shift everything by one position
		for i in range(FORGET_FACTOR-1, 0, -1):
			prev_points[i] = prev_points[i-1]
			prev_t[i] = prev_t[i-1]
		# Update location 0
		prev_points[0] = vo.points_3D
		prev_t[0] = cur_t

		# Draw circles and fade
		for i in range(FORGET_FACTOR-1, 0, -1):
			if prev_points[i] is None:
				continue
			for point in prev_points[i]:
				draw_x, draw_y = point[0] + 300 + prev_t[i][0], point[1]+300 + prev_t[i][2]
				cv2.circle(traj, (draw_x,draw_y), 2, (max(255/(i/4),64),min(1024-i,64),max(255/(i/4), 64), 1))
		# Draw most recent 3d points
		for point in prev_points[0]:
			draw_x, draw_y = point[0] + 300 + prev_t[0][0], point[1]+300 + prev_t[0][2]
			cv2.circle(traj, (draw_x,draw_y), 2, (255,0,255), 1)


		# # x, y, z = cur_t[0], cur_t[1], cur_t[2]
		# for point in vo.points_3D:
		# 	# print(point)
		# 	draw_x, draw_y = point[0] + 300 + cur_t[0], point[1]+300 + cur_t[2]
		# 	cv2.circle(traj, (draw_x,draw_y), 2, (img_id*255/4540,255-img_id*255/4540,0), 1)
		# for point in prevPoints:
		# 	draw_x, draw_y = point[0] + 300 + prevt[0], point[1]+300 + prevt[2]
		# 	cv2.circle(traj, (draw_x,draw_y), 2, (0,0,0), 1)
		# prevPoints = vo.points_3D
		# prevt = cur_t


	else:
		x, y, z = 0., 0., 0.
		# draw_x, draw_y = int(x)+300, int(z)+300
		# cv2.circle(traj, (draw_x,draw_y), 2, (img_id*255/4540,255-img_id*255/4540,0), 1)
		# cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	# true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

	# Add tracked features to video feed
	for i in px_ref:
		# print(i)
		coord = (i[0],i[1])
		img = cv2.circle(img,coord,3,(255,0,255),-1)


	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
	# cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	both = np.hstack((img,traj))

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	videoWriter.write(both)	
	cv2.waitKey(1)

	img_id = img_id + 1

cv2.imwrite('map.png', traj)
