import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

# RMS: 1.936188835506069
# camera matrix:
#  [[1.02915033e+03 0.00000000e+00 6.23657540e+02]
#  [0.00000000e+00 1.03483203e+03 3.31378673e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# distortion coefficients:  [ 0.06349627 -0.44519871 -0.00914446 -0.0082998   0.49303731]

cap = cv2.VideoCapture('webcam_vid5.avi')


# cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cam_w = 1280
cam_h = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH,cam_w);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_h);

print(cam_w , cam_h)

cam = PinholeCamera(cam_w, cam_h, 1029.15, 1034.83, cam_w/2, cam_h/2)
vo = VisualOdometry(cam)

# cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
# vo = VisualOdometry(cam)

traj = np.zeros((cam_h,cam_w,3), dtype=np.uint8)

size = (int(cam_w*2),int(cam_h))
videoWriter = cv2.VideoWriter(
    'ECEn631_Visual_Odometry.avi', cv2.VideoWriter_fourcc('M','P','E','G'), 20.0, size, isColor=True)

img_id = 0

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

	cur_t = vo.cur_t
	px_ref = vo.px_ref
	if(img_id > 2):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x)+300, int(z)+300 
	# true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

	for i in px_ref:
		# print(i)
		coord = (i[0],i[1])
		img = cv2.circle(img,coord,5,(0,255,555),-1)

	cv2.circle(traj, (draw_x,draw_y), 2, (img_id*255/4540,255-img_id*255/4540,0), 1)
	# cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
	# cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	both = np.hstack((img,traj))

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	videoWriter.write(both)	
	cv2.waitKey(1)

	img_id = img_id + 1

cv2.imwrite('map.png', traj)
