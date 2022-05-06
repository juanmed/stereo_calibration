"""
The following codes demonstrate how to take two images, taken (ideally) at the same time,
and compute the homogeneous transformation (rotation + translation) between them.
This is known as "extrinsic calibration"
The images should contain a checkerboard pattern with known (and very precise) dimensions.
We also need knowledge of the intrinsic calibration paremeters (intrinsic matrix,
distortion coefficients), which should be obtained in a previous step known as
"intrinsic calibration"

For documentation on the openCV Functions used, you can search for:
openCV Camera Calibration and 3D Reconstruction
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

"""

import cv2
import numpy as np 
import sys
import matplotlib.pyplot as plt


# Parameters of chessboard used. Also, create a matrix of corner positions (obj_points) in 
# the frame of the chessboard. 
pattern_size = (4,6) #col, row
pattern_length = 0.076 # meters
obj_points = np.zeros((np.prod(pattern_size), 3), np.float32)
obj_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
obj_points *= pattern_length

# Azure and helios intrinsic parameters. These values were obtained
# from the accompaning software to each camera
azure_K = np.array([975.7100830078125, 0.0, 1028.1383056640625, 
					0.0, 975.6234130859375, 766.72119140625, 
					0.0, 0.0, 1.0]).reshape(3,3)
azure_D = np.array( [0.585040807723999, -2.6008243560791016, 0.0004827356606256217, 
					-0.00011253785487497225, 1.4300870895385742, 0.4682285487651825, 
					-2.440357208251953, 1.367829442024231])
azure_width = 2048
azure_height = 1536

helios_K = np.array([525.993332,0., 318.877762,
					 0, 525.993332, 238.640747,
					 0., 0., 1]).reshape(3,3)
helios_D = np.array([-0.228452,0.109995, 
					 -0.000057,-0.000041, -0.038704])
helios_width = 640
helios_height = 480

# Read azure and helios images
azure_image_path = sys.argv[1]
helios_image_path = sys.argv[2]

azure_image = cv2.imread(azure_image_path, cv2.IMREAD_UNCHANGED)
helios_file = open(sys.argv[2])
helios_data = np.fromfile(helios_file, dtype=np.uint16, count = 640*480*4)
helios_data = helios_data.reshape((helios_height, helios_width, -1))
helios_file.close()

# conver5 azure image to grayscale and get intensity image from helios .raw file
azure_gray = cv2.cvtColor(azure_image, cv2.COLOR_BGR2GRAY)
helios_image = helios_data[:,:,3].astype(np.float64)*0.25/1000.
helios_gray = (255 * helios_image / np.max(helios_image)).astype(np.uint8)

# find chessboard corners over each image
azure_success, azure_corners = cv2.findChessboardCorners(azure_gray, pattern_size, None)
helios_success, helios_corners = cv2.findChessboardCorners(helios_gray, pattern_size, None)

# if corners where found, refine them using cornerSubPix
if azure_success:
	print("Azure Image Corners FOUND!")
else:
	raise Exception
if helios_success:
	print("Helios Image Corners FOUND!")
else:
	raise Exception

# the criteria for refinement end is 30 iterations or epsilon < 0.001
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
r_azure_corners = cv2.cornerSubPix(azure_gray, azure_corners, (11, 11), (-1, -1), criteria)
r_helios_corners = cv2.cornerSubPix(helios_gray, helios_corners, (11, 11), (-1, -1), criteria)

# draw the corners to verify
azure_image = cv2.drawChessboardCorners(azure_image, pattern_size, 
										r_azure_corners, azure_success)

helios_rgb = cv2.cvtColor(helios_gray, cv2.COLOR_GRAY2RGB)
helios_rgb = cv2.drawChessboardCorners(helios_rgb, pattern_size, 
										r_helios_corners, helios_success)


# calibrate the stereo pair
metric = {}   

calibration_term_crit=(cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS, 30, 1e-6) # default 30,1e-6
calibration_flag = 0
calibration_flag |= cv2.CALIB_USE_INTRINSIC_GUESS  #
calibration_flag |= cv2.CALIB_FIX_INTRINSIC # Use the given intrinsic values

# we have to resize each array of corners to fit what is expecte by stereoCalibrate
# read more https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
obj_points = obj_points
r_azure_corners2 = r_azure_corners.reshape(np.prod(pattern_size),-1)	
r_helios_corners2 = r_helios_corners.reshape(np.prod(pattern_size),-1)


(metric['retval'], metric['K1'], metric['d1'], metric['K2'], metric['d2'],
metric['R'], metric['t'], metric['E'],
metric['F']) = cv2.stereoCalibrate(
        [obj_points], # chessboard corners in chessboard's local frame
        [r_azure_corners2], # corners in image plane of camera 1
        [r_helios_corners2], # corners in imaga plane of camera 2
        azure_K, 
        azure_D,
        helios_K,
        helios_D,
        imageSize=(azure_width, azure_height), #not important because we used cv2.CALIB_FIX_INTRINSIC option
        flags=calibration_flag,
        criteria=calibration_term_crit
        )

print("Pose of azure camera as seen from helios camera: ")
print("Position:\n{}".format(metric['t']))
print("Orientation:\n{}".format(metric['R']))
print("Fundamental Matrix:\n{}".format(metric['F']))
print("Essential Matrix:\n{}".format(metric['E']))

H_AH = np.zeros((4,4))  # Homogeneous transform from azure to helios
H_AH[0:3,0:3] = metric['R']
H_AH[0:3,3] = metric['t'].T[0]
H_AH[3,3] = 1.0

print("Transform Azure -> Helios obtained by stereoCalibrate")
print("[%f, %f, %f, %f,"%(H_AH[0][0], H_AH[0][1], H_AH[0][2], H_AH[0][3]))
print("%f, %f, %f, %f,"%(H_AH[1][0], H_AH[1][1], H_AH[1][2], H_AH[1][3]))
print("%f, %f, %f, %f,"%(H_AH[2][0], H_AH[2][1], H_AH[2][2], H_AH[2][3]))
print("%f, %f, %f, %f]"%(H_AH[3][0], H_AH[3][1], H_AH[3][2], H_AH[3][3]))

# show images
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
ax1.imshow(cv2.cvtColor(azure_image, cv2.COLOR_BGR2RGB))
ax2.imshow(helios_rgb)

plt.tight_layout()
plt.show()

# The same but step by step for each camera
# rvec,tvec  are a transformation from the "w"orld coordinate system to the "c"amera 
# coordinate system

azure_success, azure_rvec, azure_tvec = cv2.solvePnP(obj_points, r_azure_corners,
										 azure_K, azure_D, flags = cv2.SOLVEPNP_ITERATIVE)
azure_rvec, azure_tvec = cv2.solvePnPRefineVVS(obj_points, r_azure_corners,
										 azure_K, azure_D, azure_rvec, azure_tvec)
helios_success, helios_rvec, helios_tvec = cv2.solvePnP(obj_points, r_helios_corners, 
										   helios_K, helios_D, flags = cv2.SOLVEPNP_ITERATIVE)
helios_rvec, helios_tvec = cv2.solvePnPRefineVVS(obj_points, r_helios_corners, 
										helios_K, helios_D, helios_rvec, helios_tvec)


azure_R = np.zeros((3,3))
helios_R = np.zeros((3,3))

T_WA = np.zeros((4,4))  # World -> Azure camera transformation
T_WH = np.zeros((4,4))  # World -> Helios camera transformation

cv2.Rodrigues(azure_rvec, azure_R)
cv2.Rodrigues(helios_rvec, helios_R)

T_WA[0:3,0:3] = azure_R
T_WA[0:3,3] = azure_tvec.T[0]

T_WH[0:3,0:3] = helios_R
T_WH[0:3,3] = helios_tvec.T[0]

T_WA[3][3] = 1.0
T_WH[3][3] = 1.0

T_AW = np.zeros((4,4)) # Azure -> World transformation
T_AW[0:3, 0:3] = azure_R.T
T_AW[0:3,3] = -np.dot(azure_R.T,azure_tvec.T[0])
T_AW[3,3] = 1.0

T_AH = T_WH.dot(T_AW) # Azure -> Helios Transformation
T_HA = np.linalg.inv(T_AH)
#T_AH = T_WH.dot(np.linalg.inv(T_WA))

print("Transforms obtained using independent PnP reprojection")
print("Transfrom Azure -> Helios")
print("[%f, %f, %f, %f,"%(T_AH[0][0], T_AH[0][1], T_AH[0][2], T_AH[0][3]))
print("%f, %f, %f, %f,"%(T_AH[1][0], T_AH[1][1], T_AH[1][2], T_AH[1][3]))
print("%f, %f, %f, %f,"%(T_AH[2][0], T_AH[2][1], T_AH[2][2], T_AH[2][3]))
print("%f, %f, %f, %f]"%(T_AH[3][0], T_AH[3][1], T_AH[3][2], T_AH[3][3]))

print("Tranform Helios - > Azure")
print("[%f, %f, %f, %f,"%(T_HA[0][0], T_HA[0][1], T_HA[0][2], T_HA[0][3]))
print("%f, %f, %f, %f,"%(T_HA[1][0], T_HA[1][1], T_HA[1][2], T_HA[1][3]))
print("%f, %f, %f, %f,"%(T_HA[2][0], T_HA[2][1], T_HA[2][2], T_HA[2][3]))
print("%f, %f, %f, %f]"%(T_HA[3][0], T_HA[3][1], T_HA[3][2], T_HA[3][3]))



