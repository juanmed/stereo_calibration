import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

header = "ply\n\
format ascii 1.0\n\
element vertex {}\n\
property float x\n\
property float y\n\
property float z\n\
property uint8 red\n\
property uint8 green\n\
property uint8 blue\n\
end_header\n" 

def get_colorPointCloud(depth, img, intrinsic):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) # & (depth < maxval)
    z = depth #np.where(valid, depth, np.nan)
    x =  z * (c - intrinsic[0,2]) / intrinsic[0,0] #np.where(valid, z * (c - self.rgb_cx) / self.rgb_fx, 0)
    y = z * (r - intrinsic[1,2]) / intrinsic[1,1] # np.where(valid, z * (r - self.rgb_cy) / self.rgb_fy, 0)  
    return np.dstack((x,y,z,img)).astype(np.float32) 

def get_pointCloud(depth, img, intrinsic):
    """
    To generate pointCloud2 message refer to:
    https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
    """
    points = []
    points = get_colorPointCloud(depth, img, intrinsic)
    rows,cols,channels = points.shape
    points = points.reshape(rows*cols,channels)
    points = points[~np.isnan(points).any(axis=1), :]
    cond1= (points[:,0]>0) 
    cond2= (points[:,1]>0)
    cond3= (points[:,2]>0)
    points = points[np.logical_and(cond1,np.logical_and(cond2, cond3))]
    #print("Before sampling: ", points.shape)
    #factor=1
    #points = points[::factor]
    #print("After sampling: ",points.shape)
    # Displaying the array
    file = open("cloud.ply", "w+")    
    content = str(points)
    file.write(header.format(points.shape[0]))
    for row in points:
        line = ""
        for i in range(3):
            line += str(row[i])+" "
        for i in range(3,6):
            line += str(int(row[i])) + " "
        file.write(line + "\n")
    file.close()
    return 0

# my implementaion that produces the expected output
def registerDepth(intrinsics_depth, intrinsics_color, dist_coef_color, extrinsics, depth, shape, depthDilation):
    #assert dist_coef_color is None
    assert depthDilation==False

    width, height = shape
    out = np.zeros((height,width))
    y,x = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
    x=x.reshape(1,-1)
    y=y.reshape(1,-1)
    z=depth.reshape(1,-1)
    x=(x-intrinsics_depth[0,2])/intrinsics_depth[0,0]
    y=(y-intrinsics_depth[1,2])/intrinsics_depth[1,1]
    pts = np.vstack((x*z,y*z,z))
    pts = extrinsics[:3,:3]@pts+extrinsics[:3,3:]
    pts = intrinsics_color@pts
    px = np.round(pts[0,:]/pts[2,:])
    py = np.round(pts[1,:]/pts[2,:])
    mask = (px>=0) * (py>=0) * (px<width) * (py<height)
    out[py[mask].astype(int),px[mask].astype(int)] = pts[2,mask]
    return out

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

# convert azure image to grayscale and get intensity image from helios .raw file
azure_gray = cv2.cvtColor(azure_image, cv2.COLOR_BGR2GRAY)
helios_depth = helios_data[:,:,2].astype(np.float32)*0.25/1000.

#Homogeneous transform from helios to azure
R = np.array([[ 0.9998842, 0.0103368,-0.01116872],
              [-0.01028264,0.99993515,0.00489567],
              [ 0.0112186, -0.00478026,0.99992564]])
t = np.array([-0.00307042,
              -0.06776994,
              -0.02385673])
Rha = np.zeros((4,4), dtype=np.float32) 
Rha[0:3,0:3] = R
Rha[0:3,3] = t
Rha[3,3] = 1.0
Rha = np.linalg.inv(Rha)
Rha = np.array([0.999822, -0.010424, 0.015758, -0.001993,
0.010317, 0.999923, 0.006839, 0.055544,
-0.015828, -0.006676, 0.999852, 0.007331,
0.000000, 0.000000, 0.000000, 1.000000]).reshape(4,4)

# undistort the rgb and depth image before registration
# cv2.undistort uses bilinear interpolation... fine for RGB
# but for depth, NEAREST interpolation must be used, otherwise 
# some incorrect artifacts appear
azure_K_new, roi = cv2.getOptimalNewCameraMatrix(azure_K, azure_D, (azure_width, azure_height), 1.0, (azure_width,azure_height))
print(azure_K_new)
azure_image_undistort = cv2.undistort(azure_image, azure_K, azure_D, None, azure_K_new) 

helios_K_new, roi = cv2.getOptimalNewCameraMatrix(helios_K, helios_D, (helios_width, helios_height), 1.0, (helios_width,helios_height))
mapx, mapy = cv2.initUndistortRectifyMap(helios_K, helios_D, None, helios_K_new, (helios_width,helios_height), cv2.CV_32FC1)
helios_depth_undistort = cv2.remap(helios_depth, mapx, mapy, cv2.INTER_NEAREST)



# project the depth image onto the rgb image coordinate frame: registration
registered_depth = registerDepth(helios_K_new, azure_K_new, azure_D, Rha, helios_depth_undistort, (azure_width, azure_height), depthDilation=False)
registered_depth_gray = (255 * registered_depth / np.max(registered_depth)).astype(np.uint8)
aligned_depth_colormap = cv2.applyColorMap( registered_depth_gray, cv2.COLORMAP_HOT )

# create and save a point cloud to verify the result
pc_pcl = get_pointCloud(helios_depth, np.ones((helios_depth.shape[0], helios_depth.shape[1],3))*255, helios_K)

# show images
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
ax1.imshow(cv2.cvtColor(azure_image_undistort, cv2.COLOR_BGR2RGB))
ax2.imshow(registered_depth_gray, cmap='viridis')

cv2.imwrite("helios_registered.png",registered_depth_gray)
cv2.imwrite("azure_undistorted.png",azure_image_undistort)


plt.tight_layout()
plt.show()

