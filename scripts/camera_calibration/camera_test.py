import os
import cv2
import pyzed.sl as sl

save_dir = 'calib_data'



# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

# Enable positional tracking with default parameters
tracking_parameters = sl.PositionalTrackingParameters()
err = zed.enable_positional_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# image = sl.Mat()
# if zed.grab() == sl.ERROR_CODE.SUCCESS:
#   # A new image is available if grab() returns SUCCESS
#   ret = zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve the left image
#   if ret == sl.ERROR_CODE.SUCCESS:
#       f = os.path.join(save_dir, 'left', 'test1080.png')
#       image.write(f)

sensors_data = sl.SensorsData()

# Grab new frames and retrieve sensors data
if zed.grab() == sl.ERROR_CODE.SUCCESS :
    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) # Retrieve only frame synchronized data


# Extract IMU data
imu_data = sensors_data.get_imu_data()
pose = sl.Transform()
imu_data.get_pose(pose)
print(pose)

