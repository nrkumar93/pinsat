# zed
import pyzed.sl as sl
import pickle

# Create a ZED camera object
zed = sl.Camera()
# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K
init_params.camera_disable_self_calib = True

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
