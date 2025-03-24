from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import cv2
config = OpenCVCameraConfig(camera_index=2)
camera = OpenCVCamera(config)
camera.connect()
while True:
    color_image = camera.read()
    cv2.imshow("Camera", color_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    

camera.disconnect()
#print(color_image.shape)
#print(color_image.dtype)


