import cv2
import numpy as np
import math
import os

from mapping import *
from depth import *

######## FILL IN PARAMETERS HERE ########
baseline = 0.2              # distance (meter) between the optical centers of the cameras
fov = 90 * math.pi / 180    # field of view angle (radians) for the cubemap sides (should be 90 or 120 degrees)
equirectangular_res = [4096, 2048]
pinhole_res = 1024          # width and height (pixels) of one cubemap side
paths = [
    "class_dynamic_top.mp4",   # left/top
    "class_dynamic_bot.mp4"    # right/bottom
]
# depth estimation
max_disp = 80     # maximum disparity (pixels), try to stay as low as possible
blockSize = 10
use_wls = False   # whether or not to use Weighted Least Squares (WLS)

# save depth as uint16, so depth range [near, far] is mapped to [0, 2^16 - 1]
near = 0.5
far = 20

output_folder = "out/"    # directory in which to write the depth maps
#########################################

focal = pinhole_res * 0.5  # focal length (pixels) of cubemap pinnhole camera

videos = [cv2.VideoCapture(p) for p in paths]
try:
    frame = 0
    while videos[0].isOpened() and videos[1].isOpened():
        # read in 1 frame of each video
        images = [video.read()[1] for video in videos]
        if images[0] is None or images[1] is None:
            print("reading from one of the video files returned None, stopping")
            break

        # equirectangular to cubemap (only keep side faces: Left, Front, Right, Back)
        lfrb_top = cubemap_side_faces_from_equirect(images[0], pinhole_res, fov)
        lfrb_bot = cubemap_side_faces_from_equirect(images[1], pinhole_res, fov)
        
        top_depths = []
        bot_depths = []
        for face in range(4):
            top_depth, bot_depth = stereo_depth_estimation(lfrb_top[face], lfrb_bot[face], max_disp, baseline, focal, blockSize, use_wls)
            
            # for 360deg depth maps, we need the Euclidean depth, not the Z-depth
            top_depth = zdepth_to_euclidean_depth(top_depth, focal)
            bot_depth = zdepth_to_euclidean_depth(bot_depth, focal)
            
            top_depths.append(top_depth)
            bot_depths.append(bot_depth)
            
        top_depth = np.hstack(top_depths)
        bot_depth = np.hstack(bot_depths)
        
        # IMPORTANT: read the notes in depth.py in normalize_depth_to_uint16()
        top_depth_norm = normalize_depth_to_uint16(top_depth, near, far)
        bot_depth_norm = normalize_depth_to_uint16(bot_depth, near, far)

        # convert back to 360 depth map
        top_depth_norm = equirect_from_cubemap_lfrb(top_depth_norm, pinhole_res, equirectangular_res[0], equirectangular_res[1], fov)
        bot_depth_norm = equirect_from_cubemap_lfrb(bot_depth_norm, pinhole_res, equirectangular_res[0], equirectangular_res[1], fov)

        # save as 16-bit greyscale image
        cv2.imwrite(os.path.join(output_folder, "top_depth{:04d}.png".format(frame)), top_depth_norm)
        cv2.imwrite(os.path.join(output_folder, "bot_depth{:04d}.png".format(frame)), bot_depth_norm)

        print("frame", frame, "done")
        frame += 1
except Exception as e:
    print(e)
    raise e
finally:
    for video in videos:
        if video.isOpened():
            video.release()
    cv2.destroyAllWindows()
print("DONE")