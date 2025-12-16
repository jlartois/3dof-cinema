import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

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
#########################################

def cubemap_side_faces_from_equirect(img, width, fov):
    """
    Convert a 360deg image to a cubemap (but only keep 4 of the faces: left, front, right, and back).
    Returns a list of 4 images of resolution (width, width).
    """
    h, w = img.shape[:2]
    u = np.linspace(-1, 1, width)
    v = np.linspace(-1, 1, width)
    u, v = np.meshgrid(u, v)
    f = 1 / math.tan(fov / 2)

    x, y, z = u, -v, np.ones_like(u) * f
    norm = np.sqrt(x*x + y*y + z*z)
    x, y, z = x/norm, y/norm, z/norm
    
    phi = np.arcsin(y)
    v_eq = (0.5 - phi / math.pi) * h
    map_y = v_eq.astype(np.float32)

    lfrb_faces = []
    for face, theta_offset in zip(["l", "f", "r", "b"] , [-math.pi * 0.5, 0, math.pi * 0.5, math.pi]):
        theta = np.arctan2(x, z) + theta_offset
        u_eq = (theta / (2 * math.pi) + 0.5) * w
    
        map_x = u_eq.astype(np.float32)
    
        lfrb_faces.append(cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
    return lfrb_faces
    
def calculate_disparity_maps(left, right, max_disp, blockSize, use_wls):
    # calculate the left disparity map
    left_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=blockSize)
    left_disp = left_matcher.compute(left, right)
    
    # calculate the right disparity map
    left_180 = cv2.rotate(left, cv2.ROTATE_180)
    right_180 = cv2.rotate(right, cv2.ROTATE_180)
    right_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=blockSize)
    right_disp_180 = right_matcher.compute(right_180, left_180) # order important! 
    right_disp = cv2.rotate(right_disp_180, cv2.ROTATE_180)
    
    # wls filter
    if use_wls:
        left_disp_180 = cv2.rotate(left_disp, cv2.ROTATE_180)
    
        wls_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=1, blockSize=blockSize)
        left_wls_filter = cv2.ximgproc.createDisparityWLSFilter(wls_matcher)
        right_wls_filter = cv2.ximgproc.createDisparityWLSFilter(wls_matcher)
    
        left_disp = left_wls_filter.filter(left_disp, left, disparity_map_right=-right_disp, right_view=right)
        right_disp = right_wls_filter.filter(right_disp_180, right_180, disparity_map_right=-left_disp_180, right_view=left_180)
        right_disp = cv2.rotate(right_disp, cv2.ROTATE_180)
    
    return left_disp, right_disp
    
def disp_to_depth(disp, baseline, focal):
    # disp < 0 means occlusions
    # disp = 0 means infinite depth
    is_occlusion = disp < 0
    is_inf_depth = disp == 0
    
    disp = np.clip(disp, 0, None)
    disp_f = disp.astype(np.float32) / 16  # disp is stored in multiples of 16
    
    # disparity to depth
    with np.errstate(divide='ignore'):
        depth = np.divide(baseline * focal, disp_f)
    depth[is_occlusion] = 0
    depth[is_inf_depth] = 1000
    return depth
    
def stereo_depth_estimation(top, bot, max_disp, baseline, focal, blockSize, use_wls):
    left  = cv2.rotate(top, cv2.ROTATE_90_COUNTERCLOCKWISE)
    right = cv2.rotate(bot, cv2.ROTATE_90_COUNTERCLOCKWISE)
    left  = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp, right_disp = calculate_disparity_maps(left, right, max_disp, blockSize, use_wls)
        
    left_depth  = disp_to_depth(left_disp, baseline, focal)
    right_depth = disp_to_depth(right_disp, baseline, focal)
    
    top_depth = cv2.rotate(left_depth , cv2.ROTATE_90_CLOCKWISE)
    bot_depth = cv2.rotate(right_depth, cv2.ROTATE_90_CLOCKWISE)
        
    return top_depth, bot_depth
    
def zdepth_to_euclidean_depth(zdepth, focal):
    """
    Convert a depth map with Z-depths to one with Euclidean depths.
    Z-depth = Z-coordinate of the point in view space.
    Euclidean depth = distance from point in view space to camera origin.
    """
    h, w = zdepth.shape
    cx = w * 0.5
    cy = h * 0.5

    # pixel grid
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    # unproject to view space
    X = (uu + 0.5 - cx) / focal * zdepth
    Y = (vv + 0.5 - cy) / focal * zdepth
    Z = zdepth

    # distance to camera origin
    return np.sqrt(X*X + Y*Y + Z*Z)
    
def normalize_depth_to_uint16(depth, near, far):
    is_occlusion = depth == 0
    is_inf_depth = depth == 1000
    depth_m = np.clip(depth, near, far)
    
    # IMPORTANT:
    # like MPEG-I datasets, scale the metric depth 'depth_m' to a normalized one 'depth_norm'
    # so that they are inversely proportional.
    # That way, at low depths, the precision is higher than for far away depths.
    depth_norm = (1 / depth_m - 1.0 / far) / (1.0 / near - 1.0 / far)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_norm = (depth_norm * 65535).astype(np.uint16)
    depth_norm = np.clip(depth_norm, 1, 65534)
    
    # IMPORTANT: 
    # Set occlusions to 2^16 - 1
    # Set infinite depth to 0
    depth_norm[is_occlusion] = 65535
    depth_norm[is_inf_depth] = 0
    
    return depth_norm
    
def equirect_from_cubemap_lfrb(faces_merged, in_w, out_w, out_h, fov):
    faces = [faces_merged[:,:in_w], faces_merged[:,in_w:2*in_w], faces_merged[:,2*in_w:3*in_w], faces_merged[:,3*in_w:]]
    f = 1.0 / math.tan(fov * 0.5)

    # equirect sampling grid
    u = (np.arange(out_w) + 0.5) / out_w
    v = (np.arange(out_h) + 0.5) / out_h
    uu, vv = np.meshgrid(u, v)

    theta = (uu - 0.5) * 2 * math.pi
    phi = (0.5 - vv) * math.pi

    # spherical -> cartesian
    x_sph = np.cos(phi) * np.sin(theta)
    y_sph = np.sin(phi)
    z_sph = np.cos(phi) * np.cos(theta)

    # Face centers (yaw)
    centers = np.array([-0.5*math.pi, 0.0, 0.5*math.pi, math.pi], dtype=float)

    # pick closest face
    th = theta[None, ...]
    centers_sh = centers[:, None, None]
    delta = (th - centers_sh + math.pi) % (2*math.pi) - math.pi
    face_idx = np.argmin(np.abs(delta), axis=0).astype(np.int32)
    
    # rotations
    cos_c = np.cos(centers)
    sin_c = np.sin(centers)
    x = x_sph[None, ...]; y = y_sph[None, ...]; z = z_sph[None, ...]
    cos = cos_c[:, None, None]; sin = sin_c[:, None, None]
    x_rot = cos * x - sin * z
    y_rot = y
    z_rot = sin * x + cos * z

    eps = 1e-9
    z_rot_safe = z_rot + (z_rot == 0) * eps
    x_face = x_rot / z_rot_safe
    y_face = -y_rot / z_rot_safe

    u_face = ((x_face * f + 1) * 0.5 * (in_w - 1)).astype(np.float32)
    v_face = ((y_face * f + 1) * 0.5 * (in_w - 1)).astype(np.float32)

    eq = np.zeros((out_h, out_w, 3) if faces[0].shape[-1] == 3 else (out_h, out_w), dtype=faces[0].dtype) - 1
    for i, face_img in enumerate(faces):
        mask = (face_idx == i)
        if not mask.any():
            continue
        map_x = u_face[i][mask].reshape((out_h, out_w // 4))  # (2048, 1024)
        map_y = v_face[i][mask].reshape((out_h, out_w // 4))  # (2048, 1024)

        # mask2 denotes the pixels that are inside of the face
        # mask3 idem, but full width 4096 instead of 1/4th width 1024
        mask2 = np.logical_and(map_y > 0, map_y < in_w)                                 # (2048, 1024)
        mask3 = np.logical_and(np.logical_and(mask, v_face[i] > 0), v_face[i] < in_w)   # (2048, 4096)
        map_x[~mask2] = 0
        map_y[~mask2] = 0
        warped = cv2.remap(face_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        eq[mask3] = warped[mask2]
    return eq


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
        
        # IMPORTANT: read the notes in normalize_depth_to_uint16()
        top_depth_norm = normalize_depth_to_uint16(top_depth, near, far)
        bot_depth_norm = normalize_depth_to_uint16(bot_depth, near, far)

        # convert back to 360 depth map
        top_depth_norm = equirect_from_cubemap_lfrb(top_depth_norm, pinhole_res, equirectangular_res[0], equirectangular_res[1], fov)
        bot_depth_norm = equirect_from_cubemap_lfrb(bot_depth_norm, pinhole_res, equirectangular_res[0], equirectangular_res[1], fov)

        # save as 16-bit greyscale image
        cv2.imwrite("top_depth{:04d}.png".format(frame), top_depth_norm)
        cv2.imwrite("bot_depth{:04d}.png".format(frame), bot_depth_norm)

        print("frame", frame, "done")
        frame += 1
        if frame > 0:
            break
except Exception as e:
    print(e)
    raise e
finally:
    for video in videos:
        if video.isOpened():
            video.release()
    cv2.destroyAllWindows()
print("DONE")