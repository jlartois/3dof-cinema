import cv2
import numpy as np
import math

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
    
def save_uint16_to_yuv420p16le(filename, img, is_append):
    H, W = img.shape
    Y = img
    U = np.full((H // 2, W // 2), 32768, dtype=np.uint16)
    V = np.full((H // 2, W // 2), 32768, dtype=np.uint16)

    with open(filename, "wb" if not is_append else "ab") as f:
        Y.tofile(f)
        U.tofile(f)
        V.tofile(f)