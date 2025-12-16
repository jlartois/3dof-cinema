import cv2
import numpy as np
import math

def cubemap_side_faces_from_equirect(img, width, fov):
    """
    Convert a 360deg image to a cubemap (but only keep 4 of the faces: left, front, right, and back).
    Returns a list of 4 images of shape (width, width).
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
    
def equirect_from_cubemap_lfrb(faces_merged, in_w, out_w, out_h, fov):
    """
    Convert a cubemap (only the 4 sides, each shape (in_w, in_w)) back to a 360deg image of shape (out_h, out_w).
    Returns a 360deg image.
    """
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