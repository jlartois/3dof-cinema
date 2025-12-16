# 3DOF+ Cinema

**In short:** Create 3DoF+ experiences using two 360° cameras (stacked on on top of the other), stereo depth estimation ([OpenCV SGBM](https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html)) and Depth-Image-Based rendering ([OpenDIBR](https://github.com/IDLabMedia/open-dibr)).

## 360° stereo depth estimation

The camera rig consists of two 360° cameras stacked one on top of the other. We provide the video that each camera recorded: download `class_dynamic_top.mp4` and `class_dynamic_bot.mp4` from [here](https://cloud.ilabt.imec.be/index.php/s/REtkkGjEASytkJZ). Estimate a depth map for each frame of each video by running:

```bash
python estimate_depth.py
```

The file allows to set parameters, rather than expecting command line arguments. The output depth maps are written to `out/` as `uint16`/`gray16be` PNGs.

**Important notes:**

* The depth maps contains '**Euclidean depths**', so each depth is the distance from the camera's origin to the point in 3D. 

* The depth maps are stored as `uint16`s. The following conversion was done to go from depth in meters `depth_m` to a `uint16` depth `depth_16b`:
  
  ```
  depth_16b = (1 / depth_m - 1 / far) / (1 / near - 1 / far) * 65535
  ```
  
  So `depth_16b` is inversely proportional to `depth_m`. This is intentional as to increase the precision for smaller depth values. Depth range `[near, far]` is chosen by the user. Note that `65535 = 2^16 - 1`.

* During the disparity estimation, two masks are created: one indicating pixels with **infinite depth** (disparity 0), and one indicating pixels that were **occluded and have no depth** (disparity < 0). In the saved depth maps, the pixels with infinite depth are set to 65535, and the occluded pixels are set to 0. During DIBR, these values need to be handled accordingly (depth = 0 means discard pixels, depth = 65535 means infinite depth).

## Depth-Image-Based Rendering (DIBR)

Firstly convert the depth map PNGs to videos using ffmpeg:

```bash
ffmpeg -framerate 25 -i out\top_depth%04d.png -c:v libx265 -x265-params lossless=1:bframes=0 -preset slow -an -pix_fmt yuv420p12le top_depth.mp4
ffmpeg -framerate 25 -i out\bot_depth%04d.png -c:v libx265 -x265-params lossless=1:bframes=0 -preset slow -an -pix_fmt yuv420p12le bot_depth.mp4
```

Note that **lossless** compression is applied, and that the `uint16` depth maps are converted to **`yuv420p12le`**. This was the maximum bit depth allowed by HEVC/H265 for OpenDIBR when testing.
