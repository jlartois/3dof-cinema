# 3DOF+ Cinema

**In short:** Create 3DoF+ experiences using two 360° cameras (stacked on on top of the other), stereo depth estimation ([OpenCV SGBM](https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html)) and Depth-Image-Based rendering ([OpenDIBR](https://github.com/IDLabMedia/open-dibr)).

## 360° stereo depth estimation

The camera rig consists of two 360° cameras stacked one on top of the other. We provide the video that each camera recorded: download `class_dynamic_top.mp4` and `class_dynamic_bot.mp4` from [here](https://cloud.ilabt.imec.be/index.php/s/REtkkGjEASytkJZ). Estimate a depth map for each frame of each video by running:

```bash
python estimate_depth.py
```

The file allows to set parameters, rather than expecting command line arguments. For example, you can enable the Weighted Least Squares filter using variable `use_wls`. The output depth maps are written to `out/top_depth.yuv` and `out/bot_depth.yuv` as `yuv420p16le` .

**Important notes:**

* The depth maps contains '**Euclidean depths**', so each depth is the distance from the camera's origin to the point in 3D. 

* The depth maps are stored as `uint16`s. The following conversion was done to go from depth in meters `depth_m` to a `uint16` depth `depth_16b`:
  
  ```
  depth_16b = (1 / depth_m - 1 / far) / (1 / near - 1 / far) * 65535
  ```
  
  So `depth_16b` is inversely proportional to `depth_m`. This is intentional as to increase the precision for smaller depth values. Depth range `[near, far]` is chosen by the user. Note that `65535 = 2^16 - 1`.

* During the disparity estimation, two masks are created: one indicating pixels with **infinite depth** (disparity 0), and one indicating pixels that were **occluded and have no depth** (disparity < 0). In the saved depth maps, the pixels with infinite depth are set to 65535, and the occluded pixels are set to 0. During DIBR, these values need to be handled accordingly (depth = 0 means discard pixels, depth = 65535 means infinite depth).

Lastly convert the depth map .yuv files to videos using ffmpeg:

```bash
ffmpeg -hide_banner -s:v 4096x2048 -r 25 -pix_fmt yuv420p16le -i out\top_depth.yuv -c:v libx265 -x265-params lossless=1:bframes=0 -preset slow -an -pix_fmt yuv420p12le top_depth.mp4
ffmpeg -hide_banner -s:v 4096x2048 -r 25 -pix_fmt yuv420p16le -i out\bot_depth.yuv -c:v libx265 -x265-params lossless=1:bframes=0 -preset slow -an -pix_fmt yuv420p12le bot_depth.mp4
```

Note that **lossless** compression is applied, and that the `yuv420p16le` depth maps are converted to **`yuv420p12le`**. This was the maximum bit depth allowed by HEVC/H265 for OpenDIBR when testing.

## OpenDIBR

The scene can now be displayed in real-time using DIBR. The input is the original color videos, and the estimated depth map videos. There is also a `config_eq.json` and `config_pinhole.json` which describe the intrinsics and extrinsics of the input camera, and the viewport (the output camera).

Folder `open-dibr` contains a modified version of [OpenDIBR](https://github.com/IDLabMedia/open-dibr). The same requirements as that repo apply:

- A C++11 capable compiler. The following options have been tested:
  - Windows: Visual Studio 2019
  - Ubuntu 18.04 LTS: gcc, g++
- CMake 3.7 or higher
- An NVidia GPU that supports hardware accelerated decoding for H265 4:2:0 (preferably up to 12-bit), [check your GPU here](https://en.wikipedia.org/wiki/Nvidia_NVDEC#GPU_support)
- [CUDA Toolkit 10.2 or higher](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-cuda-development-tools)
- [NVidia Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)

If you want to use Virtual Reality:

- Steam and SteamVR
- The following HMDs have been tested:
  - HTC Vive
  - HTV Vive Pro
  - Oculus Rift S

For Linux systems, there are several dependencies on libraries that need to be installed: FFmpeg, OpenGL, GLEW and SDL2. For example on Ubuntu:

```
sudo apt update
sudo apt install -y pkg-config libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libglew-dev libsdl2-2.0
```

Use CMake to compile and build an executable. Then run one of these commands:

```bash
"path/to/RealtimeDIBR.exe" -i "./" -j "config_pinhole.json" --triangle_deletion_margin 400
"path/to/RealtimeDIBR.exe" -i "./" -j "config_eq.json" --triangle_deletion_margin 400
```

depending on wether you want a pinhole or equirectangular output camera. We refer to the [OpenDIBR Wiki](https://github.com/IDLabMedia/open-dibr/wiki/Running-the-application) for more info on command line options etc.
