[opencv-download]:                 https://opencv.org/releases.html
[opencv-version-badge]:            https://img.shields.io/badge/OpenCV%20Version-4.0.1-green.svg

[![Github Release][opencv-version-badge]][opencv-download]

# SETUP
## 1. Download OpenCV and extract it somewhere

## 2. Locate Edit Env Variables
**Run** Win+R => `rundll32 sysdm.cpl,EditEnvironmentVariables` or Simply locate it in Windows Settings

## 3. Create Windows Environment System variable:
**Key**: `OPENCV_DIR`

**Value**: Directory to extracted opencv build dir, example `C:\opencv_41\opencv\build`

## 4. Append OpenCV bin path to PATH (User Variable)
Example `C:\opencv\build\x64\vc14\bin`
