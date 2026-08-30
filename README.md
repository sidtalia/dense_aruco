# dense_aruco

C++ ArUco detector with optional **dense internal-bit pose refinement**.
Python API is a drop-in for `cv2.aruco.ArucoDetector`.

This is a **pip / scikit-build** package, not a ROS package. `COLCON_IGNORE` is
present so `colcon build` skips it.

## Install

Needs OpenCV with ArUco (`opencv2/objdetect/aruco_detector.hpp` on 4.7+, or
`opencv2/aruco.hpp` from contrib on older builds), CMake, and a C++17 compiler.

```bash
python3 -m pip install --user --break-system-packages -e /path/to/dense_aruco
```

If bring-up runs as `root`, install with that same interpreter (or `sudo` the
command). Confirm:

```bash
python3 -c "from dense_aruco import ArucoDetector; print('ok')"
```

Rebuild after C++ changes by re-running the pip install.

## Use (OpenCV-shaped)

```python
import cv2
import numpy as np
from dense_aruco import ArucoDetector, DetectorParameters

params = DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
params.cornerRefinementWinSize = 10
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.1

det = ArucoDetector(
    cv2.aruco.DICT_4X4_1000,   # or "DICT_4X4_1000"
    params,
    denseRefine="all",         # off | outer | internal | all
    denseOptimize="both",      # h | pose | both
)

corners, ids, rejected = det.detectMarkers(image)
rvecs, tvecs, ok = det.estimatePoseSingleMarkers(
    corners,
    marker_length_m,
    K,
    dist,
    ids=ids,
    image=image,               # omit to skip dense refine (PnP only)
)
det.drawLastRefinement(image)  # in-place overlay
```

`detectMarkers` returns the same shapes as OpenCV (list of `(1,4,2)` corners,
`(N,1)` ids). `estimatePoseSingleMarkers` returns `rvecs`/`tvecs` as `(N,1,3)`
plus a list of booleans. Pose + dense refine for **2+ markers** runs on a
thread pool (`min(N, hardware_concurrency)`). One marker stays single-threaded.

Pass `image=` so the C++ backend can sample internal black/white cell edges.
Without it, pose is IPPE + LM on the four outer corners only.

## Constructor

| argument | meaning |
|---|---|
| `dictionary` | `cv2.aruco.DICT_*` int or name string (not a `Dictionary` object) |
| `detectorParams` | `DetectorParameters` or an OpenCV params object with the same attribute names |
| `denseRefine` | `off`, `outer`, `internal`, `all` |
| `denseOptimize` | `h` (homography), `pose` (6-DoF), `both` |
| `samplesPerEdge` | samples along each bit/outer boundary (default 6) |

`DetectorParameters` fields match OpenCV names:
`cornerRefinementMethod`, `cornerRefinementWinSize`,
`cornerRefinementMaxIterations`, `cornerRefinementMinAccuracy`.

## ROS (`aruco_registration`)

The Python detector node imports this backend.
SSoT keys:

```yaml
dictionary: DICT_4X4_1000
dense_refine: all          # off | outer | internal | all
dense_refine_optimize: both
dense_refine_draw: true
```

Map←odom stays in `aruco_registration`; this package is detect + refine only.
