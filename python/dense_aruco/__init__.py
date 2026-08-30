"""OpenCV-style ArUco detector with a C++ dense-refinement backend.

Drop-in for ``cv2.aruco.ArucoDetector``:

    det = ArucoDetector(cv2.aruco.DICT_4X4_1000, params, denseRefine="all")
    corners, ids, rejected = det.detectMarkers(image)
    rvecs, tvecs, ok = det.estimatePoseSingleMarkers(
        corners, marker_length, K, dist, ids=ids, image=image
    )
    det.drawLastRefinement(image)
"""

from dense_aruco._core import ArucoDetector, DetectorParameters, getPredefinedDictionary

__all__ = ["ArucoDetector", "DetectorParameters", "getPredefinedDictionary"]
