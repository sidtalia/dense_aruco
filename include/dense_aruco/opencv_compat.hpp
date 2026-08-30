#pragma once

#include <vector>

#include <opencv2/core.hpp>

#if defined(__has_include)
#if __has_include(<opencv2/objdetect/aruco_detector.hpp>)
#define DENSE_ARUCO_OPENCV47 1
#include <opencv2/objdetect/aruco_detector.hpp>
#elif __has_include(<opencv2/aruco.hpp>)
#include <opencv2/aruco.hpp>
#else
#error "OpenCV ArUco not found (need objdetect/aruco_detector.hpp or aruco.hpp)"
#endif
#else
#include <opencv2/aruco.hpp>
#endif

namespace dense_aruco {
namespace cvx {

struct Dictionary {
#if DENSE_ARUCO_OPENCV47
  cv::aruco::Dictionary impl;
  int markerSize() const { return impl.markerSize; }
#else
  cv::Ptr<cv::aruco::Dictionary> impl;
  int markerSize() const { return impl->markerSize; }
#endif
};

inline Dictionary get_predefined(int id) {
  Dictionary d;
  d.impl = cv::aruco::getPredefinedDictionary(id);
  return d;
}

inline void generate_marker(const Dictionary& d, int id, int side, cv::Mat& img) {
#if DENSE_ARUCO_OPENCV47
  cv::aruco::generateImageMarker(d.impl, id, side, img, 1);
#else
  cv::aruco::drawMarker(d.impl, id, side, img, 1);
#endif
}

inline cv::aruco::DetectorParameters to_cv_params(int method, int win, int iters,
                                                 double acc) {
  cv::aruco::DetectorParameters p;
  p.cornerRefinementMethod = method;
  p.cornerRefinementWinSize = win;
  p.cornerRefinementMaxIterations = iters;
  p.cornerRefinementMinAccuracy = acc;
  return p;
}

inline void detect_markers(const cv::Mat& image, const Dictionary& d,
                           const cv::aruco::DetectorParameters& params,
                           std::vector<std::vector<cv::Point2f>>& corners,
                           cv::Mat& ids,
                           std::vector<std::vector<cv::Point2f>>& rejected) {
#if DENSE_ARUCO_OPENCV47
  cv::aruco::ArucoDetector detector(d.impl, params);
  detector.detectMarkers(image, corners, ids, rejected);
#else
  auto pp = cv::makePtr<cv::aruco::DetectorParameters>(params);
  cv::aruco::detectMarkers(image, d.impl, corners, ids, pp, rejected);
#endif
}

}  // namespace cvx
}  // namespace dense_aruco
