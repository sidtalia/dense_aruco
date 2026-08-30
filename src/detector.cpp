#include "dense_aruco/aruco.hpp"
#include "dense_aruco/refine.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <thread>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace dense_aruco {
namespace {

std::string lower(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

cv::aruco::DetectorParameters to_cv(const DetectorParameters& p) {
  return cvx::to_cv_params(p.corner_refinement_method, p.corner_refinement_win_size,
                           p.corner_refinement_max_iter,
                           p.corner_refinement_min_accuracy);
}

std::vector<cv::Point3f> object_corners(double L) {
  const float s = static_cast<float>(0.5 * L);
  return {{-s, s, 0.f}, {s, s, 0.f}, {s, -s, 0.f}, {-s, -s, 0.f}};
}

bool solve_marker_pose(const std::vector<cv::Point2f>& corners, double L,
                       const cv::Mat& K, const cv::Mat& dist, cv::Vec3d& rvec,
                       cv::Vec3d& tvec) {
  const auto obj = object_corners(L);
  std::vector<cv::Point2f> img = corners;
  cv::Mat dist0 = cv::Mat::zeros(5, 1, CV_64F);
  cv::Mat und;
  if (!dist.empty() && cv::norm(dist) > 1e-12) {
    cv::undistortPoints(img, und, K, dist, cv::noArray(), K);
    img.clear();
    und.copyTo(img);
  }
  std::vector<cv::Mat> rvecs, tvecs;
  cv::solvePnPGeneric(obj, img, K, dist0, rvecs, tvecs, false,
                      cv::SOLVEPNP_IPPE_SQUARE);
  if (rvecs.empty()) {
    return false;
  }
  double best = 1e18;
  int bi = 0;
  for (size_t i = 0; i < rvecs.size(); ++i) {
    cv::Mat r = rvecs[i].clone();
    cv::Mat t = tvecs[i].clone();
    cv::solvePnPRefineLM(obj, img, K, dist0, r, t);
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, r, t, K, dist0, proj);
    double e = 0.0;
    for (size_t k = 0; k < proj.size(); ++k) {
      e += cv::norm(proj[k] - img[k]);
    }
    e /= static_cast<double>(proj.size());
    if (e < best) {
      best = e;
      bi = static_cast<int>(i);
      rvec = r;
      tvec = t;
    }
  }
  (void)bi;
  return true;
}

}  // namespace

RefineMode parse_refine_mode(const std::string& name) {
  const std::string key = lower(name);
  if (key == "off" || key == "none" || key == "0" || key == "false") {
    return RefineMode::Off;
  }
  if (key == "outer" || key == "outer_boundary") {
    return RefineMode::Outer;
  }
  if (key == "internal" || key == "internal_boundaries") {
    return RefineMode::Internal;
  }
  if (key == "all" || key == "all_boundaries" || key == "both") {
    return RefineMode::All;
  }
  throw std::invalid_argument("dense_refine must be off|outer|internal|all");
}

OptimizeMode parse_optimize_mode(const std::string& name) {
  const std::string key = lower(name);
  if (key == "h" || key == "homography") {
    return OptimizeMode::H;
  }
  if (key == "pose" || key == "6dof") {
    return OptimizeMode::Pose;
  }
  return OptimizeMode::Both;
}

int dictionary_id_from_name(const std::string& name) {
  std::string key = name;
  for (char& c : key) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  if (key.rfind("DICT_", 0) != 0) {
    key = "DICT_" + key;
  }
  static const std::unordered_map<std::string, int> kIds = {
      {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
      {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
      {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
      {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
      {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
      {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
      {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
      {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
      {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
      {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
      {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
      {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
      {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
      {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
      {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
      {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
      {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
      {"DICT_APRILTAG_36H11", cv::aruco::DICT_APRILTAG_36h11},
  };
  auto it = kIds.find(key);
  if (it == kIds.end()) {
    throw std::invalid_argument("unknown ArUco dictionary " + name);
  }
  return it->second;
}

ArucoDetector::ArucoDetector(int dictionary_id, const DetectorParameters& params,
                             RefineMode refine, OptimizeMode optimize,
                             int samples_per_edge)
    : dict_(cvx::get_predefined(dictionary_id)),
      params_(params),
      refine_(refine),
      optimize_(optimize),
      samples_per_edge_(samples_per_edge),
      debug_mu_(std::make_unique<std::mutex>()) {}

void ArucoDetector::detectMarkers(
    const cv::Mat& image, std::vector<std::vector<cv::Point2f>>& corners,
    std::vector<int>& ids,
    std::vector<std::vector<cv::Point2f>>& rejected) const {
  cv::Mat ids_m;
  cvx::detect_markers(image, dict_, to_cv(params_), corners, ids_m, rejected);
  ids.clear();
  if (!ids_m.empty()) {
    ids_m.reshape(1, static_cast<int>(ids_m.total())).copyTo(ids);
  }
}

std::vector<PoseEstimate> ArucoDetector::estimatePoseSingleMarkers(
    const std::vector<std::vector<cv::Point2f>>& corners,
    const std::vector<int>& ids, double marker_length, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const cv::Mat& image) const {
  std::vector<PoseEstimate> out(corners.size());
  cv::Mat K;
  camera_matrix.convertTo(K, CV_64F);
  cv::Mat dist;
  if (dist_coeffs.empty()) {
    dist = cv::Mat::zeros(5, 1, CV_64F);
  } else {
    dist_coeffs.reshape(1, 1).convertTo(dist, CV_64F);
  }
  cv::Mat gray;
  if (!image.empty() && refine_ != RefineMode::Off) {
    if (image.channels() == 3) {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = image;
    }
  }

  auto process_one = [&](size_t i) {
    PoseEstimate& pe = out[i];
    if (corners[i].size() < 4) {
      return;
    }
    if (!solve_marker_pose(corners[i], marker_length, K, dist, pe.rvec, pe.tvec)) {
      return;
    }
    pe.ok = true;
    if (gray.empty() || refine_ == RefineMode::Off) {
      return;
    }
    const int mid = i < ids.size() ? ids[i] : 0;
    const Geom geom =
        build_geom(dict_, mid, marker_length, refine_, samples_per_edge_);
    const RefineResult rr =
        refine_pose(gray, geom, K, pe.rvec, pe.tvec, marker_length, optimize_, refine_);
    pe.initial_error = rr.initial_error;
    pe.final_error = rr.final_error;
    pe.iterations = rr.iterations;
    pe.dense_ok = rr.success;
    if (rr.success) {
      pe.rvec = rr.rvec;
      pe.tvec = rr.tvec;
      std::lock_guard<std::mutex> lock(*debug_mu_);
      debug_ = rr.debug;
    }
  };

  const size_t n = corners.size();
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 2;
  }
  const unsigned nworkers =
      n <= 1 ? 1u : std::min(hw, static_cast<unsigned>(n));
  if (nworkers <= 1) {
    for (size_t i = 0; i < n; ++i) {
      process_one(i);
    }
  } else {
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(nworkers);
    for (unsigned w = 0; w < nworkers; ++w) {
      workers.emplace_back([&]() {
        for (;;) {
          const size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= n) {
            break;
          }
          process_one(i);
        }
      });
    }
    for (auto& th : workers) {
      th.join();
    }
  }
  return out;
}

void ArucoDetector::drawLastRefinement(cv::Mat& bgr) const {
  if (bgr.empty()) {
    return;
  }
  const cv::Mat K(debug_.K);
  auto draw_edge = [&](const cv::Point3f& a, const cv::Point3f& b,
                       const cv::Vec3d& r, const cv::Vec3d& t,
                       const cv::Scalar& color) {
    std::vector<cv::Point3f> obj{a, b};
    std::vector<cv::Point2f> img;
    cv::projectPoints(obj, r, t, K, cv::noArray(), img);
    cv::line(bgr, img[0], img[1], color, 1, cv::LINE_AA);
  };
  for (size_t i = 0; i < debug_.p0.size(); ++i) {
    const cv::Scalar init = debug_.edge_kind[i] ? cv::Scalar(80, 180, 180)
                                                : cv::Scalar(180, 180, 80);
    const cv::Scalar fin = debug_.edge_kind[i] ? cv::Scalar(0, 255, 0)
                                               : cv::Scalar(0, 165, 255);
    draw_edge(debug_.p0[i], debug_.p1[i], debug_.rvec0, debug_.tvec0, init);
    draw_edge(debug_.p0[i], debug_.p1[i], debug_.rvec1, debug_.tvec1, fin);
  }
  for (size_t i = 0; i < debug_.pts.size(); ++i) {
    const cv::Point p(cvRound(debug_.pts[i].x), cvRound(debug_.pts[i].y));
    cv::Scalar col(40, 40, 220);
    if (i < debug_.ok.size() && debug_.ok[i]) {
      col = (i < debug_.kind.size() && debug_.kind[i] == 0)
                ? cv::Scalar(0, 165, 255)
                : cv::Scalar(0, 255, 0);
    }
    cv::circle(bgr, p, 2, col, -1, cv::LINE_AA);
  }
  cv::putText(bgr, "cyan=init  green=internal  orange=outer  red=reject", {8, 44},
              cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 2, cv::LINE_AA);
}

}  // namespace dense_aruco
