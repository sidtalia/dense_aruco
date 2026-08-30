#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "dense_aruco/opencv_compat.hpp"

namespace dense_aruco {

enum class RefineMode { Off, Outer, Internal, All };
enum class OptimizeMode { H, Pose, Both };

struct DetectorParameters {
  int corner_refinement_method = 3;  // CORNER_REFINE_APRILTAG when available
  int corner_refinement_win_size = 5;
  int corner_refinement_max_iter = 30;
  double corner_refinement_min_accuracy = 0.1;
};

struct PoseEstimate {
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  bool ok = false;
  bool dense_ok = false;
  double initial_error = 0.0;
  double final_error = 0.0;
  int iterations = 0;
};

struct RefineDebug {
  std::vector<cv::Point2f> pts;
  std::vector<uint8_t> ok;
  std::vector<uint8_t> kind;  // 0 outer, 1 internal
  std::vector<cv::Point3f> p0;
  std::vector<cv::Point3f> p1;
  std::vector<uint8_t> edge_kind;
  cv::Vec3d rvec0{};
  cv::Vec3d tvec0{};
  cv::Vec3d rvec1{};
  cv::Vec3d tvec1{};
  cv::Matx33d K = cv::Matx33d::eye();
};

RefineMode parse_refine_mode(const std::string& name);
OptimizeMode parse_optimize_mode(const std::string& name);
int dictionary_id_from_name(const std::string& name);

class ArucoDetector {
 public:
  ArucoDetector(int dictionary_id, const DetectorParameters& params,
                RefineMode refine = RefineMode::Off,
                OptimizeMode optimize = OptimizeMode::Both,
                int samples_per_edge = 6);

  void detectMarkers(const cv::Mat& image,
                     std::vector<std::vector<cv::Point2f>>& corners,
                     std::vector<int>& ids,
                     std::vector<std::vector<cv::Point2f>>& rejected) const;

  std::vector<PoseEstimate> estimatePoseSingleMarkers(
      const std::vector<std::vector<cv::Point2f>>& corners,
      const std::vector<int>& ids, double marker_length,
      const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
      const cv::Mat& image = cv::Mat()) const;

  const RefineDebug& last_debug() const { return debug_; }
  void drawLastRefinement(cv::Mat& bgr) const;

  RefineMode refine_mode() const { return refine_; }

 private:
  cvx::Dictionary dict_;
  DetectorParameters params_{};
  RefineMode refine_;
  OptimizeMode optimize_;
  int samples_per_edge_;
  mutable RefineDebug debug_;
  mutable std::unique_ptr<std::mutex> debug_mu_;
};

}  // namespace dense_aruco
