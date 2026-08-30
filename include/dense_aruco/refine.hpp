#pragma once

#include "dense_aruco/aruco.hpp"

#include <opencv2/core.hpp>

namespace dense_aruco {

struct Boundary {
  cv::Point3d p0{};
  cv::Point3d p1{};
  cv::Point2d n_xy{};
  uint8_t kind = 0;
};

struct Geom {
  int n_cells = 0;
  std::vector<Boundary> bounds;
  std::vector<cv::Point3d> xyz;
  std::vector<cv::Point2d> nxy;
  std::vector<uint8_t> kind;
};

cv::Mat reconstruct_cells(const cvx::Dictionary& dict, int marker_id);
std::vector<Boundary> generate_boundaries(const cv::Mat& cells_white,
                                          double marker_length, bool outer,
                                          bool internal);
Geom build_geom(const cvx::Dictionary& dict, int marker_id,
                double marker_length, RefineMode mode, int samples_per_edge);

struct RefineResult {
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  bool success = false;
  double initial_error = 0.0;
  double final_error = 0.0;
  int iterations = 0;
  RefineDebug debug{};
};

RefineResult refine_pose(const cv::Mat& gray, const Geom& geom,
                         const cv::Mat& K, const cv::Vec3d& rvec0,
                         const cv::Vec3d& tvec0, double marker_length,
                         OptimizeMode optimize, RefineMode mode);

}  // namespace dense_aruco
