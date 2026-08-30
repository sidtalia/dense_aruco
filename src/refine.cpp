#include "dense_aruco/refine.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace dense_aruco {
namespace {

constexpr double kProfileOff[] = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
constexpr int kProfileN = 7;
constexpr double kHuberDelta = 1.25;

inline double huber(double r) {
  const double a = std::abs(r);
  if (a <= kHuberDelta) {
    return r;
  }
  return std::copysign(std::sqrt(2.0 * kHuberDelta * a - kHuberDelta * kHuberDelta), r);
}

cv::Point2d project_one(const cv::Point3d& p, const cv::Vec3d& rvec,
                        const cv::Vec3d& tvec, const cv::Mat& K) {
  std::vector<cv::Point3d> obj{p};
  std::vector<cv::Point2d> img;
  cv::projectPoints(obj, rvec, tvec, K, cv::noArray(), img);
  return img[0];
}

void project_all(const std::vector<cv::Point3d>& xyz, const cv::Vec3d& rvec,
                 const cv::Vec3d& tvec, const cv::Mat& K,
                 std::vector<cv::Point2d>& uv) {
  if (xyz.empty()) {
    uv.clear();
    return;
  }
  std::vector<cv::Point2d> img;
  cv::projectPoints(xyz, rvec, tvec, K, cv::noArray(), img);
  uv.assign(img.begin(), img.end());
}

cv::Point2d apply_H(const cv::Matx33d& H, const cv::Point2d& xy) {
  const cv::Vec3d s = H * cv::Vec3d(xy.x, xy.y, 1.0);
  const double w = std::max(std::abs(s[2]), 1e-12);
  return {s[0] / w, s[1] / w};
}

float bilinear(const cv::Mat& gray, double x, double y) {
  const int w = gray.cols;
  const int h = gray.rows;
  const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, w - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, h - 1);
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const double ax = x - std::floor(x);
  const double ay = y - std::floor(y);
  const auto at = [&](int yy, int xx) {
    return static_cast<float>(gray.at<float>(yy, xx));
  };
  return static_cast<float>((1.0 - ay) * ((1.0 - ax) * at(y0, x0) + ax * at(y0, x1)) +
                            ay * ((1.0 - ax) * at(y1, x0) + ax * at(y1, x1)));
}

void sample_residuals(const cv::Mat& gray, const std::vector<cv::Point2d>& uv,
                      const std::vector<cv::Point2d>& n_img, double margin,
                      std::vector<double>& y, std::vector<int>& keep) {
  const int n = static_cast<int>(uv.size());
  y.assign(n, 0.0);
  keep.clear();
  const int w = gray.cols;
  const int h = gray.rows;
  double prof[kProfileN];
  for (int i = 0; i < n; ++i) {
    const double ln = std::hypot(n_img[i].x, n_img[i].y) + 1e-12;
    const double nx = n_img[i].x / ln;
    const double ny = n_img[i].y / ln;
    bool inside = std::isfinite(uv[i].x) && std::isfinite(uv[i].y);
    double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
    for (int k = 0; k < kProfileN; ++k) {
      const double xs = uv[i].x + kProfileOff[k] * nx;
      const double ys = uv[i].y + kProfileOff[k] * ny;
      xmin = std::min(xmin, xs);
      xmax = std::max(xmax, xs);
      ymin = std::min(ymin, ys);
      ymax = std::max(ymax, ys);
      if (inside) {
        prof[k] = bilinear(gray, xs, ys);
      }
    }
    if (!inside || xmin < margin || ymin < margin || xmax > w - 1 - margin ||
        ymax > h - 1 - margin) {
      continue;
    }
    double sorted[kProfileN];
    std::copy(prof, prof + kProfileN, sorted);
    std::sort(sorted, sorted + kProfileN);
    const double lo = sorted[1];
    const double hi = sorted[5];
    const double span = hi - lo;
    if (span < 8.0) {
      continue;
    }
    double rise[kProfileN - 1];
    double strength = 0.0;
    for (int k = 0; k < kProfileN - 1; ++k) {
      const double a = (prof[k] - lo) / span;
      const double b = (prof[k + 1] - lo) / span;
      rise[k] = std::max(b - a, 0.0);
      strength += rise[k];
    }
    if (strength < 0.35) {
      continue;
    }
    double resid = 0.0;
    for (int k = 0; k < kProfileN - 1; ++k) {
      const double mid = 0.5 * (kProfileOff[k] + kProfileOff[k + 1]);
      resid += rise[k] * mid;
    }
    resid /= strength;
    y[i] = huber(resid);
    keep.push_back(i);
  }
}

void image_normals(const std::vector<cv::Point3d>& xyz,
                   const std::vector<cv::Point2d>& nxy, const cv::Vec3d& rvec,
                   const cv::Vec3d& tvec, const cv::Mat& K, double eps,
                   std::vector<cv::Point2d>& uv, std::vector<cv::Point2d>& dn) {
  project_all(xyz, rvec, tvec, K, uv);
  std::vector<cv::Point3d> off = xyz;
  for (size_t i = 0; i < off.size(); ++i) {
    off[i].x += eps * nxy[i].x;
    off[i].y += eps * nxy[i].y;
  }
  std::vector<cv::Point2d> uv2;
  project_all(off, rvec, tvec, K, uv2);
  dn.resize(uv.size());
  for (size_t i = 0; i < uv.size(); ++i) {
    dn[i] = uv2[i] - uv[i];
  }
}

void image_normals_H(const std::vector<cv::Point3d>& xyz,
                     const std::vector<cv::Point2d>& nxy, const cv::Matx33d& H,
                     double eps, std::vector<cv::Point2d>& uv,
                     std::vector<cv::Point2d>& dn) {
  uv.resize(xyz.size());
  dn.resize(xyz.size());
  for (size_t i = 0; i < xyz.size(); ++i) {
    uv[i] = apply_H(H, {xyz[i].x, xyz[i].y});
    const cv::Point2d uv2 =
        apply_H(H, {xyz[i].x + eps * nxy[i].x, xyz[i].y + eps * nxy[i].y});
    dn[i] = uv2 - uv[i];
  }
}

void residuals_pose(const cv::Mat& gray, const Geom& geom, const cv::Mat& K,
                    const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    double marker_length, std::vector<double>& y,
                    std::vector<int>& keep) {
  const double eps = 0.02 * marker_length / std::max(geom.n_cells, 1);
  std::vector<cv::Point2d> uv, dn;
  image_normals(geom.xyz, geom.nxy, rvec, tvec, K, eps, uv, dn);
  sample_residuals(gray, uv, dn, 3.0, y, keep);
}

void residuals_H(const cv::Mat& gray, const Geom& geom, const cv::Matx33d& H,
                 double marker_length, std::vector<double>& y,
                 std::vector<int>& keep) {
  const double eps = 0.02 * marker_length / std::max(geom.n_cells, 1);
  std::vector<cv::Point2d> uv, dn;
  image_normals_H(geom.xyz, geom.nxy, H, eps, uv, dn);
  sample_residuals(gray, uv, dn, 3.0, y, keep);
}

cv::Matx33d unpack_H(const cv::Mat& p) {
  cv::Matx33d H = cv::Matx33d::eye();
  for (int i = 0; i < 8; ++i) {
    H.val[i] = p.at<double>(i);
  }
  return H;
}

cv::Mat pack_H(const cv::Matx33d& Hin) {
  cv::Matx33d H = Hin;
  const double s = std::abs(H(2, 2)) > 1e-12 ? H(2, 2) : 1.0;
  H = H * (1.0 / s);
  cv::Mat p(8, 1, CV_64F);
  for (int i = 0; i < 8; ++i) {
    p.at<double>(i) = H.val[i];
  }
  return p;
}

bool H_from_pose(const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Mat& K,
                 double L, cv::Matx33d& H) {
  const double half = 0.5 * L;
  std::vector<cv::Point2d> obj = {{-half, half}, {half, half}, {half, -half}, {-half, -half}};
  std::vector<cv::Point3d> xyz;
  for (const auto& p : obj) {
    xyz.emplace_back(p.x, p.y, 0.0);
  }
  std::vector<cv::Point2d> img;
  project_all(xyz, rvec, tvec, K, img);
  const cv::Mat Hh = cv::findHomography(obj, img, 0);
  if (Hh.empty()) {
    return false;
  }
  H = Hh;
  return true;
}

bool pose_from_H(const cv::Matx33d& H, const cv::Mat& K, double L,
                 const cv::Vec3d& r0, const cv::Vec3d& t0, cv::Vec3d& rvec,
                 cv::Vec3d& tvec) {
  const double half = 0.5 * L;
  std::vector<cv::Point3f> obj = {
      {-static_cast<float>(half), static_cast<float>(half), 0.f},
      {static_cast<float>(half), static_cast<float>(half), 0.f},
      {static_cast<float>(half), -static_cast<float>(half), 0.f},
      {-static_cast<float>(half), -static_cast<float>(half), 0.f},
  };
  std::vector<cv::Point2f> img;
  for (const auto& p : obj) {
    const auto q = apply_H(H, {p.x, p.y});
    img.emplace_back(static_cast<float>(q.x), static_cast<float>(q.y));
  }
  cv::Mat r = cv::Mat(r0).clone();
  cv::Mat t = cv::Mat(t0).clone();
  if (!cv::solvePnP(obj, img, K, cv::Mat::zeros(5, 1, CV_64F), r, t, true,
                    cv::SOLVEPNP_ITERATIVE)) {
    return false;
  }
  rvec = r;
  tvec = t;
  return true;
}

cv::Mat scale_K(const cv::Mat& K, double s) {
  cv::Mat out = K.clone();
  out.at<double>(0, 0) *= s;
  out.at<double>(1, 1) *= s;
  out.at<double>(0, 2) *= s;
  out.at<double>(1, 2) *= s;
  return out;
}

using ResidualFn = std::function<void(const cv::Mat&, std::vector<double>&)>;

void jacobian_fd(const ResidualFn& fn, const cv::Mat& x, const std::vector<double>& y0,
                 cv::Mat& J) {
  const int m = static_cast<int>(y0.size());
  const int n = x.rows;
  J.create(m, n, CV_64F);
  J.setTo(0);
  std::vector<double> yi;
  for (int i = 0; i < n; ++i) {
    cv::Mat xp = x.clone();
    const double step = 1e-5 * std::max(std::abs(x.at<double>(i)), 1.0);
    xp.at<double>(i) += step;
    fn(xp, yi);
    const int lim = std::min(m, static_cast<int>(yi.size()));
    for (int r = 0; r < lim; ++r) {
      J.at<double>(r, i) = (yi[r] - y0[r]) / step;
    }
  }
}

void lm(const ResidualFn& fn, cv::Mat& x, int max_iter, int& iters, double& best) {
  std::vector<double> y;
  fn(x, y);
  best = 0.0;
  for (double v : y) {
    best += 0.5 * v * v;
  }
  double lam = 1e-2;
  iters = 0;
  const int n = x.rows;
  for (int it = 0; it < max_iter; ++it) {
    ++iters;
    if (static_cast<int>(y.size()) < n) {
      break;
    }
    cv::Mat J;
    jacobian_fd(fn, x, y, J);
    cv::Mat A = J.t() * J + lam * cv::Mat::eye(n, n, CV_64F);
    cv::Mat ymat(static_cast<int>(y.size()), 1, CV_64F);
    for (size_t i = 0; i < y.size(); ++i) {
      ymat.at<double>(static_cast<int>(i)) = y[i];
    }
    cv::Mat g = J.t() * ymat;
    cv::Mat dx;
    if (!cv::solve(A, -g, dx, cv::DECOMP_LU)) {
      lam = std::min(lam * 4.0, 1e8);
      continue;
    }
    cv::Mat xtry = x + dx;
    std::vector<double> ytry;
    fn(xtry, ytry);
    if (ytry.size() != y.size()) {
      lam = std::min(lam * 4.0, 1e8);
      continue;
    }
    double cost = 0.0;
    for (double v : ytry) {
      cost += 0.5 * v * v;
    }
    if (cost < best) {
      x = xtry;
      y.swap(ytry);
      best = cost;
      lam = std::max(lam / 3.0, 1e-8);
      if (cv::norm(dx) < 1e-7) {
        break;
      }
    } else {
      lam = std::min(lam * 4.0, 1e8);
    }
  }
}

double mean_sq_kept(const std::vector<double>& y, const std::vector<int>& keep) {
  if (keep.empty()) {
    return 1e6;
  }
  double s = 0.0;
  for (int i : keep) {
    s += y[i] * y[i];
  }
  return s / static_cast<double>(keep.size());
}

}  // namespace

cv::Mat reconstruct_cells(const cvx::Dictionary& dict, int marker_id) {
  const int n = dict.markerSize() + 2;
  cv::Mat img;
  cvx::generate_marker(dict, marker_id, n, img);
  cv::Mat white;
  cv::compare(img, 127, white, cv::CMP_GT);
  return white;
}

std::vector<Boundary> generate_boundaries(const cv::Mat& cells_white,
                                          double marker_length, bool outer,
                                          bool internal) {
  const int n = cells_white.rows;
  const double L = marker_length;
  const double cs = L / n;
  auto corner = [&](double col, double row) {
    return cv::Point3d(-0.5 * L + col * cs, 0.5 * L - row * cs, 0.0);
  };
  auto white = [&](int r, int c) {
    return cells_white.at<uint8_t>(r, c) != 0;
  };
  std::vector<Boundary> out;
  if (internal) {
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n - 1; ++c) {
        const bool a = white(r, c);
        const bool b = white(r, c + 1);
        if (a == b) {
          continue;
        }
        out.push_back({corner(c + 1, r + 1), corner(c + 1, r),
                       {b ? 1.0 : -1.0, 0.0}, 1});
      }
    }
    for (int r = 0; r < n - 1; ++r) {
      for (int c = 0; c < n; ++c) {
        const bool top = white(r, c);
        const bool bot = white(r + 1, c);
        if (top == bot) {
          continue;
        }
        out.push_back({corner(c, r + 1), corner(c + 1, r + 1),
                       {0.0, top ? 1.0 : -1.0}, 1});
      }
    }
  }
  if (outer) {
    const double half = 0.5 * L;
    out.push_back({{-half, half, 0}, {half, half, 0}, {0.0, 1.0}, 0});
    out.push_back({{half, half, 0}, {half, -half, 0}, {1.0, 0.0}, 0});
    out.push_back({{half, -half, 0}, {-half, -half, 0}, {0.0, -1.0}, 0});
    out.push_back({{-half, -half, 0}, {-half, half, 0}, {-1.0, 0.0}, 0});
  }
  return out;
}

Geom build_geom(const cvx::Dictionary& dict, int marker_id,
                double marker_length, RefineMode mode, int samples_per_edge) {
  Geom g;
  const cv::Mat cells = reconstruct_cells(dict, marker_id);
  g.n_cells = cells.rows;
  const bool outer = mode != RefineMode::Internal;
  const bool internal = mode != RefineMode::Outer;
  g.bounds = generate_boundaries(cells, marker_length, outer, internal);
  const int n_per = std::max(samples_per_edge, 3);
  for (const auto& b : g.bounds) {
    for (int i = 0; i < n_per; ++i) {
      const double t = 0.12 + (0.76 * i) / std::max(n_per - 1, 1);
      g.xyz.push_back((1.0 - t) * b.p0 + t * b.p1);
      g.nxy.push_back(b.n_xy);
      g.kind.push_back(b.kind);
    }
  }
  return g;
}

RefineResult refine_pose(const cv::Mat& gray_in, const Geom& geom, const cv::Mat& K0,
                         const cv::Vec3d& rvec0, const cv::Vec3d& tvec0,
                         double marker_length, OptimizeMode optimize,
                         RefineMode mode) {
  RefineResult out;
  out.rvec = rvec0;
  out.tvec = tvec0;
  cv::Mat gray;
  if (gray_in.type() != CV_32F) {
    gray_in.convertTo(gray, CV_32F);
  } else {
    gray = gray_in;
  }
  const double half = 0.5 * marker_length;
  const std::vector<cv::Point3d> corners = {
      {-half, half, 0}, {half, half, 0}, {half, -half, 0}, {-half, -half, 0}};
  std::vector<cv::Point2d> uv_c;
  project_all(corners, rvec0, tvec0, K0, uv_c);
  for (const auto& p : uv_c) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
        p.x > gray.cols - 1 || p.y > gray.rows - 1) {
      return out;
    }
  }
  const double side_px =
      std::min(cv::norm(uv_c[0] - uv_c[1]), cv::norm(uv_c[1] - uv_c[2]));
  if (mode != RefineMode::Outer && side_px / std::max(geom.n_cells, 1) < 3.0) {
    return out;
  }

  const int n_pyr = side_px > 90.0 ? 1 : 0;
  std::vector<cv::Mat> pyr(n_pyr + 1);
  pyr[0] = gray;
  for (int i = 0; i < n_pyr; ++i) {
    cv::Mat down;
    cv::pyrDown(pyr[i], down);
    if (down.type() != CV_32F) {
      down.convertTo(pyr[i + 1], CV_32F);
    } else {
      pyr[i + 1] = down;
    }
  }

  std::vector<double> y0;
  std::vector<int> keep0;
  residuals_pose(pyr[0], geom, K0, rvec0, tvec0, marker_length, y0, keep0);
  out.initial_error = mean_sq_kept(y0, keep0);

  cv::Vec3d rvec = rvec0;
  cv::Vec3d tvec = tvec0;
  int iters = 0;
  try {
    for (int level = n_pyr; level >= 0; --level) {
      const cv::Mat& g = pyr[level];
      const cv::Mat K = scale_K(K0, std::pow(0.5, level));
      const bool do_h = (optimize == OptimizeMode::H || optimize == OptimizeMode::Both) &&
                        level == n_pyr;
      const bool do_pose =
          (optimize == OptimizeMode::Pose || optimize == OptimizeMode::Both) &&
          (level == 0 || optimize == OptimizeMode::Pose);
      if (do_h) {
        cv::Matx33d H;
        if (H_from_pose(rvec, tvec, K, marker_length, H)) {
          cv::Mat p = pack_H(H);
          auto rh = [&](const cv::Mat& pp, std::vector<double>& yy) {
            std::vector<int> kk;
            residuals_H(g, geom, unpack_H(pp), marker_length, yy, kk);
          };
          double dummy = 0;
          int it = 0;
          lm(rh, p, 5, it, dummy);
          iters += it;
          cv::Vec3d rn, tn;
          if (pose_from_H(unpack_H(p), K, marker_length, rvec, tvec, rn, tn)) {
            rvec = rn;
            tvec = tn;
          }
        }
      }
      if (do_pose) {
        cv::Mat p(6, 1, CV_64F);
        p.at<double>(0) = rvec[0];
        p.at<double>(1) = rvec[1];
        p.at<double>(2) = rvec[2];
        p.at<double>(3) = tvec[0];
        p.at<double>(4) = tvec[1];
        p.at<double>(5) = tvec[2];
        auto rp = [&](const cv::Mat& pp, std::vector<double>& yy) {
          std::vector<int> kk;
          residuals_pose(g, geom, K,
                         {pp.at<double>(0), pp.at<double>(1), pp.at<double>(2)},
                         {pp.at<double>(3), pp.at<double>(4), pp.at<double>(5)},
                         marker_length, yy, kk);
        };
        double dummy = 0;
        int it = 0;
        lm(rp, p, 5, it, dummy);
        iters += it;
        rvec = {p.at<double>(0), p.at<double>(1), p.at<double>(2)};
        tvec = {p.at<double>(3), p.at<double>(4), p.at<double>(5)};
      }
    }
  } catch (...) {
    return out;
  }

  std::vector<double> y1;
  std::vector<int> keep1;
  residuals_pose(pyr[0], geom, K0, rvec, tvec, marker_length, y1, keep1);
  out.final_error = mean_sq_kept(y1, keep1);
  out.iterations = iters;
  out.success = keep1.size() >= 8 && out.final_error <= out.initial_error * 1.25;
  if (out.success) {
    out.rvec = rvec;
    out.tvec = tvec;
  }

  const double eps = 0.02 * marker_length / std::max(geom.n_cells, 1);
  std::vector<cv::Point2d> uv, dn;
  image_normals(geom.xyz, geom.nxy, out.rvec, out.tvec, K0, eps, uv, dn);
  out.debug.rvec0 = rvec0;
  out.debug.tvec0 = tvec0;
  out.debug.rvec1 = out.rvec;
  out.debug.tvec1 = out.tvec;
  {
    cv::Mat Kd;
    K0.convertTo(Kd, CV_64F);
    out.debug.K = Kd;
  }
  out.debug.pts.resize(uv.size());
  out.debug.ok.assign(uv.size(), 0);
  out.debug.kind = geom.kind;
  for (size_t i = 0; i < uv.size(); ++i) {
    out.debug.pts[i] = uv[i];
  }
  for (int i : (out.success ? keep1 : keep0)) {
    if (i >= 0 && i < static_cast<int>(out.debug.ok.size())) {
      out.debug.ok[i] = 1;
    }
  }
  out.debug.p0.reserve(geom.bounds.size());
  out.debug.p1.reserve(geom.bounds.size());
  out.debug.edge_kind.reserve(geom.bounds.size());
  for (const auto& b : geom.bounds) {
    out.debug.p0.push_back(b.p0);
    out.debug.p1.push_back(b.p1);
    out.debug.edge_kind.push_back(b.kind);
  }
  return out;
}

}  // namespace dense_aruco
