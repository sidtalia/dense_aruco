#include "dense_aruco/aruco.hpp"

#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace py = pybind11;

namespace {

cv::Mat numpy_to_mat(const py::array& arr) {
  py::buffer_info info = arr.request();
  if (info.ndim == 2) {
    int type = CV_8U;
    if (info.format == py::format_descriptor<uint8_t>::format()) {
      type = CV_8U;
    } else if (info.format == py::format_descriptor<float>::format()) {
      type = CV_32F;
    } else if (info.format == py::format_descriptor<double>::format()) {
      type = CV_64F;
    } else {
      throw std::runtime_error("unsupported grayscale dtype");
    }
    return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]),
                   type, info.ptr, static_cast<size_t>(info.strides[0]));
  }
  if (info.ndim == 3) {
    const int ch = static_cast<int>(info.shape[2]);
    int type = CV_8UC(ch);
    if (info.format == py::format_descriptor<float>::format()) {
      type = CV_32FC(ch);
    }
    return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]),
                   type, info.ptr, static_cast<size_t>(info.strides[0]));
  }
  throw std::runtime_error("image must be HxW or HxWxC");
}

cv::Mat numpy_to_mat_owned(const py::array& arr) {
  return numpy_to_mat(arr).clone();
}

py::array_t<double> mat33_to_numpy(const cv::Mat& m) {
  cv::Mat d;
  m.convertTo(d, CV_64F);
  auto out = py::array_t<double>({3, 3});
  auto r = out.mutable_unchecked<2>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      r(i, j) = d.at<double>(i, j);
    }
  }
  return out;
}

cv::Mat numpy_to_k(const py::array& arr) {
  auto a = py::array_t<double>::ensure(arr);
  if (a.size() < 9) {
    throw std::runtime_error("cameraMatrix must be 3x3");
  }
  cv::Mat K(3, 3, CV_64F);
  const double* p = a.data();
  for (int i = 0; i < 9; ++i) {
    K.at<double>(i / 3, i % 3) = p[i];
  }
  return K;
}

cv::Mat numpy_to_dist(const py::array& arr) {
  auto a = py::array_t<double>::ensure(arr);
  cv::Mat d(5, 1, CV_64F, cv::Scalar(0));
  const int n = std::min(5, static_cast<int>(a.size()));
  const double* p = a.data();
  for (int i = 0; i < n; ++i) {
    d.at<double>(i) = p[i];
  }
  return d;
}

std::vector<std::vector<cv::Point2f>> parse_corners(const py::object& corners) {
  std::vector<std::vector<cv::Point2f>> out;
  if (corners.is_none()) {
    return out;
  }
  py::sequence seq = corners;
  for (auto item : seq) {
    auto a = py::array_t<float>::ensure(item);
    std::vector<cv::Point2f> c;
    const float* p = a.data();
    const int n = static_cast<int>(a.size() / 2);
    for (int i = 0; i < n; ++i) {
      c.emplace_back(p[2 * i], p[2 * i + 1]);
    }
    out.push_back(std::move(c));
  }
  return out;
}

py::list corners_to_python(const std::vector<std::vector<cv::Point2f>>& corners) {
  py::list out;
  for (const auto& c : corners) {
    auto a = py::array_t<float>({1, static_cast<int>(c.size()), 2});
    auto r = a.mutable_unchecked<3>();
    for (size_t i = 0; i < c.size(); ++i) {
      r(0, static_cast<ssize_t>(i), 0) = c[i].x;
      r(0, static_cast<ssize_t>(i), 1) = c[i].y;
    }
    out.append(a);
  }
  return out;
}

py::object ids_to_python(const std::vector<int>& ids) {
  if (ids.empty()) {
    return py::none();
  }
  auto a = py::array_t<int>({static_cast<int>(ids.size()), 1});
  auto r = a.mutable_unchecked<2>();
  for (size_t i = 0; i < ids.size(); ++i) {
    r(static_cast<ssize_t>(i), 0) = ids[i];
  }
  return a;
}

int dict_from_obj(const py::object& obj) {
  if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int>();
  }
  if (py::isinstance<py::str>(obj)) {
    return dense_aruco::dictionary_id_from_name(obj.cast<std::string>());
  }
  if (py::hasattr(obj, "bytesList")) {
    throw std::runtime_error(
        "pass cv2.aruco.DICT_* (int) or dictionary name string, not a Dictionary object");
  }
  throw std::runtime_error("dictionary must be int or name string");
}

dense_aruco::DetectorParameters params_from_obj(const py::object& obj) {
  dense_aruco::DetectorParameters p;
  if (obj.is_none()) {
    return p;
  }
  auto get_int = [&](const char* a, const char* b, int def) {
    if (py::hasattr(obj, a)) {
      return obj.attr(a).cast<int>();
    }
    if (py::hasattr(obj, b)) {
      return obj.attr(b).cast<int>();
    }
    return def;
  };
  auto get_dbl = [&](const char* a, const char* b, double def) {
    if (py::hasattr(obj, a)) {
      return obj.attr(a).cast<double>();
    }
    if (py::hasattr(obj, b)) {
      return obj.attr(b).cast<double>();
    }
    return def;
  };
  p.corner_refinement_method =
      get_int("cornerRefinementMethod", "corner_refinement_method", p.corner_refinement_method);
  p.corner_refinement_win_size =
      get_int("cornerRefinementWinSize", "corner_refinement_win_size", p.corner_refinement_win_size);
  p.corner_refinement_max_iter = get_int("cornerRefinementMaxIterations",
                                         "corner_refinement_max_iter", p.corner_refinement_max_iter);
  p.corner_refinement_min_accuracy = get_dbl(
      "cornerRefinementMinAccuracy", "corner_refinement_min_accuracy",
      p.corner_refinement_min_accuracy);
  return p;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ ArUco detector + dense internal-bit pose refinement";

  py::class_<dense_aruco::DetectorParameters>(m, "DetectorParameters")
      .def(py::init<>())
      .def_readwrite("cornerRefinementMethod",
                     &dense_aruco::DetectorParameters::corner_refinement_method)
      .def_readwrite("cornerRefinementWinSize",
                     &dense_aruco::DetectorParameters::corner_refinement_win_size)
      .def_readwrite("cornerRefinementMaxIterations",
                     &dense_aruco::DetectorParameters::corner_refinement_max_iter)
      .def_readwrite("cornerRefinementMinAccuracy",
                     &dense_aruco::DetectorParameters::corner_refinement_min_accuracy);

  py::class_<dense_aruco::ArucoDetector>(m, "ArucoDetector")
      .def(py::init([](const py::object& dictionary, const py::object& params,
                       const std::string& dense_refine, const std::string& dense_optimize,
                       int samples_per_edge) {
             return dense_aruco::ArucoDetector(
                 dict_from_obj(dictionary), params_from_obj(params),
                 dense_aruco::parse_refine_mode(dense_refine),
                 dense_aruco::parse_optimize_mode(dense_optimize), samples_per_edge);
           }),
           py::arg("dictionary"), py::arg("detectorParams") = py::none(),
           py::arg("denseRefine") = "off", py::arg("denseOptimize") = "both",
           py::arg("samplesPerEdge") = 6)
      .def(
          "detectMarkers",
          [](const dense_aruco::ArucoDetector& self, const py::array& image) {
            const cv::Mat img = numpy_to_mat(image);
            std::vector<std::vector<cv::Point2f>> corners, rejected;
            std::vector<int> ids;
            self.detectMarkers(img, corners, ids, rejected);
            return py::make_tuple(corners_to_python(corners), ids_to_python(ids),
                                  corners_to_python(rejected));
          },
          py::arg("image"))
      .def(
          "estimatePoseSingleMarkers",
          [](dense_aruco::ArucoDetector& self, const py::object& corners,
             double markerLength, const py::array& cameraMatrix,
             const py::array& distCoeffs, const py::object& ids,
             const py::object& image) {
            const auto cs = parse_corners(corners);
            std::vector<int> idv;
            if (!ids.is_none()) {
              auto a = py::array_t<int>::ensure(ids);
              idv.assign(a.data(), a.data() + a.size());
            }
            cv::Mat img;
            if (!image.is_none()) {
              img = numpy_to_mat_owned(py::array::ensure(image));
            }
            const auto est = self.estimatePoseSingleMarkers(
                cs, idv, markerLength, numpy_to_k(cameraMatrix),
                numpy_to_dist(distCoeffs), img);
            const int n = static_cast<int>(est.size());
            auto rvecs = py::array_t<double>({n, 1, 3});
            auto tvecs = py::array_t<double>({n, 1, 3});
            auto rr = rvecs.mutable_unchecked<3>();
            auto tt = tvecs.mutable_unchecked<3>();
            py::list ok;
            for (int i = 0; i < n; ++i) {
              rr(i, 0, 0) = est[i].rvec[0];
              rr(i, 0, 1) = est[i].rvec[1];
              rr(i, 0, 2) = est[i].rvec[2];
              tt(i, 0, 0) = est[i].tvec[0];
              tt(i, 0, 1) = est[i].tvec[1];
              tt(i, 0, 2) = est[i].tvec[2];
              ok.append(est[i].ok);
            }
            return py::make_tuple(rvecs, tvecs, ok);
          },
          py::arg("corners"), py::arg("markerLength"), py::arg("cameraMatrix"),
          py::arg("distCoeffs"), py::arg("ids") = py::none(),
          py::arg("image") = py::none())
      .def(
          "drawLastRefinement",
          [](const dense_aruco::ArucoDetector& self, py::array image) {
            cv::Mat img = numpy_to_mat(image);
            self.drawLastRefinement(img);
          },
          py::arg("image"));

  m.def("getPredefinedDictionary", &dense_aruco::dictionary_id_from_name);
}
