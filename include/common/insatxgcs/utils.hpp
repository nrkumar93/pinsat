#pragma once

#include <memory>
#include <math.h>
#include <functional>
#include <Eigen/Core>
#include <vector>
#include <fstream>
#include <iostream>
#include <drake/geometry/optimization/hpolyhedron.h>

using drake::geometry::optimization::HPolyhedron;

namespace utils
{

// dynamic_unique_cast for unique_ptr
  template <typename To, typename From, typename Deleter>
  std::unique_ptr<To, Deleter> DynamicUniqueCast(std::unique_ptr<From, Deleter>&& p) {
    if (To* cast = dynamic_cast<To*>(p.get())) {
      std::unique_ptr<To, Deleter> result(cast, std::move(p.get_deleter()));
      p.release();
      return result;
    }
    // return std::unique_ptr<To, Deleter>(nullptr); // or throw std::bad_cast() if you prefer
    throw std::runtime_error("dynamic_unique_cast failed");
  }

// overload << for std::vector<T>
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      os << v[i];
      if (i < v.size() - 1)
        os << ", ";
    }
    os << "]";
    return os;
  }

  double Brentq(const std::function<double(double)>& func, double xa, double xb,
                double xtol = 1e-8, double rtol = 1e-8, int iter = 1000);

  std::shared_ptr<std::vector<std::pair<int,int>>> DeserializeEdges(const std::string& file_path);

  std::vector<HPolyhedron> DeserializeRegions(const std::string& file_path);

  void SerializeRegions(const std::vector<HPolyhedron>& regions, const std::string& file_path);

} // namespace utils