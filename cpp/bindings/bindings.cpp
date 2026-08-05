// pybind11 bridge: exposes zane::ZaneAnalytics to Python as the `zane_cpp`
// extension module. Blocking C++ calls release the GIL so Python's asyncio
// event loop keeps making progress while the worker pool computes.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "zane/analytics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(zane_cpp, m) {
    m.doc() = "Zane's C++ analytical core: thread-safe probability matrix engine "
              "and data-stream formatter, bridged into Python via pybind11.";

    py::class_<zane::ProbabilityFactors>(m, "ProbabilityFactors")
        .def(py::init<>())
        .def(py::init([](double danger_level, double team_synergy,
                          double resource_availability, double historical_success_rate,
                          double complexity_index) {
                 zane::ProbabilityFactors f;
                 f.danger_level = danger_level;
                 f.team_synergy = team_synergy;
                 f.resource_availability = resource_availability;
                 f.historical_success_rate = historical_success_rate;
                 f.complexity_index = complexity_index;
                 return f;
             }),
             py::arg("danger_level") = 0.0, py::arg("team_synergy") = 0.0,
             py::arg("resource_availability") = 0.0,
             py::arg("historical_success_rate") = 0.0, py::arg("complexity_index") = 0.0)
        .def_readwrite("danger_level", &zane::ProbabilityFactors::danger_level)
        .def_readwrite("team_synergy", &zane::ProbabilityFactors::team_synergy)
        .def_readwrite("resource_availability", &zane::ProbabilityFactors::resource_availability)
        .def_readwrite("historical_success_rate",
                        &zane::ProbabilityFactors::historical_success_rate)
        .def_readwrite("complexity_index", &zane::ProbabilityFactors::complexity_index)
        .def("__repr__", [](const zane::ProbabilityFactors& f) {
            return "<ProbabilityFactors danger=" + std::to_string(f.danger_level) +
                   " synergy=" + std::to_string(f.team_synergy) +
                   " resources=" + std::to_string(f.resource_availability) +
                   " history=" + std::to_string(f.historical_success_rate) +
                   " complexity=" + std::to_string(f.complexity_index) + ">";
        });

    py::class_<zane::ProbabilityResult>(m, "ProbabilityResult")
        .def(py::init<>())
        .def_readonly("computation_id", &zane::ProbabilityResult::computation_id)
        .def_readonly("success_probability", &zane::ProbabilityResult::success_probability)
        .def_readonly("risk_index", &zane::ProbabilityResult::risk_index)
        .def_readonly("confidence_interval_low",
                       &zane::ProbabilityResult::confidence_interval_low)
        .def_readonly("confidence_interval_high",
                       &zane::ProbabilityResult::confidence_interval_high)
        .def_readonly("analytical_summary", &zane::ProbabilityResult::analytical_summary)
        .def("__repr__", [](const zane::ProbabilityResult& r) {
            return "<ProbabilityResult success=" + std::to_string(r.success_probability) +
                   "% risk=" + std::to_string(r.risk_index) + "%>";
        });

    py::class_<zane::DataStreamFrame>(m, "DataStreamFrame")
        .def(py::init<>())
        .def(py::init([](std::string label, std::vector<double> values, uint64_t timestamp_ns) {
                 zane::DataStreamFrame frame;
                 frame.label = std::move(label);
                 frame.values = std::move(values);
                 frame.timestamp_ns = timestamp_ns;
                 return frame;
             }),
             py::arg("label"), py::arg("values"), py::arg("timestamp_ns") = 0)
        .def_readwrite("label", &zane::DataStreamFrame::label)
        .def_readwrite("values", &zane::DataStreamFrame::values)
        .def_readwrite("timestamp_ns", &zane::DataStreamFrame::timestamp_ns);

    py::class_<zane::ZaneAnalytics>(m, "ZaneAnalytics")
        .def(py::init<unsigned int, uint64_t>(), py::arg("worker_threads") = 2,
             py::arg("seed") = 0)
        .def(
            "calculate_success_probability",
            [](zane::ZaneAnalytics& self, const zane::ProbabilityFactors& factors) {
                py::gil_scoped_release release;
                return self.calculate_success_probability(factors);
            },
            py::arg("factors"),
            "Synchronous, blocking probability calculation. Releases the GIL "
            "while the C++ side computes.")
        .def("submit_calculation", &zane::ZaneAnalytics::submit_calculation, py::arg("factors"))
        .def(
            "try_get_result",
            [](zane::ZaneAnalytics& self, uint64_t computation_id)
                -> py::object {
                zane::ProbabilityResult result;
                bool ok;
                {
                    py::gil_scoped_release release;
                    ok = self.try_get_result(computation_id, result);
                }
                if (!ok) {
                    return py::none();
                }
                return py::cast(result);
            },
            py::arg("computation_id"),
            "Returns the ProbabilityResult if ready, else None (non-blocking).")
        .def(
            "get_result_blocking",
            [](zane::ZaneAnalytics& self, uint64_t computation_id, int timeout_ms) {
                py::gil_scoped_release release;
                return self.get_result_blocking(computation_id, timeout_ms);
            },
            py::arg("computation_id"), py::arg("timeout_ms") = 5000)
        .def("format_data_stream", &zane::ZaneAnalytics::format_data_stream, py::arg("frame"))
        .def("get_internal_state_snapshot", &zane::ZaneAnalytics::get_internal_state_snapshot)
        .def("update_internal_state", &zane::ZaneAnalytics::update_internal_state,
             py::arg("key"), py::arg("value"))
        .def("shutdown", &zane::ZaneAnalytics::shutdown);
}
