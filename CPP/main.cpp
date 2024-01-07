#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // needed for automatic type conversion (e.g., for lists)
#include <pybind11/numpy.h>  // needed for numpy array support (returns and operates directly on numpy arrays)

//#include "SomeClass.h"
#include "Simulator.h"

namespace py = pybind11;


PYBIND11_MODULE(simulator, m) {
    m.doc() = "This is the module docs.";

//    py::class_<SomeClass>(m, "PySomeClass")
//            .def(py::init<float>())
//            .def("multiply", &SomeClass::multiply)
//            .def("multiply_list", &SomeClass::multiply_list)
//            .def("multiply_ndarray", [](SomeClass &self, std::vector<float> &array) {
//                py::array out = py::cast(self.multiply_ndarray(array));
//                return out;
//            });

    py::class_<Simulator>(m, "Simulator")
            .def(py::init<int, int, std::vector<unsigned short> &, std::vector<std::vector<unsigned short>> &>())
            .def("getThreadCount", &Simulator::getThreadCount)
            .def("getMaxSimDepth", &Simulator::getMaxSimDepth)
            .def("getSolutionState", &Simulator::getSolutionState)
            .def("getMove", &Simulator::getMove)
            .def("getMoves", [](Simulator &self) {
                return py::cast(self.getMoves());
            })
            .def("getSimulationResults", &Simulator::getSimulationResults);
}