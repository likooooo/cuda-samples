#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <filesystem>
#include <cassert>
#include <vector>
namespace py = boost::python;
namespace np = boost::python::numpy;

#define catch_py_error(code) do{try{code;}catch (py::error_already_set) {PyErr_Print();exit(1);}}while(0)

struct py_loader final
{
    py_loader(const char* file = "__main__"){
        py::object main_module = py::import(file);
        this->py_namespace = main_module.attr("__dict__");
        //import_to_current("sys");
        //this->py_env_path = eval<py::object>("sys.path");
        this->py_env_path = py::import("sys").attr("path");
    }
    std::vector<py::object> import_to_current(const std::filesystem::path& path) const {
        std::vector<py::object> vec;
        append_import_to_current(path, vec);
        return vec;
    }

    void append_import_to_current(const std::filesystem::path& path, std::vector<py::object>& vec) const {
        namespace fs = std::filesystem;
        if (fs::is_directory(path)) {
            //TODO : vec.reserve(vec.capacity() + files);
            for (const auto& entry : fs::directory_iterator(path)) {
                this->import_to_current(entry.path());
            }
        } 
        vec.push_back(py::import(path.generic_string().c_str()));
    }
    void exec(const char* cmd) const {
        catch_py_error(py::exec(cmd, py_namespace));
    }
    template<class T = void>
    T eval(const char* code) const {
        if constexpr(std::is_same<T, void>::value) catch_py_error(py::eval(code));
        else if constexpr (std::is_same<T, py::object>::value) catch_py_error(return py::eval(code));
        else catch_py_error(return py::extract<T>(py::eval(code))());
        return T();
    }
    py::list add_env_path(const char* path = PY_SOURCE_DIR) {
        py::list pathList = py::extract<py::list>(this->py_env_path);
        pathList.append(path);
        return pathList;
    }
    py::object py_namespace;
    py::object py_env_path;
private:
    struct py_lifetime{
        bool is_py_running;
        py_lifetime(bool init = false) : is_py_running(init){
            if(!init) return;
            Py_Initialize();
            np::initialize();
        }
        ~py_lifetime(){
            if(!is_py_running) return;
            //np::finalize();
        }
    };
    static py_lifetime& get_py_inter(){
        static py_lifetime py(false); 
        return py;
    };
public:
    static void init(){
        get_py_inter() = py_lifetime(true);
    }
    static void dispose(){
        get_py_inter() = py_lifetime(false);
    }
};