/***********************************
******  timvx.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <memory>
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11_json.hpp"
#include "timvx_engine.h"
#include "common/timvx_log.h"

namespace py = pybind11;
using namespace TimVX;

bool compareDims(std::vector<uint32_t> left, std::vector<uint32_t> right, bool reverse = false)
{
    if (left.size() != right.size())
        return false;
    int n_dims = left.size();
    for (int i = 0; i < n_dims; i++)
    {
        // np array dims reverse order with timvx tensor dims
        if (true == reverse && left[i] != right[n_dims - 1 - i])
            return false;
        else if (false == reverse && left[i] != right[i])
            return false;
        else
            continue;
    }
    return true;
}

std::string dimsToString(std::vector<uint32_t> dims, bool reverse = false)
{
    std::string dims_str;
    dims_str += "[";
    int n_dims = dims.size();
    for (int i = 0; i < n_dims && i < 10; i++)
    {
        if (reverse)
            dims_str += std::to_string(dims[n_dims - 1 - i]);
        else
            dims_str += std::to_string(dims[i]);
        dims_str += ",";
    }
    dims_str += "]";
    return dims_str;
}

size_t getTensorByteSize(TimVXEngine* timvx_engine, const std::string tensor_name)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return -1;
    }
    return timvx_engine->getTensorByteSize(tensor_name);
}

bool createTensor(TimVXEngine* timvx_engine, const std::string tensor_name, 
    const py::dict& tensor_dict_info, const py::array& data_array)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    std::shared_ptr<char> data_array_ptr;
    int num_bytes = data_array.nbytes();
    if (0 < num_bytes)
    {
        data_array_ptr.reset(new char[num_bytes], [](char* data_array_ptr){delete [] data_array_ptr;});
        if (nullptr == data_array_ptr.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory buffer for tensor {} fail!", tensor_name);
            return false;
        }
        memcpy((void*)data_array_ptr.get(), data_array.data(), num_bytes);
    }
    nl::json tensor_json_info = tensor_dict_info;
    return timvx_engine->createTensor(tensor_name, tensor_json_info, data_array_ptr.get(), num_bytes);
}

bool copyDataFromTensor(TimVXEngine* timvx_engine, const std::string tensor_name, py::array& data_array)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }

    auto tensor = timvx_engine->getTensor(tensor_name);
    if (nullptr == tensor)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
        return false;
    }

    // compare np byte size with tensor byte size
    size_t total_np_size = data_array.nbytes();
    size_t total_tensor_size = timvx_engine->getTensorByteSize(tensor_name);
    if (total_tensor_size != total_np_size)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} not equal to numpy array size:{}", 
            tensor_name, total_tensor_size, total_np_size);
        return false;
    }

    // compare np dims with tensor dims
    std::vector<uint32_t> np_dims(data_array.shape(), data_array.shape() + data_array.ndim());
    std::vector<uint32_t> tensor_dims = tensor->GetShape();
    if (false == compareDims(np_dims, tensor_dims, true))
    {
        auto np_dims_str = dimsToString(np_dims);
        auto tensor_dims_str = dimsToString(tensor_dims, true);
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} dims:{} not equal to numpy array dims:{}", 
            tensor_name, tensor_dims_str, np_dims_str);
        return false;
    }
    char* buffer_data = (char*)data_array.mutable_data();
    int buffer_len = total_np_size;
    return timvx_engine->copyDataFromTensor(tensor_name, buffer_data, buffer_len);
}

bool copyDataToTensor(TimVXEngine* timvx_engine, const std::string tensor_name, const py::array& data_array)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }

    auto tensor = timvx_engine->getTensor(tensor_name);
    if (nullptr == tensor)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
        return false;
    }

    // compare np byte size with tensor byte size
    size_t total_np_size = data_array.nbytes();
    size_t total_tensor_size = timvx_engine->getTensorByteSize(tensor_name);
    if (total_tensor_size != total_np_size)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} not equal to numpy array size:{}", 
            tensor_name, total_tensor_size, total_np_size);
        return false;
    }

    // compare np dims with tensor dims
    std::vector<uint32_t> np_dims(data_array.shape(), data_array.shape() + data_array.ndim());
    std::vector<uint32_t> tensor_dims = tensor->GetShape();
    if (false == compareDims(np_dims, tensor_dims, true))
    {
        auto np_dims_str = dimsToString(np_dims);
        auto tensor_dims_str = dimsToString(tensor_dims, true);
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} dims:{} not equal to numpy array dims:{}", 
            tensor_name, tensor_dims_str, np_dims_str);
        return false;
    }
    const char* buffer_data = (const char*)data_array.data();
    int buffer_len = total_np_size;
    return timvx_engine->copyDataToTensor(tensor_name, buffer_data, buffer_len);
}

bool createOperation(TimVXEngine* timvx_engine, py::dict& op_dict_info)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    nl::json op_json_info = op_dict_info;
    return timvx_engine->createOperation(op_json_info);
}

py::dict getOpInfo(TimVXEngine* timvx_engine, const std::string op_name)
{
    py::dict op_dict;
    if (nullptr == timvx_engine)
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
    else
        op_dict = timvx_engine->getOpInfo(op_name);
    return op_dict;
}

bool bindInputs(TimVXEngine* timvx_engine, const std::string op_name, const std::vector<std::string> input_list)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->bindInputs(op_name, input_list);
}

bool bindOutputs(TimVXEngine* timvx_engine, const std::string op_name, const std::vector<std::string> output_list)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->bindOutputs(op_name, output_list);
}

bool bindInput(TimVXEngine* timvx_engine, const std::string op_name, const std::string input_name)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->bindInput(op_name, input_name);
}

bool bindOutput(TimVXEngine* timvx_engine, const std::string op_name, const std::string output_name)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->bindOutput(op_name, output_name);
}

// graph uitls
bool createGraph(TimVXEngine* timvx_engine)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->createGraph();
}

bool verifyGraph(TimVXEngine* timvx_engine)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->verifyGraph();
}

bool compileGraph(TimVXEngine* timvx_engine)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->compileGraph();
}

bool runGraph(TimVXEngine* timvx_engine)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return false;
    }
    return timvx_engine->runGraph();
}

py::bytearray compileToBinary(TimVXEngine* timvx_engine)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return py::bytearray();
    }
    size_t bin_size = 0;
    std::vector<uint8_t> nbg_buf;
    if (false == timvx_engine->compileToBinary(nbg_buf, bin_size) || 
        0 == bin_size || 0 == nbg_buf.size())
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "compile graph to binary data fail!");
        return py::bytearray();
    }
    return py::bytearray((char*)nbg_buf.data(), bin_size);
}

std::string getGraphName(TimVXEngine* timvx_engine)
{
    if (nullptr == timvx_engine)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input timvx engine parameter is nullptr!");
        return std::string();
    }
    return timvx_engine->getGraphName();
}

void setLogLevel(int log_level)
{
    // spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
    spdlog::default_logger_raw()->set_level(static_cast<spdlog::level::level_enum>(log_level));
    return;
}

PYBIND11_MODULE(pytimvx, m)
{
    m.doc() = "timvx python interface, convert rknn/tflite to timvx model and run model with timvx engine";
    m.def("get_tensor_size",        &getTensorByteSize,           "get graph's tensor size by tensor name in ");
    m.def("create_tensor",          &createTensor,            "create graph's tensor with tensor info");
    m.def("copy_data_from_tensor",  &copyDataFromTensor,      "copy data form graph's tensor");
    m.def("copy_data_to_tensor",    &copyDataToTensor,        "copy data to graph's tensor");
    m.def("create_operation",       &createOperation,         "create graph's operation with op info");
    m.def("get_op_info",            &getOpInfo,               "get graph's op info by op name");
    m.def("bind_inputs",            &bindInputs,              "bind graph's operation inputs");
    m.def("bind_outputs",           &bindOutputs,             "bind graph's operation outputs");
    m.def("bind_input",             &bindInput,               "bind graph's operation input");
    m.def("bind_output",            &bindOutput,              "bind graph's operation output");
    m.def("create_graph",           &createGraph,             "create graph");
    m.def("verify_graph",           &verifyGraph,             "verify graph");
    m.def("compile_graph",          &compileGraph,            "compile graph");
    m.def("run_graph",              &runGraph,                "run graph");
    m.def("get_graph_name",         &getGraphName,            "get graph's name");
    m.def("compile_to_binary",      &compileToBinary,         "compile graph to binary data");
    m.def("set_log_level",          &setLogLevel,             "set log level");

    py::class_<TimVXEngine>(m, "TimVXEngine")
    .def(py::init<const std::string>());
}