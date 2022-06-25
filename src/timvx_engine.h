/***********************************
******  timvx_engine.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#pragma once
#include <map>
#include <vector>
#include <string>
#include "engine_common.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/operation.h"
#include "nlohmann/json.hpp"
using namespace tim::vx;
using namespace nlohmann;

namespace TIMVX
{

    class TimVXEngine 
    {
    public:
        TimVXEngine(const std::string &graph_name);
        ~TimVXEngine();

        // tensor utils
        size_t getTensorSize(const std::string &tensor_name);
        bool createTensor(const std::string &tensor_name, const json &tensor_info, 
            const char *weight_data = nullptr, const int weight_len = 0);
        bool getTensorInfo(const std::string &tensor_name, TimvxTensorAttr& tensor_spec);
        bool copyDataFromTensor(const std::string &tensor_name, char* data, const int data_len);
        bool copyDataToTensor(const std::string &tensor_name, const char* data, const int data_len);

        // operation utils
        bool createOperation(const json &op_info);
        bool bindInputs(const std::string &op_name, const std::vector<std::string> &input_list);
        bool bindOutputs(const std::string &op_name, const std::vector<std::string> &output_list);
        bool bindInput(const std::string &op_name, const std::string &input_name);
        bool bindOutput(const std::string &op_name, const std::string &output_name);

        // graph uitls
        bool createGraph();
        bool compileGraph();
        bool runGraph();
        std::string getGraphName();

    private:
        // util func
        uint32_t typeGetBits(DataType type);
        convertDataType(DataType type)

    private:
        // tensor names
        std::vector<std::string>                           m_input_tensor_names;
        std::vector<std::string>                           m_output_tensor_names;
        // tensors
        std::map<std::string, std::shared_ptr<Tensor>>     m_tensors;
        std::map<std::string, TensorSpec>                  m_tensors_spec;
        std::map<std::string, std::shared_ptr<char>>       m_tensors_data;

        // operation
        std::map<std::string, Operation*>                  m_operations;
        // engine context/graph/name
        std::shared_ptr<Context>                           m_context;
        std::shared_ptr<Graph>                             m_graph;
        std::string                                        m_graph_name;
    };
} //namespace TIMVXPY
