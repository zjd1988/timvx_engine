/***********************************
******  timvx_engine.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#pragma once
#include <map>
#include <vector>
#include <string>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/operation.h"
#include "nlohmann/json.hpp"
#include "engine_common.h"
#include "timvx_define.h"
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
        int createTensor(const std::string &tensor_name, const json &tensor_info, 
            const char *weight_data = nullptr, const int weight_len = 0);
        int getTensorInfo(const std::string &tensor_name, TimvxTensorAttr& tensor_spec);
        int copyDataFromTensor(const std::string &tensor_name, char* data, const int data_len);
        int copyDataToTensor(const std::string &tensor_name, const char* data, const int data_len);
        int getInputOutputNum(TimvxInputOutputNum &io_num);
        int getInputTensorAttr(int input_index, TimvxTensorAttr &tensor_attr);
        int getOutputTensorAttr(int output_index, TimvxTensorAttr &tensor_attr);

        // operation utils
        int createOperation(const json &op_info);

        // parse norm info
        int parseNormInfo(const json &norm_info);

        // graph uitls
        int createGraph();
        int compileGraph();
        int runGraph();
        std::string getGraphName();

        // set inputs / get outputs
        int setInputs(std::vector<TimvxInput> &input_data);
        int getOutputs(std::vector<TimvxOutput> &output_data);

    private:
        int bindInputs(const std::string &op_name, const std::vector<std::string> &input_list);
        int bindOutputs(const std::string &op_name, const std::vector<std::string> &output_list);
        int bindInput(const std::string &op_name, const std::string &input_name);
        int bindOutput(const std::string &op_name, const std::string &output_name);

        // util func
        uint32_t typeGetBits(DataType type);
        int convertToTimVxDataType(DataType type, TimvxTensorType& tensor_type);
        int convertToDataType(TimvxTensorType tensor_type, DataType &type);
        size_t getTensorByteSize(const std::string &tensor_name);
        size_t getTensorElemSize(const std::string &tensor_name);

    private:

        // norm funcs
        // channel rgb2bgr or bgr2rgb
        int inputDataReorder(char *input_data, const int input_len, char* process_data, std::vector<int> order);
        // (data - mean) / std
        int inputDataMeanStd(char *input_data, const int input_len, float* process_data, std::vector<float> mean, std::vector<float> std);
        // nhwc to nchw
        template <class T>
        int inputDataTranspose(T *input_data, const int input_len, int channel_num, T* process_data)
        {
            if (1 == channel_num)
            {
                if (input_data != process_data)
                    memcpy(process_data, input_data, input_len);
                return 0;
            }
            std::shared_ptr<char> temp_data;
            T* dst_data = process_data;
            T* src_data = input_data;
            if (input_data == process_data)
            {
                std::shared_ptr<char> temp_data = std::shared_ptr<char>(new char[input_len], std::default_delete<char []>());
                dst_data = temp_data.get();
            }
            int row = input_len / channel_num;
            int col = channel_num;
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    int src_index = i * col + j;
                    int dst_index = j * row + i;
                    dst_data[dst_index] = src_data[src_index];
                }
            }
            if (temp_data.get())
                memcpy(process_data, temp_data.get(), input_len);
            return 0;
        }
        // norm process (contain reorder/norm/transpose)
        int inputDataNorm(TimvxInput input_data, std::string input_name, std::shared_ptr<char>& norm_data, int &norm_len);
        int quantTensorData(std::string tensor_name, float* src_data, int src_len, uint8_t* quant_data);
        int dequantTensorData(std::string tensor_name, uint8_t* src_data, int src_len, float* dequant_data);
        int outputDataConvert(TimvxOutput output_data, std::string output_name, std::shared_ptr<char>& convert_data, int &convert_len);

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

        //input tensor norm info
        std::map<std::string, std::vector<float>>          m_tensor_means;
        std::map<std::string, std::vector<float>>          m_tensor_stds;
        std::map<std::string, std::vector<int>>            m_tensor_reorders;

        // output tensor data
        std::map<std::string, std::shared_ptr<char>>       m_output_tensor_datas;
    };
} //namespace TIMVX
