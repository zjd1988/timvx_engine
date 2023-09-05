/***********************************
******  timvx_engine.h
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#pragma once
#include <map>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/operation.h"
#include "tim/transform/layout_inference.h"
#include "common/timvx_log.h"
#include "nlohmann/json.hpp"
#include "timvx_c_api.h"
using namespace tim::vx;
using namespace tim::transform;
using namespace nlohmann;

namespace TimVX
{

    class TimVXEngine 
    {
    public:
        TimVXEngine(const std::string& graph_name);
        ~TimVXEngine();

        // tensor utils
        size_t getTensorElemCount(const std::string& tensor_name);
        size_t getTensorByteSize(const std::string& tensor_name);
        std::vector<uint32_t> getTensorDims(const std::string& tensor_name);
        Tensor* getTensor(const std::string& tensor_name);
        bool createTensor(const std::string& tensor_name, const json& tensor_info, 
            const char* weight_data = nullptr, const int weight_len = 0);
        bool copyDataFromTensor(const std::string& tensor_name, char* buffer_data, const int buffer_len);
        bool copyDataToTensor(const std::string& tensor_name, const char* buffer_data, const int buffer_len);

        // operation utils
        bool createOperation(const json& op_info);
        json getOpInfo(const std::string& op_name);
        bool bindInputs(const std::string& op_name, const std::vector<std::string>& input_list);
        bool bindOutputs(const std::string& op_name, const std::vector<std::string>& output_list);
        bool bindInput(const std::string& op_name, const std::string& input_name);
        bool bindOutput(const std::string& op_name, const std::string& output_name);

        // norm utils
        bool createNormInfo(const json& norm_json);

        // graph uitls
        bool createGraph();
        bool verifyGraph();
        bool compileGraph();
        bool runGraph();
        bool compileToBinary(std::vector<uint8_t>& nbg_buf, size_t& bin_size);
        bool compileToBinaryAndSave(const char* weight_file, const char* para_file);
        std::string getGraphName();

        // get input + output tensor num
        int getInputOutputNum(TimvxInputOutputNum& io_num);

        // set inputs / get outputs
        int setInputs(std::vector<TimvxInput>& input_datas);
        int getOutputs(std::vector<TimvxOutput>& output_datas);

        // get input/output tensor attr
        int getInputTensorAttr(int input_index, TimvxTensorAttr& tensor_attr);
        int getOutputTensorAttr(int output_index, TimvxTensorAttr& tensor_attr);

        // util func
        uint32_t getTypeBits(DataType type);
        int convertToTimVxDataType(DataType type, TimvxTensorType& tensor_type);

    private:
        int getTensorAttr(const std::string& tensor_name, TimvxTensorAttr& tensor_info);
        // quant/dequant func
        int quantTensorData(std::string tensor_name, float* src_data, int src_len, uint8_t* quant_data);
        int dequantTensorData(std::string tensor_name, uint8_t* src_data, int src_len, float* dequant_data);

        // norm funcs
        int inputDataNorm(TimvxInput input_data, std::string input_name, 
            std::shared_ptr<char>& norm_data, int& norm_len);
        // channel rgb2bgr or bgr2rgb
        int inputDataReorder(char *input_data, const int input_len, char* process_data, 
            std::vector<int> order);
        // (data - mean) / std
        int inputDataMeanStd(char *input_data, const int input_len, float* process_data, 
            std::vector<float> mean, std::vector<float> std);
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
                dst_data = (T*)temp_data.get();
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

        // output data convert func
        int outputDataConvert(TimvxOutput& out_data, std::string output_name, 
            std::shared_ptr<char>& convert_data, int& convert_len);

    private:
        // operation func
        // tensor names
        std::vector<std::string>                                            m_input_tensor_names;
        std::vector<std::string>                                            m_output_tensor_names;
        // tensors
        std::map<std::string, std::shared_ptr<Tensor>>                      m_tensors;
        std::map<std::string, std::shared_ptr<char>>                        m_tensors_data;
        // operation
        std::map<std::string, Operation*>                                   m_operations;
        std::map<std::string, json>                                         m_op_info;
        // engine context/graph/name
        std::shared_ptr<Context>                                            m_context;
        std::shared_ptr<Graph>                                              m_graph;
        std::string                                                         m_graph_name;
        // call verify graph get a new graph
        std::pair<std::shared_ptr<Graph>, 
            std::map<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>     m_layout_infered;

        //input tensor norm info
        std::map<std::string, std::vector<float>>                           m_tensor_means;
        std::map<std::string, std::vector<float>>                           m_tensor_stds;
        std::map<std::string, std::vector<int>>                             m_tensor_reorders;

        // output tensor data
        std::map<std::string, std::shared_ptr<char>>                        m_output_datas;
    };

} //namespace TimVX
