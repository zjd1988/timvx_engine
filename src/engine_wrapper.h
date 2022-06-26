/***********************************
******  engine_wrapper.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "engine_common.h"
#include "nlohmann/json.hpp"
using namespace nlohmann;

namespace TIMVX
{
    class TimVXEngine;

    class EngineWrapper
    {
    public:
        EngineWrapper() = default;
        ~EngineWrapper() = default;

        // load model
        bool loadModelFromFile(const std::string &para_file, const std::string &weight_file);
        bool loadModelFromMemory(const char *para_data, const int para_len, 
            const char *weight_data, const int weight_len);

        // tensor utils
        TimvxInputOutputNum getInputOutputNum();
        bool setInputs(std::vector<TimvxInput> &input_data);
        bool getOutputs(std::vector<TimvxOutput> &output_data);
        bool getInputTensorAttr(int input_index, TimvxTensorAttr &tensor_attr);
        bool getOutputTensorAttr(int output_index, TimvxTensorAttr &tensor_attr);

        // infer engine
        bool run_engine();

    private:
        bool getFileData(std::string file_name, std::shared_ptr<char> &file_data, int &file_len);
        bool parseModelTensors(json &para_json, const char *weight_data, const int weight_len);
        bool parseModelInputs(json &para_json);
        bool parseModelOutputs(json &para_json);
        bool parseModelNodes(json &para_json);
        bool parseModelNormInfo(json &para_json);
    
    private:
        // norm funcs
        // channel rgb2bgr or bgr2rgb
        void inputDataReorder(char *input_data, const int input_len, 
            char* process_data, std::vector<int> order);
        // (data - mean) / std
        void inputDataMeanStd(char *input_data, const int input_len, 
            float* process_data, std::vector<float> mean, std::vector<float> std);
        // nhwc to nchw
        template <class T>
        void inputDataTranspose(T *input_data, const int input_len, int channel_num, T* process_data)
        {
            if (1 == channel_num)
            {
                if (input_data != process_data)
                    memcpy(process_data, input_data, input_len);
                return;
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
            return;
        }
        bool inputDataNorm(TimvxInput input_data, std::string input_name, 
            std::shared_ptr<char>& convert_data, int &convert_len);
        bool outputDataConvert(TimvxOutput output_data, std::string input_name, 
            std::shared_ptr<char>& convert_data, int &convert_len);

    private:
        // tensor names
        std::vector<std::string>                       m_input_tensor_names;
        std::vector<std::string>                       m_output_tensor_names;

        // output tensor data
        std::map<std::string, std::shared_ptr<char>>   m_output_tensor_datas;

        // tensor attr
        std::map<std::string, TimvxTensorAttr>         m_input_tensor_attrs;
        std::map<std::string, TimvxTensorAttr>         m_output_tensor_attrs;
        //input tensor norm info
        std::map<std::string, std::vector<float>>      m_tensor_means;
        std::map<std::string, std::vector<float>>      m_tensor_stds;
        std::map<std::string, std::vector<int>>        m_tensor_reorders;

        // timvx engine
        std::shared_ptr<TimVXEngine>                   m_engine;
    };

}// namespace TIMVX