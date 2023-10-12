/***********************************
******  timvx_cxx_api.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "timvx_c_api.h"

namespace TimVX
{

    class TimVXEngine;

    class EngineInterface
    {
    public:
        EngineInterface(const std::string para_file, const std::string weight_file, bool load_only = false);
        ~EngineInterface() = default;

        // tensor utils
        int getInputOutputNum(TimvxInputOutputNum& io_num);
        int setInputs(std::vector<TimvxInput>& input_datas);
        int getOutputs(std::vector<TimvxOutput>& output_datas);
        int getInputTensorAttr(int input_index, TimvxTensorAttr& tensor_attr);
        int getOutputTensorAttr(int output_index, TimvxTensorAttr& tensor_attr);

        // infer engine
        int runEngine();

        // get engine status
        bool getEngineStatus() { return m_status; }

        // export graph
        bool exportGraph(const char* weight_file, const char* para_file);

        // export NBG graph
        bool exportNBGGraph(const char* weight_file, const char* para_file);

    private:
        // load model
        int loadModelFromFile(const std::string para_file, const std::string weight_file, bool load_only);
        int loadModelFromMemory(const char* para_data, const int para_len, const char* weight_data, 
            const int weight_len, bool load_only);

    private:
        // timvx engine
        std::shared_ptr<TimVXEngine>                   m_engine;
        bool                                           m_status = false;
    };

}// namespace TimVX