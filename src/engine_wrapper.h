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
        int loadModelFromFile(const std::string &para_file, const std::string &weight_file);
        int loadModelFromMemory(const char *para_data, const int para_len, 
            const char *weight_data, const int weight_len);

        // tensor utils
        int getInputOutputNum(TimvxInputOutputNum& io_num);
        int setInputs(std::vector<TimvxInput> &input_data);
        int getOutputs(std::vector<TimvxOutput> &output_data);
        int getInputTensorAttr(int input_index, TimvxTensorAttr &tensor_attr);
        int getOutputTensorAttr(int output_index, TimvxTensorAttr &tensor_attr);

        // infer engine
        int runEngine();

    private:
        int getFileData(std::string file_name, std::shared_ptr<char> &file_data, int &file_len);
        int parseModelTensors(const json &para_json, const char *weight_data, const int weight_len);
        int parseModelInputs(const json &para_json);
        int parseModelOutputs(const json &para_json);
        int parseModelNodes(const json &para_json);
        int parseModelNormInfo(const json &para_json);

    private:
        // timvx engine
        std::shared_ptr<TimVXEngine>                   m_engine;
    };

}// namespace TIMVX