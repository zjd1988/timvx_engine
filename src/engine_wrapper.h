/***********************************
******  engine_wrapper.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#include <iostream>
#include <string>
#include <vector>
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
        bool copy_data_from_tensor(const std::string &tensor_name, py::buffer &np_data);
        bool copy_data_to_tensor(const std::string &tensor_name, py::buffer &np_data);

        // infer engine
        bool run_engine();
    private:
        bool getFileData(std::string file_name, std::shared_ptr<char> &file_data, int &file_len);
        bool parseModelTensors(json &para_json, const char *weight_data, const int weight_len);
        bool parseModelInputs(json &para_json);
        bool parseModelOutputs(json &para_json);
        bool parseModelNodes(json &para_json);

    private:
        // tensor names
        std::vector<std::string>                       m_input_tensor_names;
        std::vector<std::string>                       m_output_tensor_names;

        // timvx engine
        std::shared_ptr<TimVXEngine>                   m_engine;
    };

}