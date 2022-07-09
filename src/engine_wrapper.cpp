/***********************************
******  engine_wrapper.cpp
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#include <iostream>
#include <fstream>
#include "timvx_define.h"
#include "timvx_engine.h"
#include "engine_wrapper.h"

namespace TIMVX
{

    int EngineWrapper::getFileData(std::string file_name, std::shared_ptr<char> &file_data, int &file_len)
    {
        file_data.reset();
        std::ifstream file_stream(file_name, std::ios::binary|std::ios::in);
        if (!file_stream.is_open())
        {
            TIMVX_ERROR("get file data from %s fail\n", file_name.c_str());
            return -1;
        }
        file_stream.seekg(0,std::ios::end);
        file_len = file_stream.tellg();
        file_stream.seekg(0,std::ios::beg);
        file_data.reset(new char[file_len], std::default_delete<char []>());
        int read_len = 0;
        file_stream.read(file_data.get(), file_len);
        if(!file_stream.bad()) 
        {
            read_len = file_stream.gcount();
        }
        if (read_len != file_len)
        {
            TIMVX_ERROR("read file %s bytes %d not equal to file actual bytes %d\n", 
                file_name.c_str(), read_len, file_len);
            return -1;
        }
        return 0;
    }
    
    int EngineWrapper::loadModelFromFile(const std::string &para_file, const std::string &weight_file)
    {
        std::shared_ptr<char> para_data;
        int para_len = 0;
        std::shared_ptr<char> weight_data;
        int weight_len = 0;
        if (!getFileData(para_file, para_data, para_len) || 
            !getFileData(weight_file, weight_data, weight_len))
            return -1;

        return loadModelFromMemory(para_data.get(), para_len, weight_data.get(), weight_len);
    
    }

    int EngineWrapper::loadModelFromMemory(const char *para_data, const int para_len, 
        const char *weight_data, const int weight_len)
    {
        m_engine.reset(new TimVXEngine("timvx_graph"));
        if (nullptr == m_engine.get() || !m_engine->createGraph())
        {
            m_engine.reset();
            TIMVX_PRINT("create timvx graph fail\n");
            return -1;
        }
        json para_json = json::parse(para_data, para_data + para_len);
        if (!parseModelInputs(para_json) || !parseModelOutputs(para_json) || 
            !parseModelTensors(para_json, weight_data, weight_len) || 
            !parseModelNodes(para_json) || !parseModelNormInfo(para_json))
            return -1;
        if (!m_engine->compileGraph())
        {
            TIMVX_PRINT("compile timvx graph fail\n");
            return -1;
        }
        return 0;
    }

    int EngineWrapper::parseModelInputs(const json &para_json)
    {
        if (!para_json.contains("inputs"))
        {
            TIMVX_ERROR("para file not contain inputs info\n");
            return -1;
        }
        if (!para_json["inputs"].is_array())
        {
            TIMVX_ERROR("para file's inputs should be array type\n");
            return -1;
        }
        for (int i = 0; i < para_json["inputs"].size(); i++)
        {
            json tensor_json = para_json["inputs"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_ERROR("para file's index:%d input tensor'name is invalid\n", i);
                return -1;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!m_engine->createTensor(tensor_name, tensor_json))
                return -1;
        }
        return 0;
    }

    int EngineWrapper::parseModelOutputs(const json &para_json)
    {
        if (!para_json.contains("outputs"))
        {
            TIMVX_ERROR("para file not contain outputs info\n");
            return -1;
        }
        if (!para_json["outputs"].is_array())
        {
            TIMVX_ERROR("para file's outputs should be array type\n");
            return -1;
        }
        for (int i = 0; i < para_json["outputs"].size(); i++)
        {
            json tensor_json = para_json["outputs"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_ERROR("para file's index:%d output tensor'name is invalid\n", i);
                return -1;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!m_engine->createTensor(tensor_name, tensor_json))
                return -1;
        }
        return 0;
    }

    int EngineWrapper::parseModelTensors(const json &para_json, const char *weight_data, const int weight_len)
    {
        if (!para_json.contains("tensors"))
        {
            TIMVX_ERROR("para file not contain tensors info\n");
            return -1;
        }
        if (!para_json["tensors"].is_array())
        {
            TIMVX_ERROR("para file's tensors should be array type\n");
            return -1;
        }
        for (int i = 0; i < para_json["tensors"].size(); i++)
        {
            json tensor_json = para_json["tensors"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_ERROR("para file's index:%d var tensor'name is invalid\n", i);
                return -1;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!m_engine->createTensor(tensor_name, tensor_json))
                return -1;
        }
        return 0;
    }

    int EngineWrapper::parseModelNodes(const json &para_json)
    {
        if (!para_json.contains("nodes"))
        {
            TIMVX_ERROR("para file not contain nodes info\n");
            return -1;
        }
        if (!para_json["nodes"].is_array())
        {
            TIMVX_ERROR("para file's nodes should be array type\n");
            return -1;
        }
        for (int i = 0; i < para_json["nodes"].size(); i++)
        {
            json node_json = para_json["nodes"][i];
            if (!m_engine->createOperation(node_json))
                return -1;
        }
        return 0;
    }

    int EngineWrapper::parseModelNormInfo(const json &para_json)
    {
        if (!para_json.contains("norm"))
        {
            TIMVX_PRINT("para file not contain norm info\n");
            return -1;
        }
        json norm_json = para_json["norm"];
        return m_engine->parseNormInfo(norm_json);
    }

    int EngineWrapper::getInputOutputNum(TimvxInputOutputNum& io_num)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return -1;
        }
        return m_engine->getInputOutputNum(io_num);
    }

    int EngineWrapper::getInputTensorAttr(int input_index, TimvxTensorAttr &tensor_attr)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return -1;
        }
        return m_engine->getInputTensorAttr(input_index, tensor_attr);
    }

    int EngineWrapper::getOutputTensorAttr(int output_index, TimvxTensorAttr &tensor_attr)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return -1;
        }
        return m_engine->getInputTensorAttr(output_index, tensor_attr);
    }

    int EngineWrapper::setInputs(std::vector<TimvxInput> &input_data)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return -1;
        }
        return m_engine->setInputs(input_data);
    }

    int EngineWrapper::getOutputs(std::vector<TimvxOutput> &output_data)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return -1;
        }
        return m_engine->getOutputs(output_data);
    }

    int EngineWrapper::runEngine()
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return -1;
        }
        return m_engine->runGraph();
    }

} // TIMVX
