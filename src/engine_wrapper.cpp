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
    bool EngineWrapper::getFileData(std::string file_name, std::shared_ptr<char> &file_data, int &file_len)
    {
        file_data.reset();
        std::ifstream file_stream(file_name, std::ios::binary|std::ios::in);
        if (!file_stream.is_open())
        {
            TIMVX_ERROR("get file data from %s fail\n", file_name.c_str());
            return false;
        }
        file_stream.seekg(0,std::ios::end);
        int file_len = file_stream.tellg();
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
            return false;
        }
        return true;
    }
    
    bool EngineWrapper::loadModelFromFile(const std::string &para_file, const std::string &weight_file)
    {
        std::shared_ptr<char> para_data;
        int para_len = 0;
        std::shared_ptr<char> weight_data;
        int weight_len = 0;
        if (!getFileData(para_file, para_data, para_len) || 
            !getFileData(weight_file, weight_data, weight_len))
            return false;

        return loadModelFromMemory(para_data.get(), para_len, weight_data.get(), weight_len);
    
    }

    bool EngineWrapper::loadModelFromMemory(const char *para_data, const int para_len, 
        const char *weight_data, const int weight_len)
    {
        m_engine.reset(new TimVXEngine("timvx_graph"));
        if (nullptr == m_engine.get() || !m_engine->create_graph())
        {
            TIMVX_PRINT("create timvx graph fail\n");
            return false;
        }
        json para_json = json::parse(para_data, para_data + para_len);
        if (!parseModelInputs(para_json) || !parseModelOutputs(para_json) || 
            !parseModelTensors(para_json, weight_data, weight_len) || !parseModelNodes(para_json))
            return false;
        if (!m_engine.compile_graph())
        {
            TIMVX_PRINT("compile timvx graph fail\n");
            return false;
        }
        return true;
    }

    bool EngineWrapper::parseModelInputs(json &para_json)
    {
        if (!para_json.contains("inputs"))
        {
            TIMVX_ERROR("para file not contain inputs info\n");
            return false;
        }
        if (!para_json["inputs"].is_array())
        {
            TIMVX_ERROR("para file's inputs should be array type\n");
            return false;
        }
        for (int i = 0; i < para_json["inputs"].size(); i++)
        {
            json tensor_json = para_json["inputs"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_ERROR("para file's index:%d input tensor'name is invalid\n", i);
                return false;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!m_engine->create_tensor(tensor_name, tensor_json))
                return false;
        }
        return true;
    }

    bool EngineWrapper::parseModelOutputs(json &para_json)
    {
        if (!para_json.contains("outputs"))
        {
            TIMVX_ERROR("para file not contain outputs info\n");
            return false;
        }
        if (!para_json["outputs"].is_array())
        {
            TIMVX_ERROR("para file's outputs should be array type\n");
            return false;
        }
        for (int i = 0; i < para_json["outputs"].size(); i++)
        {
            json tensor_json = para_json["outputs"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_ERROR("para file's index:%d output tensor'name is invalid\n", i);
                return false;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!m_engine->create_tensor(tensor_name, tensor_json))
                return false;
        }
        return true;
    }

    bool EngineWrapper::parseModelTensors(json &para_json, const char *weight_data, const int weight_len)
    {
        if (!para_json.contains("tensors"))
        {
            TIMVX_ERROR("para file not contain tensors info\n");
            return false;
        }
        if (!para_json["tensors"].is_array())
        {
            TIMVX_ERROR("para file's tensors should be array type\n");
            return false;
        }
        for (int i = 0; i < para_json["tensors"].size(); i++)
        {
            json tensor_json = para_json["tensors"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_ERROR("para file's index:%d var tensor'name is invalid\n", i);
                return false;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!m_engine->create_tensor(tensor_name, tensor_json))
                return false;
        }
        return true;
    }

    bool EngineWrapper::parseModelNodes(json &para_json)
    {
        if (!para_json.contains("nodes"))
        {
            TIMVX_ERROR("para file not contain nodes info\n");
            return false;
        }
        if (!para_json["nodes"].is_array())
        {
            TIMVX_ERROR("para file's nodes should be array type\n");
            return false;
        }
        for (int i = 0; i < para_json["nodes"].size(); i++)
        {
            json node_json = para_json["nodes"][i];
            if (!m_engine->create_operation(node_json))
                return false;
        }
        return true;
    }

} // TIMVX
