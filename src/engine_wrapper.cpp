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
        if (nullptr == m_engine.get() || !m_engine->createGraph())
        {
            m_engine.reset();
            TIMVX_PRINT("create timvx graph fail\n");
            return false;
        }
        json para_json = json::parse(para_data, para_data + para_len);
        if (!parseModelInputs(para_json) || !parseModelOutputs(para_json) || 
            !parseModelTensors(para_json, weight_data, weight_len) || 
            !parseModelNodes(para_json) || !parseModelNormInfo(para_json))
            return false;
        if (!m_engine->compileGraph())
        {
            TIMVX_PRINT("compile timvx graph fail\n");
            return false;
        }
        for (int i = 0; i < m_input_tensor_names.size(); i++)
        {
            TimvxTensorAttr tensor_info;
            std::string tensor_name = m_input_tensor_names[i];
            if (m_engine->getTensorInfo(tensor_name, tensor_info))
            {
                TIMVX_PRINT("get input tesnor %s spec fail\n", tensor_name.c_str());
                return false;
            }
            tensor_info.index = i;
            m_input_tensor_attrs[tensor_name] = tensor_info;
        }
        for (int i = 0; i < m_output_tensor_names.size(); i++)
        {
            TimvxTensorAttr tensor_info;
            std::string tensor_name = m_output_tensor_names[i];
            if (m_engine->getTensorInfo(tensor_name, tensor_info))
            {
                TIMVX_PRINT("get output tesnor %s attr fail\n", tensor_name.c_str());
                return false;
            }
            tensor_info.index = i;
            m_output_tensor_attrs[tensor_name] = tensor_info;
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
            if (!m_engine->createTensor(tensor_name, tensor_json))
                return false;
            m_input_tensor_names.push_back(tensor_name);
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
            if (!m_engine->createTensor(tensor_name, tensor_json))
                return false;
            m_output_tensor_names.push_back(tensor_name);
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
            if (!m_engine->createTensor(tensor_name, tensor_json))
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
            if (!m_engine->createOperation(node_json))
                return false;
        }
        return true;
    }

    bool EngineWrapper::parseModelNormInfo(json &para_json)
    {
        if (!para_json.contains("norm"))
        {
            TIMVX_PRINT("para file not contain norm info\n");
            return false;
        }
        json norm_json = para_json["norm"];
        try 
        {
            for (int i = 0; i < m_input_tensor_names.size(); i++)
            {
                std::string tensor_name = m_input_tensor_names[i];
                if (!norm_json.contains(tensor_name.c_str()))
                    continue;
                json tensor_norm = norm_json[tensor_name.c_str()];
                std::vector<float> mean_val;
                std::vector<float> std_val;
                std::vector<int> reorder_val;
                if (tensor_norm.contains("mean") && !tensor_norm["mean"].is_array())
                {
                    TIMVX_ERROR("para file's nodes should be array type\n");
                    return false;
                }
                mean_val = tensor_norm["mean"].get<std::vector<float>>();
                if (tensor_norm.contains("std") && !tensor_norm["std"].is_array())
                {
                    TIMVX_ERROR("para file's nodes should be array type\n");
                    return false;
                }
                std_val = tensor_norm["std"].get<std::vector<float>>();
                if (tensor_norm.contains("reorder") && !tensor_norm["reorder"].is_array())
                {
                    TIMVX_ERROR("para file's nodes should be array type\n");
                    return false;
                }
                reorder_val = tensor_norm["reorder"].get<std::vector<int>>();
            
                if (reorder_val.size() != 3)
                {
                    TIMVX_ERROR("norm info reorder's size should be 3\n");
                    return false;
                }
                if ((reorder_val[0] != 2 || reorder_val[1] != 1 || reorder_val[2] != 0) ||
                    (reorder_val[0] != 0 || reorder_val[1] != 1 || reorder_val[2] != 2))
                {
                    TIMVX_ERROR("norm info reorder only support [0, 1, 2] or [2, 1, 0]\n");
                    return false;
                }
                m_tensor_means[tensor_name] = mean_val;
                m_tensor_stds[tensor_name] = std_val;
                m_tensor_reorders[tensor_name] = reorder_val;
            }
        }
        catch(const std::exception& e)
        {
            TIMVX_ERROR("exception occur: %s\n", e.what());
            return false;
        }
        return true;
    }

    TimvxInputOutputNum EngineWrapper::getInputOutputNum()
    {
        TimvxInputOutputNum io_num;
        io_num.n_input = m_input_tensor_names.size();
        io_num.n_output = m_output_tensor_names.size();
        return io_num;
    }

    bool EngineWrapper::getInputTensorAttr(int input_index, TimvxTensorAttr &tensor_attr)
    {
        if (input_index < 0 || input_index >= m_input_tensor_names.size())
            return false;
        std::string tensor_name = m_input_tensor_names[input_index];
        tensor_attr = m_input_tensor_attrs[tensor_name];
        return true;
    }

    bool EngineWrapper::getOutputTensorAttr(int output_index, TimvxTensorAttr &tensor_attr)
    {
        if (output_index < 0 || output_index >= m_output_tensor_names.size())
            return false;
        std::string tensor_name = m_output_tensor_names[output_index];
        tensor_attr = m_output_tensor_attrs[tensor_name];
        return true;
    }

    bool EngineWrapper::setInputs(std::vector<TimvxInput> &input_data)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return false;
        }
        if (input_data.size() != m_input_tensor_names.size())
        {
            TIMVX_PRINT("input data size %d not equalt to engine's input size %d\n",
                input_data.size(), m_input_tensor_names.size());
            return false;
        }
        for (int i = 0; i < input_data.size(); i++)
        {
            TimvxInput input = input_data[i];
            std::string tensor_name = m_input_tensor_names[i];
            int convert_len = 0;
            std::shared_ptr<char> convert_data;
            const char* buffer_data = (const char*)input.buf;
            int buffer_len = input.size;
            if (!inputDataNorm(input, convert_data, convert_len))
            {
                TIMVX_ERROR("input data normalization fail\n");
                return false;
            }
            if (convert_data.get())
            {
                buffer_data = (const char*)convert_data.get();
                buffer_len = convert_len;
            }
            if (!m_engine->copyDataToTensor(tensor_name, buffer_data, buffer_len))
            {
                TIMVX_ERROR("copy data to tensor %s fail\n", tensor_name.c_str());
                return false;
            }
        }
        return true;
    }

    bool EngineWrapper::getOutputs(std::vector<TimvxOutput> &output_data)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return false;
        }
        if (output_data.size() <= m_output_tensor_names.size())
        {
            TIMVX_PRINT("output data size %d not equalt to engine's output size %d\n",
                output_data.size(), m_output_tensor_names.size());
            return false;
        }
        for (int i = 0; i < output_data.size(); i++)
        {
            TimvxOutput output = output_data[i];
            std::string tensor_name = m_output_tensor_names[i];
            int convert_len = 0;
            std::shared_ptr<char> convert_data;
            if (!outputDataConvert(output, convert_data, convert_len))
            {
                TIMVX_ERROR("output data convert fail\n");
                return false;
            }
            if (convert_data.get())
            {
                output.buf = convert_data.get();
                output.index = i;
                output.size = convert_len;
            }
            else if (output.want_float)
            {
                char* buffer_data = (char*)output.buf;
                int buffer_len = output.size;
                if (!m_engine->copyDataFromTensor(tensor_name, buffer_data, buffer_len))
                {
                    TIMVX_ERROR("copy data from tensor %s fail\n", tensor_name.c_str());
                    return false;
                }
            }
            else
            {
                char* buffer_data = (char*)output.buf;
                int buffer_len = output.size;
                if (!m_engine->copyDataFromTensor(tensor_name, buffer_data, buffer_len))
                {
                    TIMVX_ERROR("copy data from tensor %s fail\n", tensor_name.c_str());
                    return false;
                }
            }
        }
        return true;
    }

    bool EngineWrapper::run_engine()
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_ERROR("timvx infer engine is null, please init first\n");
            return false;
        }
        return m_engine->runGraph();
    }

    void EngineWrapper::inputDataReorder(char *input_data, const int input_len, char* reorder_data)
    {

        return;
    }

    void EngineWrapper::inputDataMeanStd(char *input_data, const int input_len, char* reorder_data)
    {

        return;
    }

    bool EngineWrapper::inputDataNorm(TimvxInput input_data, std::shared_ptr<char>& convert_data, int &convert_len)
    {
        if (input_data.pass_through)
            return true;
        
        return true;
    }


    bool EngineWrapper::outputDataConvert(TimvxOutput output_data, std::shared_ptr<char>& convert_data, int &convert_len)
    {
        if (!output_data.is_prealloc)
        {

        }
        return true;
    }


} // TIMVX
