/***********************************
******  timvx_cxx_api.cpp
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#include <iostream>
#include <fstream>
#include "common/timvx_log.h"
#include "common/io_uitl.h"
#include "common/json_parse.h"
#include "timvx_cxx_api.h"
#include "timvx_engine.h"

namespace TimVX
{

    static int parseModelTensors(TimVXEngine* engine, const json& para_json, const char* weight_data, const int weight_len)
    {
        if (nullptr == engine)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx engine obj is nullptr");
            return -1;
        }
        if (!para_json.contains("tensors"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file not contain tensors info");
            return -1;
        }
        if (!para_json["tensors"].is_array())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's tensors should be array type");
            return -1;
        }
        for (int i = 0; i < para_json["tensors"].size(); i++)
        {
            json tensor_json = para_json["tensors"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's index:%d var tensor'name is invalid", i);
                return -1;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!engine->createTensor(tensor_name, tensor_json, weight_data, weight_len))
                return -1;
        }
        return 0;
    }

    static int parseModelInputs(TimVXEngine* engine, const json& para_json)
    {
        if (nullptr == engine)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx engine obj is nullptr");
            return -1;
        }
        if (!para_json.contains("inputs"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file not contain inputs info");
            return -1;
        }
        if (!para_json["inputs"].is_array())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's inputs should be array type");
            return -1;
        }
        for (int i = 0; i < para_json["inputs"].size(); i++)
        {
            json tensor_json = para_json["inputs"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's index:%d input tensor name is invalid", i);
                return -1;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!engine->createTensor(tensor_name, tensor_json))
                return -1;
        }
        return 0;
    }

    static int parseModelOutputs(TimVXEngine* engine, const json& para_json)
    {
        if (nullptr == engine)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx engine obj is nullptr");
            return -1;
        }
        if (!para_json.contains("outputs"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file not contain outputs info");
            return -1;
        }
        if (!para_json["outputs"].is_array())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's outputs should be array type");
            return -1;
        }
        for (int i = 0; i < para_json["outputs"].size(); i++)
        {
            json tensor_json = para_json["outputs"][i];
            if (!tensor_json.contains("name") || !tensor_json["name"].is_string())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's index:%d output tensor name is invalid", i);
                return -1;
            }
            std::string tensor_name = tensor_json.at("name");
            if (!engine->createTensor(tensor_name, tensor_json))
                return -1;
        }
        return 0;
    }

    static int parseModelNodes(TimVXEngine* engine, const json& para_json)
    {
        if (nullptr == engine)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx engine obj is nullptr");
            return -1;
        }
        if (!para_json.contains("nodes"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file not contain nodes info");
            return -1;
        }
        if (!para_json["nodes"].is_array())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's nodes should be array type");
            return -1;
        }
        for (int i = 0; i < para_json["nodes"].size(); i++)
        {
            json node_json = para_json["nodes"][i];
            if (!engine->createOperation(node_json))
                return -1;

            // bind input/output tensors
            std::string op_name = node_json.at("op_name");
            if (!node_json.contains("op_inputs") || !node_json.contains("op_outputs"))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op_info have no op_inputs or op_outputs to bind");
                return -1;
            }

            if (node_json.contains("op_inputs") && !node_json["op_inputs"].is_array())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s op_inputs item contained, but op_inputs is not list", op_name);
                return -1;
            }
            if (node_json.contains("op_outputs") && !node_json["op_outputs"].is_array())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s op_outputs item contained, but op_outputs is not list", op_name);
                return -1;
            }

            std::vector<std::string> op_inputs;
            std::vector<std::string> op_outputs;
            if (!parseDynamicList<std::string>(node_json, "op_info", "op_inputs", op_inputs) || 
                !parseDynamicList<std::string>(node_json, "op_info", "op_outputs", op_outputs) ||
                !engine->bindInputs(op_name, op_inputs) ||
                !engine->bindOutputs(op_name, op_outputs))
                return false;
        }
        return 0;
    }

    static int parseModelNormInfo(TimVXEngine* engine, const json& para_json)
    {
        if (nullptr == engine)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx engine obj is nullptr");
            return -1;
        }
        if (!para_json.contains("norm"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file not contain norm info");
            return -1;
        }
        json norm_json = para_json["norm"];
        return engine->createNormInfo(norm_json) ? 0 : -1;
    }

    int EngineInterface::loadModelFromMemory(const char* para_data, const int para_len, 
        const char* weight_data, const int weight_len)
    {
        try
        {
            m_engine.reset(new TimVXEngine("timvx_graph"));
            if (nullptr == m_engine.get() || !m_engine->createGraph())
            {
                m_engine.reset();
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "create timvx graph fail");
                return -1;
            }
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "create timvx graph success");
            json para_json = json::parse(para_data, para_data + para_len);
            if ((0 != parseModelInputs(m_engine.get(), para_json)) || 
                (0 != parseModelTensors(m_engine.get(), para_json, weight_data, weight_len)) || 
                (0 != parseModelOutputs(m_engine.get(), para_json)) || 
                (0 != parseModelNodes(m_engine.get(), para_json)) || 
                (0 != parseModelNormInfo(m_engine.get(), para_json)))
                return -1;
            if (!m_engine->compileGraph())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "compile timvx graph fail");
                return -1;
            }
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "compile timvx graph success");
        }
        catch(const std::exception& e)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "exception occur: {}", e.what());
            return -1;
        }
        return 0;
    }

    int EngineInterface::loadModelFromFile(const std::string para_file, const std::string weight_file)
    {
        std::shared_ptr<char> para_data;
        int para_len = 0;
        std::shared_ptr<char> weight_data;
        int weight_len = 0;
        if (0 != readFileData(para_file, para_data, para_len) || 
            0 != readFileData(weight_file, weight_data, weight_len))
            return -1;
        return loadModelFromMemory(para_data.get(), para_len, weight_data.get(), weight_len);
    }

    int EngineInterface::getInputOutputNum(TimvxInputOutputNum& io_num)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx infer engine is null, please init first");
            return -1;
        }
        return m_engine->getInputOutputNum(io_num);
    }

    int EngineInterface::getInputTensorAttr(int input_index, TimvxTensorAttr& tensor_attr)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx infer engine is null, please init first");
            return -1;
        }
        return m_engine->getInputTensorAttr(input_index, tensor_attr);
    }

    int EngineInterface::getOutputTensorAttr(int output_index, TimvxTensorAttr& tensor_attr)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx infer engine is null, please init first");
            return -1;
        }
        return m_engine->getOutputTensorAttr(output_index, tensor_attr);
    }

    int EngineInterface::setInputs(std::vector<TimvxInput>& input_datas)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx infer engine is null, please init first");
            return -1;
        }
        return m_engine->setInputs(input_datas);
    }

    int EngineInterface::getOutputs(std::vector<TimvxOutput>& output_datas)
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx infer engine is null, please init first");
            return -1;
        }
        return m_engine->getOutputs(output_datas);
    }

    int EngineInterface::runEngine()
    {
        if (nullptr == m_engine.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx infer engine is null, please init first");
            return -1;
        }
        return m_engine->runGraph() == true ? 0 : -1;
    }

    bool EngineInterface::compileModelAndSave(const char* weight_file, const char* para_file)
    {
        return m_engine->compileToBinaryAndSave(weight_file, para_file);
    }

    EngineInterface::EngineInterface(const std::string para_file, const std::string weight_file)
    {
        if (0 == loadModelFromFile(para_file, weight_file))
            m_status = true;
    }

} // namespace TimVX