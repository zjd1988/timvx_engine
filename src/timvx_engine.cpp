/***********************************
******  timvx_engine.cpp
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#include <mutex>
#include <iostream>
#include "timvx_engine.h"
#include "tensor_info.h"
#include "timvx_ops/op_creator.h"
#include "common/json_parse.h"

namespace TimVX
{

    #define BITS_PER_BYTE 8
    TimVXEngine::TimVXEngine(const std::string& graph_name)
    {
        m_context.reset();
        m_graph.reset();
        m_graph_name = graph_name;
    }

    TimVXEngine::~TimVXEngine()
    {
        m_tensors_data.clear();
        m_tensors.clear();
        m_graph.reset();
        m_context.reset();
    }

    Tensor* TimVXEngine::getTensor(const std::string& tensor_name)
    {
        if (m_tensors.find(tensor_name) == m_tensors.end())
            return nullptr;
        return m_tensors[tensor_name].get();
    }

    std::vector<uint32_t> TimVXEngine::getTensorDims(const std::string& tensor_name)
    {
        std::vector<uint32_t> tensor_dims;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return tensor_dims;
        }
        auto tensor = m_tensors[tensor_name];
        return tensor->GetShape();
    }

    int TimVXEngine::convertToTimVxDataType(DataType type, TimvxTensorType& tensor_type)
    {
        if (DataType::INT8 == type)
            tensor_type = TIMVX_TENSOR_INT8;
        else if (DataType::UINT8 == type)
            tensor_type = TIMVX_TENSOR_UINT8;
        else if (DataType::INT16 == type)
            tensor_type = TIMVX_TENSOR_INT16;
        else if (DataType::FLOAT16 == type)
            tensor_type = TIMVX_TENSOR_FLOAT16;
        else if (DataType::FLOAT32 == type)
            tensor_type = TIMVX_TENSOR_FLOAT32;
        else
            return -1;
        return 0;
    }

    uint32_t TimVXEngine::getTypeBits(DataType type)
    {   
        switch( type )
        {
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::BOOL8:
                return 8;
            case DataType::INT16:
            case DataType::UINT16:
            case DataType::FLOAT16:
                return 16;
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::FLOAT32:
                return 32;
            default:
                return 0;
        }
    }  

    size_t TimVXEngine::getTensorByteSize(const std::string& tensor_name)
    {
        size_t sz;
        size_t i;
        size_t bits_num;
        size_t dim_num;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return 0;
        }
        sz = 0;
        auto tensor = m_tensors[tensor_name];
        dim_num = tensor->GetShape().size();
        auto shape = tensor->GetShape();
        auto type = tensor->GetDataType();
        if(0 == dim_num)
        {
            return sz;
        }
        bits_num = getTypeBits( type );
        if( bits_num < BITS_PER_BYTE )
        {
            if(shape[0] % 2 == 0)
            {
                sz = shape[0] / 2;
            }
            else
            {
                sz = shape[0] / 2 + shape[0] % 2;
            }
        }
        else
        {
            sz = shape[0] * bits_num / BITS_PER_BYTE;
        }
        for( i = 1; i < dim_num; i ++ )
        {
            sz *= shape[i];
        }
        return sz;
    }

    size_t TimVXEngine::getTensorElemCount(const std::string& tensor_name)
    {
        size_t sz;
        size_t i;
        size_t bits_num;
        size_t dim_num;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists", tensor_name);
            return 0;
        }
        sz = 0;
        auto tensor = m_tensors[tensor_name];
        dim_num = tensor->GetShape().size();
        auto shape = tensor->GetShape();
        auto type = tensor->GetDataType();
        if(0 == dim_num)
        {
            return sz;
        }
        for( i = 1; i < dim_num; i ++ )
        {
            sz *= shape[i];
        }
        return sz;
    }

    bool TimVXEngine::createTensor(const std::string& tensor_name, const json& tensor_info,
        const char* weight_data, const int weight_len)
    {
        try 
        {
            if (m_graph.get() == nullptr)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
                return false;
            }
            if (m_tensors.find(tensor_name) != m_tensors.end())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "duplicate tensor name {} is provided, please check again!", tensor_name);
                return false;
            }

            std::string tensor_info_str = tensor_info.dump(4);
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "try to create tensor:{} with config:\n{}", tensor_name, tensor_info_str);

            // construct tensor spec
            TensorSpec tensor_spec;
            if (!TensorSpecConstruct::constructTensorspec(tensor_info, tensor_name, tensor_spec))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "construct tensor {} spec fail, please check again!", tensor_name);
                return false;
            }
            // if not contain 'offset' and 'length', weight_data is tensor data, weight_len is tensor data len
            // else weight_data is model data , weight_len is model data len
            // use offset and length update tensor_data and tensor_data_length
            if ((tensor_info.contains("offset") && !tensor_info.contains("length")) ||
                (!tensor_info.contains("offset") && tensor_info.contains("length")))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} config should both contain offset and length or not, please check again!", 
                    tensor_name);
                return false;
            }
            const char* tensor_data = weight_data;
            int tensor_data_length = weight_len;
            int tensor_data_offset = 0;
            if ((tensor_info.contains("offset") && 
                !parseValue<int>(tensor_info, "tensor_info", "offset", tensor_data_offset)) ||
                (tensor_info.contains("length") && 
                !parseValue<int>(tensor_info, "tensor_info", "length", tensor_data_length)))
                return false;
            tensor_data += tensor_data_offset;
            if (tensor_data && weight_data && 
                (tensor_data + tensor_data_length) > (weight_data + weight_len))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor data range exceed input weight data range, please check again!", tensor_name);
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "input weight data range:{}~{}, tensor data range:{}~{}", 
                    spdlog::fmt_lib::ptr(weight_data), spdlog::fmt_lib::ptr(weight_data + weight_len), 
                    spdlog::fmt_lib::ptr(tensor_data), spdlog::fmt_lib::ptr(tensor_data + tensor_data_length));
                return false;
            }

            // init tensor with tensor data
            std::shared_ptr<Tensor> tensor;
            if (tim::vx::CONSTANT != tensor_spec.attr_ || !tensor_data || 0 == tensor_data_length)
                tensor = m_graph->CreateTensor(tensor_spec);
            else
            {
                std::shared_ptr<char> data_array_ptr(new char[tensor_data_length], [](char* data_array_ptr){delete [] data_array_ptr;});
                m_tensors_data[tensor_name] = data_array_ptr;
                memcpy((void*)data_array_ptr.get(), tensor_data, tensor_data_length);
                tensor = m_graph->CreateTensor(tensor_spec, (void*)data_array_ptr.get());
            }
            if (nullptr == tensor.get())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "create tensor {} fail!", tensor_name);
                return false;
            }

            // store tensor info
            m_tensors[tensor_name] = tensor;
            if (TensorAttribute::INPUT == tensor_spec.attr_)
                m_input_tensor_names.push_back(tensor_name);
            if (TensorAttribute::OUTPUT == tensor_spec.attr_)
                m_output_tensor_names.push_back(tensor_name);
        }
        catch(const std::exception& e)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "exception occur: {}", e.what());
            return false;
        }
        return true;
    }    

    bool TimVXEngine::createNormInfo(const json& norm_json)
    {
        m_tensor_means.clear();
        m_tensor_stds.clear();
        m_tensor_reorders.clear();
        try 
        {
            for (int i = 0; i < m_input_tensor_names.size(); i++)
            {
                std::string tensor_name = m_input_tensor_names[i];
                if (!norm_json.contains(tensor_name))
                    continue;
                json tensor_norm = norm_json[tensor_name];
                std::vector<float> mean_val;
                std::vector<float> std_val;
                std::vector<int> reorder_val;
                if (tensor_norm.contains("mean") && !tensor_norm["mean"].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's nodes should be array type");
                    return false;
                }
                mean_val = tensor_norm["mean"].get<std::vector<float>>();
                if (tensor_norm.contains("std") && !tensor_norm["std"].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's nodes should be array type");
                    return false;
                }
                std_val = tensor_norm["std"].get<std::vector<float>>();
                if (tensor_norm.contains("reorder") && !tensor_norm["reorder"].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "para file's nodes should be array type");
                    return false;
                }
                reorder_val = tensor_norm["reorder"].get<std::vector<int>>();
            
                if (reorder_val.size() != 3)
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "norm info reorder's size should be 3");
                    return false;
                }
                if ((reorder_val[0] != 2 || reorder_val[1] != 1 || reorder_val[2] != 0) ||
                    (reorder_val[0] != 0 || reorder_val[1] != 1 || reorder_val[2] != 2))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "norm info reorder only support [0, 1, 2] or [2, 1, 0]");
                    return false;
                }
                m_tensor_means[tensor_name] = mean_val;
                m_tensor_stds[tensor_name] = std_val;
                m_tensor_reorders[tensor_name] = reorder_val;
            }
        }
        catch(const std::exception& e)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "exception occur: {}", e.what());
            return false;
        }
        return true;
    }

    bool TimVXEngine::copyDataFromTensor(const std::string& tensor_name, char* buffer_data, const int buffer_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "dst buffer data ptr is nullptr, when copy from tensor {}", tensor_name);
            return false;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        size_t total_tensor_size = getTensorByteSize(tensor_name);
        if (total_tensor_size != buffer_len)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} not equal to buffer data size:{}", 
                tensor_name, total_tensor_size, buffer_len);
            return false;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} is invalid!", tensor_name, total_tensor_size);
            return false;
        }
        return tensor->CopyDataFromTensor(buffer_data);
    }

    bool TimVXEngine::copyDataToTensor(const std::string& tensor_name, const char* buffer_data, 
        const int buffer_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "src buffer data ptr is nullptr, when copy to tensor {}", tensor_name);
            return false;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        int total_tensor_size = getTensorByteSize(tensor_name);
        if (total_tensor_size != buffer_len)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} not equal to buffer data size:{}",
                tensor_name, total_tensor_size, buffer_len);
            return false;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} is invalid!", tensor_name, total_tensor_size);
            return false;
        }
        return tensor->CopyDataToTensor(buffer_data, buffer_len);
    }

    bool TimVXEngine::createOperation(const json& op_info)
    {
        try
        {
            if (m_graph.get() == nullptr)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
                return false;
            }
            if (!op_info.contains("op_name") || !op_info["op_name"].is_string())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op_name item is not contained, or op_name is not string!");
                return false;
            }
            std::string op_name = op_info.at("op_name");
            if (!op_info.contains("op_type") || !op_info["op_type"].is_string())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s op_type item is not contained, or op_type is not string", op_name);
                return false;
            }
            if (!op_info.contains("op_attr") || !op_info["op_attr"].is_object())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s op_attr item is not contained, or op_attr is not dict", op_name);
                return false;
            }
            if (op_info.contains("rounding_policy") && !op_info["rounding_policy"].is_object())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s rounding_policy item is contained, but rounding_policy is not dict", op_name);
                return false;
            }
            if (m_operations.find(op_name) != m_operations.end())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op_name {} is duplicate", op_name);
                return false;
            }

            std::string op_type = op_info.at("op_type");
            OpCreator* op_creator = TimVXOp::getInstance()->getOpCreator(op_type);
            if (nullptr == op_creator)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} creator not find!", op_type);
                return false;
            }
            std::string op_info_str = op_info.dump(4);
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "try to create op:{} with config:\n{}", op_name, op_info_str);
            auto op_node = op_creator->onCreate(m_graph, op_info["op_attr"]);
            if (nullptr != op_node && op_info.contains("rounding_policy"))
            {
                json rounding_policy = op_info["rounding_policy"];
                OverflowPolicy overflow_policy_type = OverflowPolicy::SATURATE;
                RoundingPolicy rounding_policy_type = RoundingPolicy::RTNE;
                RoundType      round_type           = RoundType::FLOOR;
                uint32_t       accumulator_bits     = 0;
                op_creator->parseOverflowPolicyType(rounding_policy, op_name, "overflow_policy", overflow_policy_type, -1);
                op_creator->parseRoundingPolicyType(rounding_policy, op_name, "rounding_policy", rounding_policy_type, -1);
                op_creator->parseRoundType(rounding_policy, op_name, "down_scale_size_rounding", round_type, -1);
                op_creator->parseValue<uint32_t>(rounding_policy, op_name, "accumulator_bits", accumulator_bits, -1);
                op_node->SetRoundingPolicy(overflow_policy_type, rounding_policy_type, round_type, accumulator_bits);
            }
            if (nullptr == op_node)
                return false;
            m_operations[op_name] = op_node;
            m_op_info[op_name] = op_info;
        }
        catch(const std::exception& e)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "exception occur: {}", e.what());
            return false;
        }
        return true;
    }

    bool TimVXEngine::bindInputs(const std::string& op_name, const std::vector<std::string>& input_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (input_list.size() <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "bind input list is empty!");
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        for (int i = 0; i < input_list.size(); i++)
        {
            std::string tensor_name = input_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
                return false;
            }
            input_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindInputs(input_tensors);
        return true;
    }

    bool TimVXEngine::bindOutputs(const std::string& op_name, const std::vector<std::string>& output_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (output_list.size() <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "bind output list is empty!");
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> output_tensors;
        for (int i = 0; i < output_list.size(); i++)
        {
            std::string tensor_name = output_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
                return false;
            }
            output_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindOutputs(output_tensors);
        return true;
    }

    bool TimVXEngine::bindInput(const std::string& op_name, const std::string& input_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (m_tensors.find(input_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", input_name);
            return false;
        }
        std::shared_ptr<Tensor> input_tensor = m_tensors[input_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindInput(input_tensor);
        return true;
    }

    bool TimVXEngine::bindOutput(const std::string& op_name, const std::string& output_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (m_tensors.find(output_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", output_name);
            return false;
        }
        std::shared_ptr<Tensor> out_tensor = m_tensors[output_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindOutput(out_tensor);
        return true;
    }
    
    json TimVXEngine::getOpInfo(const std::string& op_name)
    {
        json op_json;
        if (m_op_info.find(op_name) != m_op_info.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return op_json;
        }
        return m_op_info[op_name];
    }

    bool TimVXEngine::createGraph()
    {
        m_context = tim::vx::Context::Create();
        if (nullptr == m_context.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "create context fail!");
            return false;
        }
        m_graph = m_context->CreateGraph();
        if (nullptr == m_graph.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "create graph fail!");
            m_context.reset();
            return false;
        }
        return true;
    }

    bool TimVXEngine::verifyGraph()
    {
        if (nullptr == m_context.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "context is invalid, please create context first!");
            return false;
        }
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        m_layout_infered = LayoutInference(m_graph, m_context);
        if (nullptr == m_layout_infered.first.get() || 0 == m_layout_infered.second.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph layout inference fail, please check ori graph!");
            return false;
        }
        return true;
    }

    bool TimVXEngine::compileGraph()
    {
        if (nullptr == m_graph.get() && nullptr == m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (nullptr != m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "layout infered graph compile ...");
            return m_layout_infered.first->Compile();
        }
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "compile graph ...");
        return m_graph->Compile();
    }

    bool TimVXEngine::runGraph()
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "run graph ...");
        return m_graph->Run();
    }

    bool TimVXEngine::compileToBinary(std::vector<uint8_t>& nbg_buf, size_t& bin_size)
    {
        Graph* graph = nullptr;
        if (nullptr == m_graph.get() && nullptr == m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }

        graph = m_graph.get();
        if (nullptr != m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_INFO, "use layout infered graph compile to binary buffer ...");
            graph = m_layout_infered.first.get();
        }

        // call compile to binary
        bin_size = -1;
        if (false == graph->CompileToBinary(nullptr, &bin_size) || 0 >= bin_size)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph compile to get binary buffer size fail ...");
            return false;
        }

        // generate binary graph does't require input data
        TIMVX_LOG(TIMVX_LEVEL_INFO, "compie binary file size is {}", bin_size);
        nbg_buf.resize(bin_size);
        if (false == graph->CompileToBinary(nbg_buf.data(), &bin_size))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph compile to binary buffer fail ...");
            return false;
        }
        return true;
    }

    bool TimVXEngine::compileToBinaryAndSave(const char* weight_file, const char* para_file)
    {
        std::vector<uint8_t> nbg_buf;
        size_t bin_size;
        if (false == compileToBinary(nbg_buf, bin_size))
            return false;
        return true;
    }

    std::string TimVXEngine::getGraphName()
    {
        return m_graph_name;
    }

    int TimVXEngine::inputDataReorder(char *input_data, const int input_len, char* process_data, std::vector<int> order)
    {
        int item_num = input_len / order.size();
        for (int i = 0; i < item_num; i++)
        {
            int index = i * order.size();
            for (int j = 0; j < order.size(); j++)
            {
                int src_index = index + order[j];
                int dst_index = index + j;
                process_data[dst_index] = input_data[src_index];
            }
        }
        return 0;
    }

    int TimVXEngine::inputDataMeanStd(char* input_data, const int input_len, float* process_data, 
        std::vector<float> mean, std::vector<float> std)
    {
        int item_num = input_len / mean.size();
        for (int i = 0; i < item_num; i++)
        {
            int index = i * mean.size();
            for (int j = 0; j < mean.size(); j++)
            {
                int src_index = index + j;
                int dst_index = src_index;
                process_data[dst_index] = (input_data[src_index] - mean[j]) / std[j];
            }
        }
        return 0;
    }

    int TimVXEngine::quantTensorData(std::string tensor_name, float* src_data, int src_len, uint8_t* quant_data)
    {
        Quantization quant_info = m_tensors[tensor_name]->GetQuantization();
        if (tim::vx::QuantType::ASYMMETRIC != quant_info.Type())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "current only support ASYMMETRIC quant type");
            return -1;
        }
        if (quant_info.Scales().size() != quant_info.ZeroPoints().size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "invalid quant info for tensor:{} , scales num not equal to zero_point", tensor_name);
            return -1;
        }

        std::vector<float> scales = quant_info.Scales();
        std::vector<int> zero_points = quant_info.ZeroPoints();
        int ch_num = quant_info.ChannelDim();
        int ch_step = src_len / ch_num;
        for (int i = 0; i < ch_num; i++)
        {
            float scale = scales[i];
            int zero_point = zero_points[i];
            int offset = ch_step * i;
            uint8_t* dst_data = (uint8_t*)quant_data + offset;
            for (int j = 0; j < ch_step; j++)
            {
                dst_data[j] = (uint8_t)(src_data[offset + j] / scale + zero_point);
            }
        }
        return 0;
    }

    int TimVXEngine::dequantTensorData(std::string tensor_name, uint8_t* src_data, int src_len, float* dequant_data)
    {
        Quantization quant_info = m_tensors[tensor_name]->GetQuantization();
        if (tim::vx::QuantType::ASYMMETRIC != quant_info.Type())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "current only support ASYMMETRIC quant type");
            return -1;
        }
        if (quant_info.Scales().size() != quant_info.ZeroPoints().size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "invalid quant info for tensor:{} , scales num not equal to zero_point", tensor_name);
            return -1;
        }

        std::vector<float> scales = quant_info.Scales();
        std::vector<int> zero_points = quant_info.ZeroPoints();
        int ch_num = quant_info.ChannelDim();
        int ch_step = src_len / ch_num;
        for (int i = 0; i < ch_num; i++)
        {
            float scale = scales[i];
            int zero_point = zero_points[i];
            int offset = ch_step * i;
            float* dst_data = dequant_data + offset;
            for (int j = 0; j < ch_step; j++)
            {
                dst_data[j] = (src_data[offset + j] - zero_point) * scale;
            }
        }
        return 0;
    }

    int TimVXEngine::inputDataNorm(TimvxInput input_data, std::string input_name, std::shared_ptr<char>& norm_data, int &norm_len)
    {
        if (input_data.pass_through)
        {
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "input tensor:{} set pass_through true, no need normalization", input_name);
            return 0;
        }

        // channel reorder
        std::shared_ptr<char> reorder_data;
        if (0 != m_tensor_reorders[input_name].size())
        {
            char* src_data = (char*)input_data.buf;
            int src_len = input_data.size;
            reorder_data = std::shared_ptr<char>(new char[src_len], std::default_delete<char []>());
            if (nullptr == reorder_data.get())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc data for tensor:{} reorder out fail", input_name);
                return -1;
            }
            std::vector<int> reorder = m_tensor_reorders[input_name];
            if (0 != inputDataReorder(src_data, src_len, reorder_data.get(), reorder))
            {
                return -1;
            }
        }

        // mean/std/transpose compute
        int ch_num = m_tensor_means.size();
        char* src_data = (char*)input_data.buf;
        int src_len = input_data.size;
        if (nullptr != reorder_data.get())
            src_data = reorder_data.get();

        if (0 != m_tensor_means.size() || 0 != m_tensor_stds.size())
        {
            std::vector<float> means = m_tensor_means[input_name];
            std::vector<float> stds = m_tensor_stds[input_name];
            norm_len = src_len * sizeof(float);
            norm_data = std::shared_ptr<char>(new char[src_len * sizeof(float)], std::default_delete<char []>());
            if (nullptr == norm_data.get())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc data for tensor:{} norm out fail", input_name);
                return -1;
            }
            if (0 != inputDataMeanStd(src_data, src_len, (float*)norm_data.get(), means, stds))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "input tensor ({} - mean) / std fail", input_name);
                return -1;
            }
            if (0 != inputDataTranspose<float>((float*)norm_data.get(), norm_len, ch_num, (float*)norm_data.get()))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "input tensor:{} transpose fail", input_name);
                return -1;
            }
        }
        else
        {
            norm_len = src_len;
            norm_data = std::shared_ptr<char>(new char[src_len], std::default_delete<char []>());
            if (0 != inputDataTranspose<uint8_t>((uint8_t*)src_data, src_len, ch_num, (uint8_t*)norm_data.get()))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "input tensor:{} transpose fail", input_name);
                return -1;
            }
        }

        return 0;
    }

    int TimVXEngine::setInputs(std::vector<TimvxInput>& input_datas)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx graph is invalid, please create graph first");
            return -1;
        }
        if (input_datas.size() != m_input_tensor_names.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "input data size {} not equalt to engine's input size {}",
                int(input_datas.size()), int(m_input_tensor_names.size()));
            return -1;
        }
        for (int i = 0; i < input_datas.size(); i++)
        {
            TimvxInput input = input_datas[i];
            std::string tensor_name = m_input_tensor_names[i];
            const char* buffer_data = (const char*)input.buf;
            int buffer_len = input.size;
            if (0 == input.pass_through)
            {
                if (m_tensor_means.end() != m_tensor_means.find(tensor_name) && 
                    m_tensor_means[tensor_name].size() && buffer_len % m_tensor_means[tensor_name].size())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "input:{} data size:{}, means size:{}", 
                        tensor_name, buffer_len, int(m_tensor_means[tensor_name].size()));
                    return -1;
                }
                if (m_tensor_stds.end() != m_tensor_stds.find(tensor_name) && 
                    m_tensor_stds[tensor_name].size() && buffer_len % m_tensor_stds[tensor_name].size())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "input:{} data size:{}, stds size:{}", 
                        tensor_name, buffer_len, int(m_tensor_stds[tensor_name].size()));
                    return -1;
                }
                // norm data
                int norm_len = 0;
                std::shared_ptr<char> norm_data;
                if (0 != inputDataNorm(input, tensor_name, norm_data, norm_len))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "input tensor:{} data normalization fail", tensor_name);
                    return -1;
                }
                if (nullptr != norm_data.get())
                {
                    buffer_data = (const char*)norm_data.get();
                    buffer_len = norm_len;
                }

                // quant data
                int quant_len = norm_len / sizeof(float);
                std::shared_ptr<char> quant_data;
                if (quant_len)
                    quant_data.reset(new char[quant_len], std::default_delete<char []>());
                if (quant_len && nullptr == quant_data.get())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc data for tensor:{} quant out fail", tensor_name);
                    return -1;
                }
                if (quant_len && nullptr != quant_data.get() && 
                    0 != quantTensorData(tensor_name, (float*)norm_data.get(), quant_len, (uint8_t*)quant_data.get()))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "input tensor:{} data quantization fail", tensor_name);
                    return -1;
                }
                if (nullptr != quant_data.get())
                {
                    buffer_data = (const char*)quant_data.get();
                    buffer_len = quant_len;
                }
            }

            // copy data to tensor
            if (!copyDataToTensor(tensor_name, buffer_data, buffer_len))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "copy data to tensor:{} fail", tensor_name);
                return -1;
            }
        }
        return 0;
    }

    int TimVXEngine::getOutputs(std::vector<TimvxOutput>& output_datas)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx graph is invalid, please create graph first");
            return -1;
        }
        auto output_tensors = m_graph->OutputsTensor();
        if (output_tensors.size() != output_datas.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "expect get {} outputs, but graph have {} outputs", 
                int(output_datas.size()), int(output_tensors.size()));
            return -1;
        }
        for (int i = 0; i < output_datas.size(); i++)
        {
            TimvxOutput output = output_datas[i];
            uint32_t tensor_index = output.index;
            auto out_tensor = output_tensors[tensor_index];
            std::string tensor_name = m_output_tensor_names[tensor_index];
            int tensor_size = getTensorByteSize(tensor_name);
            // init output ptr and output size
            char* tensor_buffer_data = (char*)output.buf;
            int tensor_buffer_len = output.size;
            char* output_buffer_data = tensor_buffer_data;
            int output_buffer_len = tensor_buffer_len;
            // malloc memory for output data if necessary
            std::shared_ptr<char> output_tensor_data;
            std::shared_ptr<char> output_data;
            int output_tensor_size = getTensorByteSize(tensor_name);
            if (output.is_prealloc)
            {
                int output_data_size = (output.want_float ? getTensorElemCount(tensor_name) * sizeof(float) : output_tensor_size);
                output_data = std::shared_ptr<char>(new char[output_data_size], std::default_delete<char []>());
                if (nullptr == output_data.get())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory for tensor output:{} fail", tensor_name);
                    return -1;
                }
                output.buf = (void*)output_data.get();
                output.size = output_data_size;
                output_buffer_data = output_data.get();
                output_buffer_len = output_data_size;
                m_output_datas[tensor_name] = output_data;
            }
            // prepare temp tensor data to store graph and compute float resutl form this
            if (output.want_float)
            {
                output_tensor_data = std::shared_ptr<char>(new char[output_tensor_size], std::default_delete<char []>());
                if (nullptr == output_tensor_data.get())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory for tensor output:{} fail", tensor_name);
                    return -1;
                }
                tensor_buffer_data = output_tensor_data.get();
                tensor_buffer_len = output_tensor_size;
            }
            // copy data from tensor
            if (!copyDataFromTensor(tensor_name, tensor_buffer_data, tensor_buffer_len))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "copy data from tensor:{} fail", tensor_name);
                return -1;
            }

            // if want float, convert tensor data to float data
            if (output.want_float)
            {
                if (0 != dequantTensorData(tensor_name, (uint8_t*)tensor_buffer_data, 
                    tensor_buffer_len, (float*)output_buffer_data))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "dequant form tensor:{} fail", tensor_name);
                    return -1;
                }
            }
        }
        return 0;
    }

    int TimVXEngine::getInputOutputNum(TimvxInputOutputNum& io_num)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx graph is invalid, please create graph first");
            return -1;
        }
        auto input_tensors = m_graph->InputsTensor();
        io_num.n_input = input_tensors.size();
        auto output_tensors = m_graph->OutputsTensor();
        io_num.n_output = output_tensors.size();
        return 0;
    }

    int TimVXEngine::getTensorAttr(const std::string& tensor_name, TimvxTensorAttr& tensor_info)
    {
        memset(&tensor_info, 0, sizeof(TimvxTensorAttr));
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx graph is invalid, please create graph first");
            return -1;
        }
        // set tensor name
        int name_len = 0;
        if (tensor_name.size() > TIMVX_MAX_NAME_LEN - 1)
            name_len = TIMVX_MAX_NAME_LEN - 1;
        else
            name_len = tensor_name.size();
        memset(tensor_info.name, 0, sizeof(tensor_info.name));
        memcpy(tensor_info.name, tensor_name.c_str(), name_len);

        // set tensor shape
        std::vector<uint32_t> tensor_shape = m_tensors[tensor_name]->GetShape();
        tensor_info.n_dims = tensor_shape.size();
        for (int i = 0; i < tensor_shape.size(); i++)
        {
            tensor_info.dims[i] = tensor_shape[i];
        }

        // set tensor fmt default is NHWC
        tensor_info.fmt = TIMVX_TENSOR_NHWC;

        // set tensor type
        DataType data_type = m_tensors[tensor_name]->GetDataType();
        if (-1 == convertToTimVxDataType(data_type, tensor_info.type))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "invlid tensor data type {}", (int)data_type);
            return -1;
        }

        // set element number / size 
        TensorSpec tensor_spec = m_tensors[tensor_name]->GetSpec();
        tensor_info.n_elems = getTensorElemCount(tensor_name);
        tensor_info.size = getTensorByteSize(tensor_name);
        // tensor_info.n_elems = tensor_spec.GetElementNum();
        // tensor_info.size = tensor_spec.GetByteSize();

        // set quant info
        Quantization tensor_quant = m_tensors[tensor_name]->GetQuantization();
        const std::vector<float> scales = tensor_quant.Scales();
        const std::vector<int> zp = tensor_quant.ZeroPoints();
        if (scales.size() > 0)
            tensor_info.scale = scales[0];
        if (zp.size() > 0)
            tensor_info.zp = zp[0];
        tensor_info.qnt_type = TimvxTensorQntType(tensor_quant.Type());

        return 0;
    }

    int TimVXEngine::getInputTensorAttr(int input_index, TimvxTensorAttr& tensor_attr)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx graph is invalid, please create graph first");
            return -1;
        }
        auto input_tensors = m_graph->InputsTensor();
        if (input_index < 0 || input_index >= input_tensors.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "want to get {}-th input tensor attr, only have {} input" , 
                input_index, int(input_tensors.size()));
            return -1;
        }
        if (m_input_tensor_names.size() != input_tensors.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "input tensor names number:{} not equal to graph input number:{}" , 
                m_input_tensor_names.size(), int(input_tensors.size()));
            return -1;
        }
        std::string tensor_name = m_input_tensor_names[input_index];
        if (0 != getTensorAttr(tensor_name, tensor_attr))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "get input:{} tesnor {} attr fail" , input_index, tensor_name);
            return -1;
        }
        tensor_attr.index = input_index;
        return 0;
    }

    int TimVXEngine::getOutputTensorAttr(int output_index, TimvxTensorAttr& tensor_attr)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "timvx graph is invalid, please create graph first");
            return -1;
        }
        auto output_tensors = m_graph->OutputsTensor();
        if (output_index < 0 || output_index >= output_tensors.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "want to get {}-th output tensor attr, only have {} output", 
                output_index, int(output_tensors.size()));
            return -1;
        }
        if (m_output_tensor_names.size() != output_tensors.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "output tensor names number:{} not equal to graph output number:{}" , 
                m_output_tensor_names.size(), int(output_tensors.size()));
            return -1;
        }
        std::string tensor_name = m_output_tensor_names[output_index];
        if (0 != getTensorAttr(tensor_name, tensor_attr))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "get output tesnor {} attr fail" , tensor_name);
            return -1;
        }
        tensor_attr.index = output_index;
        return 0;
    }

} //namespace TimVX