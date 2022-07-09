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

namespace TIMVX
{

    #define BITS_PER_BYTE 8
    TimVXEngine::TimVXEngine(const std::string &graph_name)
    {
        m_context.reset();
        m_graph.reset();
        m_graph_name = graph_name;
    }

    TimVXEngine::~TimVXEngine()
    {
        m_tensors_data.clear();
        m_tensors_spec.clear();
        m_tensors.clear();
        m_graph.reset();
        m_context.reset();
    }

    uint32_t TimVXEngine::typeGetBits(DataType type)
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

    size_t TimVXEngine::getTensorByteSize(const std::string &tensor_name)
    {
        size_t sz;
        size_t i;
        size_t bits_num;
        size_t dim_num;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("tensor %s not exists\n", tensor_name.c_str());
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
        bits_num = typeGetBits( type );
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

    size_t TimVXEngine::getTensorElemSize(const std::string &tensor_name)
    {
        size_t sz;
        size_t i;
        size_t bits_num;
        size_t dim_num;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("tensor %s not exists\n", tensor_name.c_str());
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

    int TimVXEngine::convertToDataType(TimvxTensorType tensor_type, DataType &type)
    {
        if (TIMVX_TENSOR_INT8 == tensor_type)
            type = DataType::UINT8;
        else if (TIMVX_TENSOR_UINT8 == tensor_type)
            type = DataType::UINT8;
        else if (TIMVX_TENSOR_INT16 == tensor_type)
            type = DataType::INT16;
        else if (TIMVX_TENSOR_FLOAT16 == tensor_type)
            type = DataType::FLOAT16;
        else if (TIMVX_TENSOR_FLOAT32 == tensor_type)
            type = DataType::FLOAT32;
        else
            return -1;
        return 0;
    }

    int TimVXEngine::getTensorInfo(const std::string &tensor_name, TimvxTensorAttr& tensor_info)
    {
        memset(&tensor_info, 0, sizeof(TimvxTensorAttr));
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        // set tensor name
        int name_len = 0;
        if (tensor_name.size() > TIMVX_MAX_NAME_LEN - 1)
            name_len = TIMVX_MAX_NAME_LEN - 1;
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
            TIMVX_ERROR("invlid tensor data type %d\n", (int)data_type);
            return -1;
        }

        // set element number / size 
        TensorSpec tensor_spec = m_tensors[tensor_name]->GetSpec();
        tensor_info.n_elems = tensor_spec.GetElementNum();
        tensor_info.size = tensor_spec.GetByteSize();

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

    int TimVXEngine::createTensor(const std::string &tensor_name, const json &tensor_info, 
        const char *weight_data, const int weight_len)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (m_tensors.find(tensor_name) != m_tensors.end())
        {
            TIMVX_ERROR("duplicate tensor name: %s is provided, please check again\n", tensor_name.c_str());
            return -1;
        }
        TensorSpec tensor_spec;
        if (!TensorSpecConstruct::constructTensorspec(tensor_info, tensor_name, tensor_spec))
        {
            TIMVX_ERROR("construct %s's tensor spec fail, please check again\n", tensor_name.c_str());
            return -1;
        }
        std::shared_ptr<Tensor> tensor;
        if (!tensor_info.contains("offset"))
            tensor = m_graph->CreateTensor(tensor_spec);
        else
        {
            if (!tensor_info["offset"].is_number_integer())
            {
                TIMVX_ERROR("tensor %s's offset item is not a valid integer type, please check again\n", tensor_name.c_str());
                return -1;
            }
            int offset = tensor_info.at("offset");
            int num_bytes = tensor_spec.GetByteSize();
            std::shared_ptr<char> data_array_ptr(new char[num_bytes], [](char* data_array_ptr){delete [] data_array_ptr;});
            m_tensors_data[tensor_name] = data_array_ptr;
            memcpy((void*)data_array_ptr.get(), weight_data + offset, num_bytes);
            tensor = m_graph->CreateTensor(tensor_spec, (void*)data_array_ptr.get());
        }
        if (nullptr == tensor.get())
        {
            TIMVX_ERROR("execute tensor %s's CreateTensor fail\n", tensor_name.c_str());
            return -1;
        }
        m_tensors[tensor_name] = tensor;
        return 0;
    }    

    int TimVXEngine::copyDataFromTensor(const std::string &tensor_name, char* buffer_data, const int buffer_data_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_ERROR("dest buffer data ptr is nullptr, when copy from tensor %s\n", tensor_name.c_str());
            return -1;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("tensor %s not exists\n", tensor_name.c_str());
            return -1;
        }
        auto tensor = m_tensors[tensor_name];
        size_t total_tensor_size = getTensorByteSize(tensor_name);
        if (total_tensor_size != buffer_data_len)
        {
            TIMVX_ERROR("tensor %s size:%d not equal to buffer data size:%d\n", tensor_name.c_str(),
                (int)total_tensor_size, buffer_data_len);
            return -1;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_ERROR("tensor %s size:%d not valid\n", tensor_name.c_str(), (int)total_tensor_size);
            return -1;
        }
        return tensor->CopyDataFromTensor(buffer_data);
    }

    int TimVXEngine::copyDataToTensor(const std::string &tensor_name, const char* buffer_data, 
        const int buffer_data_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_ERROR("src buffer data ptr is nullptr, when copy to tensor %s\n", tensor_name.c_str());
            return -1;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("tensor %s not exists\n", tensor_name.c_str());
            return -1;
        }
        auto tensor = m_tensors[tensor_name];
        int total_tensor_size = getTensorByteSize(tensor_name);
        if (total_tensor_size != buffer_data_len)
        {
            TIMVX_ERROR("tensor %s size:%d not equal to buffer data size:%d\n", tensor_name.c_str(),
                total_tensor_size, buffer_data_len);
            return -1;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_ERROR("tensor %s size:%d not valid\n", tensor_name.c_str(), total_tensor_size);
            return -1;
        }
        return tensor->CopyDataToTensor(buffer_data, buffer_data_len);
    }

    int TimVXEngine::createOperation(const json &op_info)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (!op_info.contains("op_name") || !op_info["op_name"].is_string())
        {
            TIMVX_ERROR("op_name item is not contained, or op_name is not string\n");
            return -1;
        }
        std::string op_name = op_info.at("op_name");
        if (m_operations.find(op_name) != m_operations.end())
        {
            TIMVX_ERROR("op_name %s is duplicate\n", op_name.c_str());
            return -1;
        }
        if (!op_info.contains("op_type") || !op_info["op_type"].is_string())
        {
            TIMVX_ERROR("%s's op_type item is not contained, or op_type is not string\n", op_name.c_str());
            return -1;
        }
        if (!op_info.contains("op_attr") || !op_info["op_attr"].is_object())
        {
            TIMVX_ERROR("%s's op_attr item is not contained, or op_attr is not dict\n", op_name.c_str());
            return -1;
        }
        if (op_info.contains("rounding_policy") || !op_info["rounding_policy"].is_object())
        {
            TIMVX_ERROR("%s's rounding_policy item is not contained, or rounding_policy is not dict\n", op_name.c_str());
            return -1;
        }

        std::string op_type = op_info.at("op_type");
        OpCreator* op_creator = TimVXOp::getInstance()->getOpCreator(op_type);
        if (nullptr == op_creator)
        {
            TIMVX_ERROR("op %s's creator not find, when create %s\n", op_type.c_str(), op_name.c_str());
            return -1;
        }
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
        if (nullptr != op_node)
        {
            m_operations[op_name] = op_node;
            return 0;
        }
        TIMVX_ERROR("create op %s fail\n", op_name.c_str());
        return -1;
    }

    int TimVXEngine::parseNormInfo(const json &norm_json)
    {
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
                    return -1;
                }
                mean_val = tensor_norm["mean"].get<std::vector<float>>();
                if (tensor_norm.contains("std") && !tensor_norm["std"].is_array())
                {
                    TIMVX_ERROR("para file's nodes should be array type\n");
                    return -1;
                }
                std_val = tensor_norm["std"].get<std::vector<float>>();
                if (tensor_norm.contains("reorder") && !tensor_norm["reorder"].is_array())
                {
                    TIMVX_ERROR("para file's nodes should be array type\n");
                    return -1;
                }
                reorder_val = tensor_norm["reorder"].get<std::vector<int>>();
            
                if (reorder_val.size() != 3)
                {
                    TIMVX_ERROR("norm info reorder's size should be 3\n");
                    return -1;
                }
                if ((reorder_val[0] != 2 || reorder_val[1] != 1 || reorder_val[2] != 0) ||
                    (reorder_val[0] != 0 || reorder_val[1] != 1 || reorder_val[2] != 2))
                {
                    TIMVX_ERROR("norm info reorder only support [0, 1, 2] or [2, 1, 0]\n");
                    return -1;
                }
                m_tensor_means[tensor_name] = mean_val;
                m_tensor_stds[tensor_name] = std_val;
                m_tensor_reorders[tensor_name] = reorder_val;
            }
        }
        catch(const std::exception& e)
        {
            TIMVX_ERROR("exception occur: %s\n", e.what());
            return -1;
        }
        return 0;
    }

    int TimVXEngine::bindInputs(const std::string &op_name, const std::vector<std::string> &input_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return -1;
        }
        if (input_list.size() <= 0)
        {
            TIMVX_ERROR("op %s's bind input list is empty\n", op_name.c_str());
            return -1;
        }
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        for (int i = 0; i < input_list.size(); i++)
        {
            std::string tensor_name = input_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_ERROR("op %s's input tensor %s not exists\n", op_name.c_str(), tensor_name.c_str());
                return -1;
            }
            input_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindInputs(input_tensors);
        return 0;
    }

    int TimVXEngine::bindOutputs(const std::string &op_name, const std::vector<std::string> &output_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return -1;
        }
        if (output_list.size() <= 0)
        {
            TIMVX_ERROR("op %s's bind output list is empty\n", op_name.c_str());
            return -1;
        }
        std::vector<std::shared_ptr<Tensor>> output_tensors;
        for (int i = 0; i < output_list.size(); i++)
        {
            std::string tensor_name = output_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_ERROR("op %s's output tensor %s not exists\n", op_name.c_str(), tensor_name.c_str());
                return -1;
            }
            output_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindOutputs(output_tensors);
        return 0;
    }

    int TimVXEngine::bindInput(const std::string &op_name, const std::string &input_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return -1;
        }
        if (m_tensors.find(input_name) == m_tensors.end())
        {
            TIMVX_ERROR("op %s's input tensor %s not exists\n", op_name.c_str(), input_name.c_str());
            return -1;
        }
        std::shared_ptr<Tensor> input_tensor = m_tensors[input_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindInput(input_tensor);
        return 0;
    }

    int TimVXEngine::bindOutput(const std::string &op_name, const std::string &output_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return -1;
        }
        if (m_tensors.find(output_name) == m_tensors.end())
        {
            TIMVX_ERROR("op %s's output tensor %s not exists\n", op_name.c_str(), output_name.c_str());
            return -1;
        }
        std::shared_ptr<Tensor> out_tensor = m_tensors[output_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindOutput(out_tensor);
        return 0;
    }

    int TimVXEngine::createGraph()
    {
        m_context = tim::vx::Context::Create();
        if (nullptr == m_context.get())
        {
            TIMVX_ERROR("create timvx context fail\n");
            return -1;
        }
        m_graph = m_context->CreateGraph();
        if (nullptr == m_graph.get())
        {
            TIMVX_ERROR("create timvx graph fail\n");
            m_context.reset();
            return -1;
        }
        return 0;
    }

    int TimVXEngine::compileGraph()
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        return m_graph->Compile();
    }

    int TimVXEngine::runGraph()
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        return m_graph->Run();
    }

    int TimVXEngine::setInputs(std::vector<TimvxInput> &input_data)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        if (input_data.size() != m_input_tensor_names.size())
        {
            TIMVX_PRINT("input data size %d not equalt to engine's input size %d\n",
                input_data.size(), m_input_tensor_names.size());
            return -1;
        }
        for (int i = 0; i < input_data.size(); i++)
        {
            TimvxInput input = input_data[i];
            std::string tensor_name = m_input_tensor_names[i];
            const char* buffer_data = (const char*)input.buf;
            int buffer_len = input.size;
            if (buffer_len % m_tensor_means[tensor_name].size())
            {
                TIMVX_ERROR("invalid input data size %d\n", buffer_len);
                return -1;
            }
            // norm data
            int norm_len = 0;
            std::shared_ptr<char> norm_data;
            if (!inputDataNorm(input, tensor_name, norm_data, norm_len))
            {
                TIMVX_ERROR("input data normalization fail\n");
                return -1;
            }
            if (nullptr != norm_data.get())
            {
                buffer_data = (const char*)norm_data.get();
                buffer_len = norm_len;
            }

            // quant data
            int quant_len = norm_len/sizeof(float);
            std::shared_ptr<char> quant_data = std::shared_ptr<char>(new char[quant_len], std::default_delete<char []>());
            if (nullptr == quant_data.get())
            {
                TIMVX_ERROR("malloc data for tensor %s quant out fail\n", tensor_name.c_str());
                return -1;
            }
            if (input.pass_through && nullptr != norm_data.get() && 
                !quantTensorData(tensor_name, (float*)norm_data.get(), quant_len, (uint8_t*)quant_data.get()))
            {
                TIMVX_ERROR("input data normalization fail\n");
                return -1;
            }
            if (nullptr != quant_data.get())
            {
                buffer_data = (const char*)quant_data.get();
                buffer_len = quant_len;
            }

            // copy data to tensor
            if (!copyDataToTensor(tensor_name, buffer_data, buffer_len))
            {
                TIMVX_ERROR("copy data to tensor %s fail\n", tensor_name.c_str());
                return -1;
            }
        }
        return 0;
    }

    int TimVXEngine::getOutputs(std::vector<TimvxOutput> &output_data)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return -1;
        }
        for (int i = 0; i < output_data.size(); i++)
        {
            TimvxOutput output = output_data[i];
            std::string tensor_name = m_output_tensor_names[i];
            int convert_len = 0;
            std::shared_ptr<char> convert_data;
            if (!outputDataConvert(output, tensor_name, convert_data, convert_len))
            {
                TIMVX_ERROR("output data convert fail\n");
                return -1;
            }
        }
        return 0;
    }

    int TimVXEngine::getInputOutputNum(TimvxInputOutputNum &io_num)
    {
        io_num.n_input = m_input_tensor_names.size();
        io_num.n_output = m_output_tensor_names.size();
        return 0;
    }

    int TimVXEngine::getInputTensorAttr(int input_index, TimvxTensorAttr &tensor_attr)
    {
        if (input_index < 0 || input_index >= m_input_tensor_names.size())
        {
            TIMVX_ERROR("input:%d is invalid, only have %d input\n", input_index, m_input_tensor_names.size());
            return -1;
        }
        std::string tensor_name = m_input_tensor_names[input_index];
        if (0 != getTensorInfo(tensor_name, tensor_attr))
        {
            TIMVX_ERROR("get input:%d tesnor %s attr fail\n", input_index, tensor_name.c_str());
            return -1;
        }
        tensor_attr.index = input_index;
        return 0;
    }

    int TimVXEngine::getOutputTensorAttr(int output_index, TimvxTensorAttr &tensor_attr)
    {
        if (output_index < 0 || output_index >= m_output_tensor_names.size())
        {
            TIMVX_ERROR("output:%d is invalid, only have %d output\n", output_index, m_output_tensor_names.size());
            return -1;
        }
        std::string tensor_name = m_output_tensor_names[output_index];
        if (0 != getTensorInfo(tensor_name, tensor_attr))
        {
            TIMVX_ERROR("get output tesnor %s attr fail\n", tensor_name.c_str());
            return -1;
        }
        tensor_attr.index = output_index;
        return 0;
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

    int TimVXEngine::inputDataMeanStd(char *input_data, const int input_len, float* process_data, std::vector<float> mean, std::vector<float> std)
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
            TIMVX_ERROR("current only support ASYMMETRIC quant type\n");
            return -1;
        }
        if (quant_info.Scales().size() != quant_info.ZeroPoints().size())
        {
            TIMVX_ERROR("invalid quant info for tensor %s , scales num not equal to zero_point\n", tensor_name.c_str());
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
            TIMVX_ERROR("current only support ASYMMETRIC quant type\n");
            return -1;
        }
        if (quant_info.Scales().size() != quant_info.ZeroPoints().size())
        {
            TIMVX_ERROR("invalid quant info for tensor %s , scales num not equal to zero_point\n", tensor_name.c_str());
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
            return 0;

        // channel reorder
        std::shared_ptr<char> reorder_data;
        if (0 != m_tensor_reorders[input_name].size())
        {
            char* src_data = (char*)input_data.buf;
            int src_len = input_data.size;
            reorder_data = std::shared_ptr<char>(new char[src_len], std::default_delete<char []>());
            if (nullptr == reorder_data.get())
            {
                TIMVX_ERROR("malloc data for tensor %s reorder out fail\n", input_name.c_str());
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
                TIMVX_ERROR("malloc data for tensor %s norm out fail\n", input_name.c_str());
                return -1;
            }
            if (0 != inputDataMeanStd(src_data, src_len, (float*)norm_data.get(), means, stds))
            {
                TIMVX_ERROR("input tensor (%s - mean) / std fail\n", input_name.c_str());
                return -1;
            }
            if (0 != inputDataTranspose<float>((float*)norm_data.get(), norm_len, ch_num, (float*)norm_data.get()))
            {
                TIMVX_ERROR("input tensor %s transpose fail\n", input_name.c_str());
                return -1;
            }
        }
        else
        {
            norm_len = src_len;
            norm_data = std::shared_ptr<char>(new char[src_len], std::default_delete<char []>());
            if (0 != inputDataTranspose<uint8_t>((uint8_t*)src_data, src_len, ch_num, (uint8_t*)norm_data.get()))
            {
                TIMVX_ERROR("input tensor %s transpose fail\n", input_name.c_str());
                return -1;
            }
        }

        return 0;
    }

    int TimVXEngine::outputDataConvert(TimvxOutput out_data, std::string output_name, 
        std::shared_ptr<char>& convert_data, int &convert_len)
    {
        int tensor_size = getTensorByteSize(output_name);
        std::shared_ptr<char> out_tensor_data;
        if (out_data.is_prealloc)
        {
            char* buffer_data = (char*)out_data.buf;
            int buffer_len = out_data.size;
            if (out_data.want_float)
            {
                out_tensor_data = std::shared_ptr<char>(new char[tensor_size], std::default_delete<char []>());
                if (nullptr == out_tensor_data.get())
                {
                    TIMVX_ERROR("malloc memory for tensor %s fail\n", output_name.c_str());
                    return -1;
                }
                buffer_data = out_tensor_data.get();
                buffer_len = getTensorByteSize(output_name);
            }
            if (!copyDataFromTensor(output_name, buffer_data, buffer_len))
            {
                TIMVX_ERROR("copy data from tensor %s fail\n", output_name.c_str());
                return -1;
            }
            if (out_data.want_float)
            {
                if (0 != dequantTensorData(output_name, (uint8_t*)out_tensor_data.get(), tensor_size, (float*)out_data.buf))
                {
                    TIMVX_ERROR("dequant form tensor %s fail\n", output_name.c_str());
                    return -1;
                }
            }
        }
        else
        {
            out_tensor_data = std::shared_ptr<char>(new char[tensor_size], std::default_delete<char []>());
            if (nullptr == out_tensor_data.get())
            {
                TIMVX_ERROR("malloc memory for tensor %s fail\n", output_name.c_str());
                return -1;
            }
            char* buffer_data = (char*)out_tensor_data.get();
            int buffer_len = tensor_size;
            if (!copyDataFromTensor(output_name, buffer_data, buffer_len))
            {
                TIMVX_ERROR("copy data from tensor %s fail\n", output_name.c_str());
                return -1;
            }
            if (!out_data.want_float)
                m_output_tensor_datas[output_name] = out_tensor_data;
            else
            {
                convert_len = getTensorElemSize(output_name) * sizeof(float);
                convert_data = std::shared_ptr<char>(new char[convert_len], std::default_delete<char []>());
                if (nullptr == convert_data.get())
                {
                    TIMVX_ERROR("malloc memory for tensor %s dequant data fail\n", output_name.c_str());
                    return -1;
                }
                if (0 != dequantTensorData(output_name, (uint8_t*)out_tensor_data.get(), 
                    tensor_size, (float*)convert_data.get()))
                {
                    TIMVX_ERROR("dequant form tensor %s fail\n", output_name.c_str());
                    return -1;
                }
                m_output_tensor_datas[output_name] = convert_data;
            }
            out_data.buf = (void*)m_output_tensor_datas[output_name].get();
        }
        return 0;
    }

} //namespace TIMVX