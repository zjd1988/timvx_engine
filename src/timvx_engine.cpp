/***********************************
******  timvx_engine.cpp
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#include <mutex>
#include <iostream>
#include "timvx_engine.h"
#include "tensor_info.h"
#include "op_factory.h"

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

    size_t TimVXEngine::getTensorSize(const std::string &tensor_name)
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

    bool TimVXEngine::convertDataType(DataType type, TimvxTensorType& tensor_type)
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
            return false;
        return true;
    }

    bool TimVXEngine::getTensorInfo(const std::string &tensor_name, TimvxTensorAttr& tensor_info)
    {
        memset(&tensor_info, 0, sizeof(TimvxTensorAttr));
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        // set tensor name
        int name_len = 0;
        if (tensor_name.size() > TIMVX_MAX_NAME_LEN - 1)
            name_len = TIMVX_MAX_NAME_LEN - 1;
        memset(tensor_info.name, 0, sizeof(tensor_info.name));
        memcpy(tensor_info.name, tensor_name.c_str(), name_len);

        // set tensor shape
        std::vector<uint32_t> tensor_shape = m_tensors[tensor_name]->getShape();
        tensor_info.n_dims = tensor_shape.size();
        for (int i = 0; i < tensor_shape.size(); i++)
        {
            tensor_info.dims[i] = tensor_shape[i];
        }

        // set tensor fmt default is NHWC
        tensor_info.fmt = TIMVX_TENSOR_NHWC;

        // set tensor type
        DataType data_type = m_tensors[tensor_name]->GetDataType();
        if (false == convertDataType(data_type, tensor_info.type))
        {
            TIMVX_ERROR("invlid tensor data type %d\n", (int)data_type);
            return false;
        }

        // set element number / size 
        TensorSpec tensor_spec = m_tensors[tensor_name]->GetSpec();
        tensor_info.n_elems = tensor_spec.GetElementNum();
        tensor_info.size = tensor_spec.GetByteSize();

        // set quant info
        Quantization tensor_quant = m_tensors[tensor_name]->GetQuantization();
        const std::vector<float> scales = tensor_quant.Scales();
        const std::vector<float> zp = tensor_quant.ZeroPoints();
        if (scales.size() > 0)
            tensor_info.scale = sclaes[0];
        if (zp.size() > 0)
            tensor_info.zp = zp[0];
        tensor_info.qnt_type = timvx_tensor_qnt_type(tensor_quant.Type());

        return true;
    }

    bool TimVXEngine::createTensor(const std::string &tensor_name, const json &tensor_info, 
        const char *weight_data, const int weight_len)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        if (m_tensors.find(tensor_name) != m_tensors.end())
        {
            TIMVX_ERROR("duplicate tensor name: %s is provided, please check again\n", tensor_name.c_str());
            return false;
        }
        TensorSpec tensor_spec;
        if (!TensorSpecConstruct::constructTensorspec(tensor_info, tensor_name, tensor_spec))
        {
            TIMVX_ERROR("construct %s's tensor spec fail, please check again\n", tensor_name.c_str());
            return false;
        }
        std::shared_ptr<Tensor> tensor;
        if (!tensor_info.contains("offset"))
            tensor = m_graph->CreateTensor(tensor_spec);
        else
        {
            if (!tensor_info["offset"].is_number_integer())
            {
                TIMVX_ERROR("tensor %s's offset item is not a valid integer type, please check again\n", tensor_name.c_str());
                return false;
            }
            int offset = tensor_info.at("offset");
            int num_bytes = tensor_spec.GetByteSize();
            std::shared_ptr<char> data_array_ptr(new char[num_bytes], [](char* data_array_ptr){delete [] data_array_ptr;});
            m_tensors_data[tensor_name] = data_array_ptr;
            memcpy((void*)data_array_ptr.get(), data_array.data(), num_bytes);
            tensor = m_graph->CreateTensor(tensor_spec, (void*)data_array_ptr.get());
        }
        if (nullptr == tensor.get())
        {
            TIMVX_ERROR("execute tensor %s's CreateTensor fail\n", tensor_name.c_str());
            return false;
        }
        m_tensors[tensor_name] = tensor;
        return true;
    }    

    bool TimVXEngine::copyDataFromTensor(const std::string &tensor_name, char* buffer_data, const int buffer_data_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_ERROR("dest buffer data ptr is nullptr, when copy from tensor %s\n", tensor_name.c_str());
            return false;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("tensor %s not exists\n", tensor_name.c_str());
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        size_t total_tensor_size = getTensorSize(tensor_name);
        if (total_tensor_size != buffer_data_len)
        {
            TIMVX_ERROR("tensor %s size:%d not equal to buffer data size:%d\n", tensor_name.c_str(),
                total_tensor_size, buffer_data_len);
            return false;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_ERROR("tensor %s size:%d not valid\n", tensor_name.c_str(), total_tensor_size);
            return false;
        }
        return tensor->CopyDataFromTensor(buffer_data);
    }

    bool TimVXEngine::copyDataToTensor(const std::string &tensor_name, const char* buffer_data, 
        const int buffer_data_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_ERROR("src buffer data ptr is nullptr, when copy to tensor %s\n", tensor_name.c_str());
            return false;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_ERROR("tensor %s not exists\n", tensor_name.c_str());
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        int total_tensor_size = getTensorSize(tensor_name);
        if (total_tensor_size != buffer_data_len)
        {
            TIMVX_ERROR("tensor %s size:%d not equal to buffer data size:%d\n", tensor_name.c_str(),
                total_tensor_size, buffer_data_len);
            return false;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_ERROR("tensor %s size:%d not valid\n", tensor_name.c_str(), total_tensor_size);
            return false;
        }
        return tensor->CopyDataToTensor(buf.ptr, buffer_data_len);
    }

    bool TimVXEngine::createOperation(const json &op_info)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        if (!op_info.contains("op_name") || !op_info["op_name"].is_string())
        {
            TIMVX_ERROR("op_name item is not contained, or op_name is not string\n");
            return false;
        }
        std::string op_name = op_info.at("op_name");
        if (m_operations.find(op_name) != m_operations.end())
        {
            TIMVX_ERROR("op_name %s is duplicate\n", op_name.c_str());
            return false;
        }
        if (!op_info.contains("op_type") || !op_info["op_type"].is_string())
        {
            TIMVX_ERROR("%s's op_type item is not contained, or op_type is not string\n", op_name.c_str());
            return false;
        }
        if (!op_info.contains("op_attr") || !op_info["op_attr"].is_object())
        {
            TIMVX_ERROR("%s's op_attr item is not contained, or op_attr is not dict\n", op_name.c_str());
            return false;
        }
        if (op_info.contains("rounding_policy") || !op_info["rounding_policy"].is_object())
        {
            TIMVX_ERROR("%s's rounding_policy item is not contained, or rounding_policy is not dict\n", op_name.c_str());
            return false;
        }

        std::string op_type = op_info.at("op_type");
        OpCreator* op_creator = TimVXOp::getInstance()->getCreator(op_type);
        if (nullptr == op_creator)
        {
            TIMVX_ERROR("op %s's creator not find, when create %s\n", op_type.c_str(), op_name.c_str());
            return false;
        }
        auto op_node = op_creator->onCreate(m_graph, op_info["op_attr"]);
        if (nullptr != op_node && op_info.contains("rounding_policy"))
        {
            json rounding_policy = op_info["rounding_policy"];
            OverflowPolicy overflow_policy_type = OverflowPolicy::SATURATE;
            RoundingPolicy rounding_policy_type = RoundingPolicy::RTNE;
            RoundType      round_type           = RoundType::FLOOR;
            uint32_t       accumulator_bits     = 0;
            op_creator->parseOverflowPolicyType(rounding_policy, op_name, "overflow_policy", overflow_policy_type, false);
            op_creator->parseRoundingPolicyType(rounding_policy, op_name, "rounding_policy", rounding_policy_type, false);
            op_creator->parseRound_type(rounding_policy, op_name, "down_scale_size_rounding", round_type, false);
            op_creator->parse_value<py::int_, uint>(rounding_policy, op_name, "accumulator_bits", accumulator_bits, false);
            op_node->SetRoundingPolicy(overflow_policy_type, rounding_policy_type, round_type, accumulator_bits);
        }
        if (nullptr != op_node)
        {
            m_operations[op_name] = op_node;
            return true;
        }
        TIMVX_ERROR("create op %s fail\n", op_name.c_str());
        return false;
    }

    bool TimVXEngine::bindInputs(const std::string &op_name, const std::vector<std::string> &input_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return false;
        }
        if (input_list.size() <= 0)
        {
            TIMVX_ERROR("op %s's bind input list is empty\n", op_name.c_str());
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        for (int i = 0; i < input_list.size(); i++)
        {
            std::string tensor_name = input_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_ERROR("op %s's input tensor %s not exists\n", op_name.c_str(), tensor_name.c_str());
                return false;
            }
            input_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindInputs(input_tensors);
        return true;
    }

    bool TimVXEngine::bindOutputs(const std::string &op_name, const std::vector<std::string> &output_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return false;
        }
        if (output_list.size() <= 0)
        {
            TIMVX_ERROR("op %s's bind output list is empty\n", op_name.c_str());
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> output_tensors;
        for (int i = 0; i < output_list.size(); i++)
        {
            std::string tensor_name = output_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_ERROR("op %s's output tensor %s not exists\n", op_name.c_str(), tensor_name.c_str());
                return false;
            }
            output_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindOutputs(output_tensors);
        return true;
    }

    bool TimVXEngine::bindInput(const std::string &op_name, const std::string &input_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return false;
        }
        if (m_tensors.find(input_name) == m_tensors.end())
        {
            TIMVX_ERROR("op %s's input tensor %s not exists\n", op_name.c_str(), tensor_name.c_str());
            return false;
        }
        std::shared_ptr<Tensor> input_tensor = m_tensors[input_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindInput(input_tensor);
        return true;
    }

    bool TimVXEngine::bindOutput(const std::string &op_name, const std::string &output_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_ERROR("op %s not exists\n", op_name.c_str());
            return false;
        }
        if (m_tensors.find(output_name) == m_tensors.end())
        {
            TIMVX_ERROR("op %s's output tensor %s not exists\n", op_name.c_str(), tensor_name.c_str());
            return false;
        }
        std::shared_ptr<Tensor> out_tensor = m_tensors[output_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindOutput(out_tensor);
        return true;
    }

    bool TimVXEngine::createGraph()
    {
        m_context = tim::vx::Context::Create();
        if (nullptr == m_context.get())
        {
            TIMVX_ERROR("create timvx context fail\n");
            return false;
        }
        m_graph = m_context->CreateGraph();
        if (nullptr == m_graph.get())
        {
            TIMVX_ERROR("create timvx graph fail\n");
            m_context.reset();
            return false;
        }
        return true;
    }

    bool TimVXEngine::compileGraph()
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        return m_graph->Compile();
    }

    bool TimVXEngine::runGraph()
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_ERROR("timvx graph is invalid, please create graph first\n");
            return false;
        }
        return m_graph->Run();
    }

} //namespace TIMVX