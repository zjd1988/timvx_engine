/***********************************
******  activation_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/activations.h"
#include "timvx_ops/activation_op.h"

namespace TimVX
{

    bool ActivationOpCreator::parseReluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }

    bool ActivationOpCreator::parseRelu1Attr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }

    bool ActivationOpCreator::parseRelu6Attr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseEluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseTanhAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseSigmoidAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseHardSwishAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseMishAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseHardSigmoidAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }


    bool ActivationOpCreator::parseSoftReluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        return true;
    }

    bool ActivationOpCreator::parsePreluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_prelu";
        return parseValue<int>(op_info, full_op_name, "axis", op_attr.prelu.axis);
    }

    bool ActivationOpCreator::parseLeakyreluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_leakyrelu";
        return parseValue<float>(op_info, full_op_name, "ratio", op_attr.leakyrelu.ratio);
    }

    bool ActivationOpCreator::parseLinearAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_linear";
        return parseValue<float>(op_info, full_op_name, "a", op_attr.linear.a) && 
            parseValue<float>(op_info, full_op_name, "b", op_attr.linear.b, false);
    }

    // bool ActivationOpCreator::parseGeluAttr(const json& op_info, ActivationOpAttr& op_attr)
    // {
    //     std::string full_op_name = m_op_name + "_gelu";
    //     return parseValue<bool>(op_info, full_op_name, "approximate", op_attr.gelu.approximate, false);
    // }

    // bool ActivationOpCreator::parseHardsigmoidAttr(const json& op_info, ActivationOpAttr& op_attr)
    // {
    //     std::string full_op_name = m_op_name + "_hardsigmoid";
    //     return parseValue<float>(op_info, full_op_name, "alpha", op_attr.hardsigmoid.alpha) &&
    //         parseValue<float>(op_info, full_op_name, "beta", op_attr.hardsigmoid.beta);
    // }

    bool ActivationOpCreator::parseOpAttr(std::string op_type, const json& op_info, ActivationOpAttr& op_attr)
    {
        // op_attr.gelu.approximate = true;
        op_attr.linear.b = 0.0f;

        if ("Relu" == op_type)
            return parseReluAttr(op_info, op_attr);
        else if ("Relu1" == op_type)
            return parseRelu1Attr(op_info, op_attr);
        else if ("Relu6" == op_type)
            return parseRelu6Attr(op_info, op_attr);
        else if ("Elu" == op_type)
            return parseEluAttr(op_info, op_attr);
        else if ("Tanh" == op_type)
            return parseTanhAttr(op_info, op_attr);
        else if ("Sigmoid" == op_type)
            return parseSigmoidAttr(op_info, op_attr);
        else if ("HardSwish" == op_type)
            return parseHardSwishAttr(op_info, op_attr);
        else if ("Mish" == op_type)
            return parseMishAttr(op_info, op_attr);
        else if ("HardSigmoid" == op_type)
            return parseHardSigmoidAttr(op_info, op_attr);
        else if ("SoftRelu" == op_type)
            return parseSoftReluAttr(op_info, op_attr);
        else if ("Prelu" == op_type)
            return parsePreluAttr(op_info, op_attr);
        else if ("Leakyrelu" == op_type)
            return parseLeakyreluAttr(op_info, op_attr);
        else if ("Linear" == op_type)
            return parseLinearAttr(op_info, op_attr);
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported activation op type: {}", op_type);
        return false;
    }

    Operation* ActivationOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ActivationOpAttr op_attr;
        std::string activation_type;
        if (!parseValue<std::string>(op_info, m_op_name, "activation_type", activation_type))
            return nullptr;
        if (!parseOpAttr(activation_type, op_info, op_attr))
            return nullptr;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, activation_type);
        if ("Relu" == activation_type)
        {
            return graph->CreateOperation<ops::Relu>().get();
        }
        else if ("Relu1" == activation_type)
        {
            return graph->CreateOperation<ops::Relu1>().get();
        }
        else if ("Relu6" == activation_type)
        {
            return graph->CreateOperation<ops::Relu6>().get();
        }
        else if ("Elu" == activation_type)
        {
            return graph->CreateOperation<ops::Elu>().get();
        }
        else if ("Tanh" == activation_type)
        {
            return graph->CreateOperation<ops::Tanh>().get();
        }
        else if ("Sigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::Sigmoid>().get();
        }
        else if ("HardSwish" == activation_type)
        {
            return graph->CreateOperation<ops::HardSwish>().get();
        }
        else if ("Mish" == activation_type)
        {
            return graph->CreateOperation<ops::Mish>().get();
        }
        else if ("HardSigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::HardSigmoid>().get();
        }
        else if ("SoftRelu" == activation_type)
        {
            return graph->CreateOperation<ops::SoftRelu>().get();
        }
        else if ("Prelu" == activation_type)
        {
            int axis = op_attr.prelu.axis;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
            return graph->CreateOperation<ops::Prelu>(axis).get();
        }
        else if ("LeakyRelu" == activation_type)
        {
            float ratio = op_attr.leakyrelu.ratio;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ratio);
            return graph->CreateOperation<ops::LeakyRelu>(ratio).get();
        }
        else if ("Linear" == activation_type)
        {
            float a = op_attr.linear.a;
            float b = op_attr.linear.b;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, a);
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, b);
            return graph->CreateOperation<ops::Linear>(a, b).get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported activation op type: {}", activation_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(ActivationOpCreator, Activation);

} // namespace TimVX