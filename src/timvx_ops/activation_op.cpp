/***********************************
******  activation_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/activations.h"
#include "timvx_ops/activation_op.h"

namespace TIMVXPY
{

    bool ActivationCreator::parsePreluAttr(const json &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_prelu";
        return parseValue<int>(op_info, full_op_name, "axis", op_attr.prelu.axis);
    }

    bool ActivationCreator::parseLeakyreluAttr(const json &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_leakyrelu";
        return parseValue<float>(op_info, full_op_name, "ratio", op_attr.leakyrelu.ratio);
    }

    bool ActivationCreator::parseLinearAttr(const json &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_linear";
        return parseValue<float>(op_info, full_op_name, "a", op_attr.linear.a)
            && parseValue<float>(op_info, full_op_name, "b", op_attr.linear.b, false);
    }

    bool ActivationCreator::parseGeluAttr(const json &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_gelu";
        return parseValue<bool>(op_info, full_op_name, "approximate", op_attr.gelu.approximate, false);
    }

    bool ActivationCreator::parseHardsigmoidAttr(const json &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_hardsigmoid";
        return parseValue<float>(op_info, full_op_name, "alpha", op_attr.hardsigmoid.alpha) &&
            parseValue<float>(op_info, full_op_name, "beta", op_attr.hardsigmoid.beta);
    }

    bool ActivationCreator::parseOpAttr(std::string op_type, const json &op_info, ActivationOpAttr &op_attr)
    {
        op_attr.gelu.approximate = true;
        op_attr.linear.b = 0.0f;
        if ("prelu" == op_type)
            return parse_preluAttr(op_info, op_attr);
        else if ("leakyrelu" == op_type)
            return parseLeakyreluAttr(op_info, op_attr);
        else if ("linear" == op_type)
            return parseLinearAttr(op_info, op_attr);
        else if ("gelu" == op_type)
            return parseGeluAttr(op_info, op_attr);
        else if ("hardsigmoid" == op_type)
            return parseHardsigmoidAttr(op_info, op_attr);
        else
            return true;
    }

    Operation* ActivationCreator::onCreate(std::shared_ptr<Graph> &graph, const json &op_info)
    {
        ActivationOpAttr op_attr;
        std::string activation_type;
        if (!parseValue<std::string>(op_info, m_op_name, "activation_type", activation_type))
            return nullptr;
        if (!parseOpAttr(activation_type, op_info, op_attr))
            return nullptr;
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
        else if ("Sigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::Sigmoid>().get();
        }
        else if ("Mish" == activation_type)
        {
            return graph->CreateOperation<ops::Mish>().get();
        }
        else if ("HardSigmoid" == activation_type)
        {
            float alpha = op_attr.hardsigmoid.alpha;
            float beta = op_attr.hardsigmoid.beta;
            return graph->CreateOperation<ops::HardSigmoid>(alpha, beta).get();
        }
        else if ("SoftRelu" == activation_type)
        {
            return graph->CreateOperation<ops::SoftRelu>().get();
        }
        else if ("HardSwish" == activation_type)
        {
            return graph->CreateOperation<ops::HardSwish>().get();
        }
        else if ("Swish" == activation_type)
        {
            return graph->CreateOperation<ops::Swish>().get();
        }
        else if ("Prelu" == activation_type)
        {
            int axis = op_attr.prelu.axis;
            return graph->CreateOperation<ops::Prelu>(axis).get();
        }        
        else if ("Tanh" == activation_type)
        {
            return graph->CreateOperation<ops::Tanh>().get();
        }
        else if ("LeakyRelu" == activation_type)
        {
            float ratio = op_attr.leakyrelu.ratio;
            return graph->CreateOperation<ops::LeakyRelu>(ratio).get();
        }
        else if ("Linear" == activation_type)
        {
            float a = op_attr.linear.a;
            float b = op_attr.linear.b;
            return graph->CreateOperation<ops::Linear>(a, b).get();
        }
        else if ("Gelu" == activation_type)
        {
            bool approximate = op_attr.gelu.approximate;
            return graph->CreateOperation<ops::Gelu>(approximate).get();
        }
        else
            TIMVX_ERROR("unsupported activation type: %s, when create %s op\n", activation_type.c_str(), m_op_name.c_str());
        return nullptr;
    }

    REGISTER_OP_CREATOR(ActivationCreator, Activation);
} // namespace TIMVXPY