/***********************************
******  localresponsenormalization_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/localresponsenormalization.h"
#include "timvx_ops/localresponsenormalization_op.h"

namespace TimVX
{

    bool LocalResponseNormalizationOpCreator::parseSizeAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "size", op_attr.size, false);
    }

    bool LocalResponseNormalizationOpCreator::parseAlphaAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "alpha", op_attr.alpha);
    }

    bool LocalResponseNormalizationOpCreator::parseBetaAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "beta", op_attr.beta);
    }

    bool LocalResponseNormalizationOpCreator::parseBiasAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "bias", op_attr.bias);
    }

    bool LocalResponseNormalizationOpCreator::parseAxisAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis, false);
    }

    bool LocalResponseNormalizationOpCreator::parseOpAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr)
    {
        return parseSizeAttr(op_info, op_attr) && parseAlphaAttr(op_info, op_attr) && 
            parseBetaAttr(op_info, op_attr) && parseBiasAttr(op_info, op_attr) && 
            parseAxisAttr(op_info, op_attr);
    }

    Operation* LocalResponseNormalizationOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        LocalResponseNormalizationOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t size = op_attr.size;
        float alpha = op_attr.alpha;
        float beta = op_attr.beta;
        float bias = op_attr.bias;
        int32_t axis = op_attr.axis;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, size);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, alpha);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, beta);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, bias);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
        return graph->CreateOperation<ops::LocalResponseNormalization>(size, alpha, beta, bias, axis).get();
    }

    REGISTER_OP_CREATOR(LocalResponseNormalizationOpCreator, LocalResponseNormalization);

} // namespace TimVX