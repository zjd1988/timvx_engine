/***********************************
******  spatial_transformer_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/spatial_transformer.h"
#include "timvx_ops/spatial_transformer_op.h"

namespace TimVX
{

    bool SpatialTransformerOpCreator::parseOutputHAttr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "output_h", op_attr.output_h);
    }

    bool SpatialTransformerOpCreator::parseOutputWAttr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "output_w", op_attr.output_w);
    }

    bool SpatialTransformerOpCreator::parseHasTheta11Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "has_theta_1_1", op_attr.has_theta_1_1);
    }

    bool SpatialTransformerOpCreator::parseHasTheta12Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "has_theta_1_2", op_attr.has_theta_1_2);
    }

    bool SpatialTransformerOpCreator::parseHasTheta13Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "has_theta_1_3", op_attr.has_theta_1_3);
    }

    bool SpatialTransformerOpCreator::parseHasTheta21Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "has_theta_2_1", op_attr.has_theta_2_1);
    }

    bool SpatialTransformerOpCreator::parseHasTheta22Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "has_theta_2_2", op_attr.has_theta_2_2);
    }

    bool SpatialTransformerOpCreator::parseHasTheta23Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<bool>(op_info, m_op_name, "has_theta_2_3", op_attr.has_theta_2_3);
    }

    bool SpatialTransformerOpCreator::parseTheta11Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "theta_1_1", op_attr.theta_1_1);
    }

    bool SpatialTransformerOpCreator::parseTheta12Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "theta_1_2", op_attr.theta_1_2);
    }

    bool SpatialTransformerOpCreator::parseTheta13Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "theta_1_3", op_attr.theta_1_3);
    }

    bool SpatialTransformerOpCreator::parseTheta21Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "theta_2_1", op_attr.theta_2_1);
    }

    bool SpatialTransformerOpCreator::parseTheta22Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "theta_2_2", op_attr.theta_2_2);
    }

    bool SpatialTransformerOpCreator::parseTheta23Attr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "theta_2_3", op_attr.theta_2_3);
    }

    bool SpatialTransformerOpCreator::parseOpAttr(const json& op_info, SpatialTransformerOpAttr& op_attr)
    {
        return parseOutputHAttr(op_info, op_attr) && parseOutputWAttr(op_info, op_attr) && 
            parseHasTheta11Attr(op_info, op_attr) && parseHasTheta12Attr(op_info, op_attr) && 
            parseHasTheta13Attr(op_info, op_attr) && parseHasTheta21Attr(op_info, op_attr) && 
            parseHasTheta22Attr(op_info, op_attr) && parseHasTheta23Attr(op_info, op_attr) && 
            parseTheta11Attr(op_info, op_attr) && parseTheta12Attr(op_info, op_attr) && 
            parseTheta13Attr(op_info, op_attr) && parseTheta21Attr(op_info, op_attr) && 
            parseTheta22Attr(op_info, op_attr) && parseTheta23Attr(op_info, op_attr);
    }

    Operation* SpatialTransformerOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SpatialTransformerOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t output_h      = op_attr.output_h;
        uint32_t output_w      = op_attr.output_h;
        bool     has_theta_1_1 = op_attr.has_theta_1_1;
        bool     has_theta_1_2 = op_attr.has_theta_1_2;
        bool     has_theta_1_3 = op_attr.has_theta_1_3;
        bool     has_theta_2_1 = op_attr.has_theta_2_1;
        bool     has_theta_2_2 = op_attr.has_theta_2_2;
        bool     has_theta_2_3 = op_attr.has_theta_2_3;
        float    theta_1_1     = op_attr.theta_1_1;
        float    theta_1_2     = op_attr.theta_1_2;
        float    theta_1_3     = op_attr.theta_1_3;
        float    theta_2_1     = op_attr.theta_2_1;
        float    theta_2_2     = op_attr.theta_2_2;
        float    theta_2_3     = op_attr.theta_2_3;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, output_h);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, output_w);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, has_theta_1_1);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, has_theta_1_2);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, has_theta_1_3);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, has_theta_2_1);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, has_theta_2_2);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, has_theta_2_3);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, theta_1_1);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, theta_1_2);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, theta_1_3);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, theta_2_1);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, theta_2_2);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, theta_2_3);
        return graph->CreateOperation<ops::SpatialTransformer>(output_h, output_w, 
            has_theta_1_1, has_theta_1_2, has_theta_1_3, 
            has_theta_2_1, has_theta_2_2, has_theta_2_3, 
            theta_1_1, theta_1_2, theta_1_3, 
            theta_2_1, theta_2_2, theta_2_3).get();
    }

    REGISTER_OP_CREATOR(SpatialTransformerOpCreator, SpatialTransformer);

} // namespace TimVX