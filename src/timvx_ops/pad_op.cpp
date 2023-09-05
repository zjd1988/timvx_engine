/***********************************
******  pad_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/pad.h"
#include "timvx_ops/pad_op.h"

namespace TimVX
{

    bool PadOpCreator::parseFrontSizeAttr(const json& op_info, PadOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "front_size", op_attr.front_size);
    }

    bool PadOpCreator::parseBackSizeAttr(const json& op_info, PadOpAttr& op_attr)
    {
        return parseDynamicList<uint32_t>(op_info, m_op_name, "back_size", op_attr.back_size);
    }

    bool PadOpCreator::parseConstValAttr(const json& op_info, PadOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "const_val", op_attr.const_val);
    }

    bool PadOpCreator::parseOpAttr(const json& op_info, PadOpAttr& op_attr)
    {
        return parseFrontSizeAttr(op_info, op_attr) && parseBackSizeAttr(op_info, op_attr) && 
            parseConstValAttr(op_info, op_attr);
    }

    Operation* PadOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        PadOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<uint32_t> front_size = op_attr.front_size;
        std::vector<uint32_t> back_size  = op_attr.back_size;
        int32_t               const_val  = op_attr.const_val;
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, front_size);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, back_size);
        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, const_val);
        return graph->CreateOperation<ops::Pad>(front_size, back_size, const_val).get();
    }

    REGISTER_OP_CREATOR(PadOpCreator, Pad);

} // namespace TimVX