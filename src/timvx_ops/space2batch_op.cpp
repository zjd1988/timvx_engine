/***********************************
******  space2batch_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/space2batch.h"
#include "timvx_ops/space2batch_op.h"

namespace TimVX
{

    bool Space2BatchOpCreator::parseBlockSizeAttr(const json& op_info, Space2BatchOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "block_size", op_attr.block_size);
    }

    bool Space2BatchOpCreator::parsePadAttr(const json& op_info, Space2BatchOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "pad", op_attr.pad);
    }

    bool Space2BatchOpCreator::parseLayoutAttr(const json& op_info, Space2BatchOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool Space2BatchOpCreator::parseOpAttr(const json& op_info, Space2BatchOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN; // always set WHCN
        return parseBlockSizeAttr(op_info, op_attr) && parseLayoutAttr(op_info, op_attr) && 
            parseLayoutAttr(op_info, op_attr);
    }

    Operation* Space2BatchOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Space2BatchOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> block_size = op_attr.block_size;
        std::vector<int32_t> pad        = op_attr.pad;
        DataLayout layout               = op_attr.layout;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, block_size);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, pad);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, layout, gDataLayoutToStrMap[layout]);
        return graph->CreateOperation<ops::Space2Batch>(block_size, pad, layout).get();
    }

    REGISTER_OP_CREATOR(Space2BatchOpCreator, Space2Batch);

} // namespace TimVX