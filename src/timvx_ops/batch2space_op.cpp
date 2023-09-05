/***********************************
******  batch2space_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/batch2space.h"
#include "timvx_ops/batch2space_op.h"

namespace TimVX
{

    bool Batch2SpaceOpCreator::parseLayoutAttr(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool Batch2SpaceOpCreator::parseBlockSizeAttr(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "block_size", op_attr.block_size);
    }

    bool Batch2SpaceOpCreator::parseCropAttr(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "crop", op_attr.crop);
    }

    bool Batch2SpaceOpCreator::parseOpAttr(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parseLayoutAttr(op_info, op_attr) && parseBlockSizeAttr(op_info, op_attr) && 
            parseCropAttr(op_info, op_attr);
    }

    Operation* Batch2SpaceOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Batch2SpaceOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> block_size = op_attr.block_size;
        std::vector<int32_t> crop = op_attr.crop;
        DataLayout layout = op_attr.layout;
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, block_size);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, crop);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, layout, gDataLayoutToStrMap[layout]);
        return graph->CreateOperation<ops::Batch2Space>(block_size, crop, layout).get();
    }

    REGISTER_OP_CREATOR(Batch2SpaceOpCreator, Batch2Space);

} // namespace TimVX