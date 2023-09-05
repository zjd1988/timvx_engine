/***********************************
******  tile_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/tile.h"
#include "timvx_ops/tile_op.h"

namespace TimVX
{

    bool TileOpCreator::parseMultiplesAttr(const json& op_info, TileOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "multiples", op_attr.multiples);
    }

    bool TileOpCreator::parseOpAttr(const json& op_info, TileOpAttr& op_attr)
    {
        return parseMultiplesAttr(op_info, op_attr);
    }

    Operation* TileOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        TileOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> multiples = op_attr.multiples;

        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, multiples);
        return graph->CreateOperation<ops::Tile>(multiples).get();
    }

    REGISTER_OP_CREATOR(TileOpCreator, Tile);

} // namespace TimVX