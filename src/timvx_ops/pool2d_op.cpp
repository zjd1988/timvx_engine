/***********************************
******  pool2d_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/pool2d.h"
#include "timvx_ops/pool2d_op.h"

namespace TimVX
{

    Pool2dOpCreator::Pool2dCfgType Pool2dOpCreator::getPool2dType(const json& op_info)
    {
        if (op_info.contains("padding") && op_info.contains("pad"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "cannot contain padding and pad same time!");
            return Pool2dCfgType::None;
        }
        if ((op_info.contains("padding") || op_info.contains("pad"))
            && op_info.contains("input_size"))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "cannot contain padding(pad) and input_size same time!");
            return Pool2dCfgType::None;
        }
        if (op_info.contains("type") && op_info.contains("padding") &&
            op_info.contains("ksize") && op_info.contains("stride"))
            return Pool2dCfgType::Classic_Pool2d_1;
        else if (op_info.contains("type") && op_info.contains("pad") &&
            op_info.contains("ksize") && op_info.contains("stride"))
            return Pool2dCfgType::Classic_Pool2d_2;
        else if (op_info.contains("type") && op_info.contains("input_size") &&
            !op_info.contains("ksize"))
            return Pool2dCfgType::Global_Pool2d;
        else if (op_info.contains("type") && op_info.contains("input_size") &&
            op_info.contains("output_size"))
            return Pool2dCfgType::Adaptive_Pool2d;
        else
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "invalid pool2d op attr!");
            return Pool2dCfgType::None;
        }
    }

    bool Pool2dOpCreator::parseTypeAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parsePoolType(op_info, m_op_name, "type", op_attr.type);
    }

    bool Pool2dOpCreator::parsePaddingAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding);
    }

    bool Pool2dOpCreator::parsePadAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad);
    }

    bool Pool2dOpCreator::parseKsizeAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize);
    }

    bool Pool2dOpCreator::parseStrideAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Pool2dOpCreator::parseInputSizeAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "input_size", op_attr.input_size);
    }

    bool Pool2dOpCreator::parseOutputSizeAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "output_size", op_attr.output_size);
    }

    bool Pool2dOpCreator::parseRoundTypeAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return OpCreator::parseRoundType(op_info, m_op_name, "round_type", op_attr.round_type, false);
    }

    bool Pool2dOpCreator::parseLayoutAttr(const json& op_info, Pool2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool Pool2dOpCreator::parseOpAttr(const json& op_info, Pool2dOpAttr& op_attr, Pool2dCfgType cfg_type)
    {
        op_attr.round_type = RoundType::FLOOR;
        op_attr.layout = DataLayout::WHCN;
        if (Classic_Pool2d_1 == cfg_type)
            return parseTypeAttr(op_info, op_attr) && parsePaddingAttr(op_info, op_attr) && 
                parseKsizeAttr(op_info, op_attr) && parseStrideAttr(op_info, op_attr) && 
                parseRoundTypeAttr(op_info, op_attr) && parseLayoutAttr(op_info, op_attr);

        else if (Classic_Pool2d_2 == cfg_type)
            return parseTypeAttr(op_info, op_attr) && parsePadAttr(op_info, op_attr) && 
                parseKsizeAttr(op_info, op_attr) && parseStrideAttr(op_info, op_attr) && 
                parseRoundTypeAttr(op_info, op_attr) && parseLayoutAttr(op_info, op_attr);

        else if (Global_Pool2d == cfg_type)
            return parseTypeAttr(op_info, op_attr) && parseInputSizeAttr(op_info, op_attr) && 
                parseRoundTypeAttr(op_info, op_attr) && parseLayoutAttr(op_info, op_attr);

        else if (Adaptive_Pool2d == cfg_type)
            return parseTypeAttr(op_info, op_attr) && parseInputSizeAttr(op_info, op_attr) && 
                parseOutputSizeAttr(op_info, op_attr) && parseRoundTypeAttr(op_info, op_attr) && 
                parseLayoutAttr(op_info, op_attr);
        else
            return false;
    }

    Operation* Pool2dOpCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        std::map<Pool2dCfgType, std::string> pool_cfg_type_map;
        pool_cfg_type_map[Classic_Pool2d_1] = "Classic_Pool2d_1";
        pool_cfg_type_map[Classic_Pool2d_2] = "Classic_Pool2d_2";
        pool_cfg_type_map[Global_Pool2d] = "Global_Pool2d";
        pool_cfg_type_map[Adaptive_Pool2d] = "Adaptive_Pool2d";
        Pool2dCfgType cfg_type;
        Pool2dOpAttr op_attr;
        cfg_type = getPool2dType(op_info);
        if (pool_cfg_type_map.find(cfg_type) == pool_cfg_type_map.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported pool cfg, please check!");
            return nullptr;
        }
        if (!parseOpAttr(op_info, op_attr, cfg_type))
            return nullptr;

        PoolType type                       = op_attr.type;
        PadType padding                     = op_attr.padding;
        std::array<uint32_t, 2> ksize       = op_attr.ksize;
        std::array<uint32_t, 2> stride      = op_attr.stride;
        RoundType round_type                = op_attr.round_type;
        DataLayout layout                   = op_attr.layout;
        std::array<uint32_t, 4> pad         = op_attr.pad;
        // std::array<uint32_t, 2> input_size  = op_attr.input_size;
        // std::array<uint32_t, 2> output_size = op_attr.output_size;
        std::array<uint32_t, 2> input_size  = {0, 0};
        std::array<uint32_t, 2> output_size = {0, 0};

        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, cfg_type, pool_cfg_type_map[cfg_type]);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, type, gPoolTypeToStrMap[type]);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, padding, gPadTypeToStrMap[padding]);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ksize);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, stride);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, round_type, gRoundTypeToStrMap[round_type]);
        TIMVX_LOG_MAP_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, layout, gDataLayoutToStrMap[layout]);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, pad);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, input_size);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, output_size);
        switch (cfg_type)
        {
            case Classic_Pool2d_1:
                return graph->CreateOperation<ops::Pool2d>(type, padding,
                    ksize, stride, round_type, layout).get();
            case Classic_Pool2d_2:
                return graph->CreateOperation<ops::Pool2d>(type, pad,
                    ksize, stride, round_type, layout).get();
            default:
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "get invalid pool2d type!");
                return nullptr;
        }
    }

    REGISTER_OP_CREATOR(Pool2dOpCreator, Pool2d);

} // namespace TimVX