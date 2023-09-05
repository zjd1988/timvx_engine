/***********************************
******  op_creator.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "timvx_ops/op_creator.h"

namespace TimVX
{

    std::map<std::string, PoolType> gStrToPoolTypeMap = {
        {"MAX",                          PoolType::MAX},
        {"AVG",                          PoolType::AVG},
        {"L2",                           PoolType::L2},
        {"AVG_ANDROID",                  PoolType::AVG_ANDROID},
    };

    std::map<PoolType, std::string> gPoolTypeToStrMap = {
        {PoolType::MAX,                  "MAX"},
        {PoolType::AVG,                  "AVG"},
        {PoolType::L2,                   "L2"},
        {PoolType::AVG_ANDROID,          "AVG_ANDROID"},
    };

    std::map<std::string, PadType> gStrToPadTypeMap = {
        {"NONE",                         PadType::NONE},
        {"AUTO",                         PadType::AUTO},
        {"VALID",                        PadType::VALID},
        {"SAME",                         PadType::SAME},
    };

    std::map<PadType, std::string> gPadTypeToStrMap = {
        {PadType::NONE,                  "NONE"},
        {PadType::AUTO,                  "AUTO"},
        {PadType::VALID,                 "VALID"},
        {PadType::SAME,                  "SAME"},
    };

    std::map<std::string, RoundType> gStrToRoundTypeMap = {
        {"CEILING",                      RoundType::CEILING},
        {"FLOOR",                        RoundType::FLOOR},
    };

    std::map<RoundType, std::string> gRoundTypeToStrMap = {
        {RoundType::CEILING,             "CEILING"},
        {RoundType::FLOOR,               "FLOOR"},
    };

    std::map<std::string, OverflowPolicy> gStrToOverflowPolicyMap = {
        {"WRAP",                         OverflowPolicy::WRAP},
        {"SATURATE",                     OverflowPolicy::SATURATE},
    };

    std::map<OverflowPolicy, std::string> gOverflowPolicyToStrMap = {
        {OverflowPolicy::WRAP,           "WRAP"},
        {OverflowPolicy::SATURATE,       "SATURATE"},
    };

    std::map<std::string, RoundingPolicy> gStrToRoundingPolicyMap = {
        {"TO_ZERO",                      RoundingPolicy::TO_ZERO},
        {"RTNE",                         RoundingPolicy::RTNE},
    };

    std::map<RoundingPolicy, std::string> gRoundingPolicyToStrMap = {
        {RoundingPolicy::TO_ZERO,        "TO_ZERO"},
        {RoundingPolicy::RTNE,           "RTNE"},
    };

    std::map<std::string, ResizeType> gStrToResizeTypeMap = {
        {"NEAREST_NEIGHBOR",             ResizeType::NEAREST_NEIGHBOR},
        {"BILINEAR",                     ResizeType::BILINEAR},
        {"AREA",                         ResizeType::AREA},
    };

    std::map<ResizeType, std::string> gResizeTypeToStrMap = {
        {ResizeType::NEAREST_NEIGHBOR,   "NEAREST_NEIGHBOR"},
        {ResizeType::BILINEAR,           "BILINEAR"},
        {ResizeType::AREA,               "AREA"},
    };

    std::map<std::string, DataLayout> gStrToDataLayoutMap = {
        {"ANY",                          DataLayout::ANY},
        {"WHCN",                         DataLayout::WHCN},
        {"CWHN",                         DataLayout::CWHN},
        {"IcWHOc",                       DataLayout::IcWHOc}, /*TF*/
        {"OcIcWH",                       DataLayout::OcIcWH}, /*TVM for classic conv2d in tflite model*/
        {"WHIcOc",                       DataLayout::WHIcOc}, /*TIM-VX default*/
    };

    std::map<DataLayout, std::string> gDataLayoutToStrMap = {
        {DataLayout::ANY,                "ANY"},
        {DataLayout::WHCN,               "WHCN"},
        {DataLayout::CWHN,               "CWHN"},
        {DataLayout::IcWHOc,             "IcWHOc"}, /*TF*/
        {DataLayout::OcIcWH,             "OcIcWH"}, /*TVM for classic conv2d in tflite model*/
        {DataLayout::WHIcOc,             "WHIcOc"}, /*TIM-VX default*/
    };

    bool OpCreator::parsePoolType(const json& op_info, const std::string& op_name, 
        const std::string& attr_name, PoolType& pool_type, bool necessary)
    {
        std::string pool_type_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, pool_type_str, necessary);
        if (parse_result)
        {
            if (gStrToPoolTypeMap.find(pool_type_str) != gStrToPoolTypeMap.end())
                pool_type = gStrToPoolTypeMap[pool_type_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} pool type", op_name.c_str(),
                    attr_name.c_str(), pool_type_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parsePadType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, PadType& pad_type, bool necessary)
    {
        std::string padding_type_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, padding_type_str, necessary);        
        if (parse_result)
        {
            if (gStrToPadTypeMap.find(padding_type_str) != gStrToPadTypeMap.end())
                pad_type = gStrToPadTypeMap[padding_type_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} padding type", op_name.c_str(),
                    attr_name.c_str(), padding_type_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }
    
    bool OpCreator::parseRoundType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, RoundType& round_type, bool necessary)
    {
        std::string round_type_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, round_type_str, necessary);        
        if (parse_result)
        {
            if (gStrToRoundTypeMap.find(round_type_str) != gStrToRoundTypeMap.end())
                round_type = gStrToRoundTypeMap[round_type_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} round type", op_name.c_str(),
                    attr_name.c_str(), round_type_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseOverflowPolicyType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, OverflowPolicy& overflow_policy_type, bool necessary)
    {
        std::string overflow_policy_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, overflow_policy_str, necessary);        
        if (parse_result)
        {
            if (gStrToOverflowPolicyMap.find(overflow_policy_str) != gStrToOverflowPolicyMap.end())
                overflow_policy_type = gStrToOverflowPolicyMap[overflow_policy_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} overflow policy type", op_name.c_str(),
                    attr_name.c_str(), overflow_policy_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseRoundingPolicyType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, RoundingPolicy& rounding_policy_type, bool necessary)
    {
        std::string rounding_policy_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, rounding_policy_str, necessary);        
        if (parse_result)
        {
            if (gStrToRoundingPolicyMap.find(rounding_policy_str) != gStrToRoundingPolicyMap.end())
                rounding_policy_type = gStrToRoundingPolicyMap[rounding_policy_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} rounding policy type", op_name.c_str(),
                    attr_name.c_str(), rounding_policy_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseResizeType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, ResizeType& resize_type, bool necessary)
    {
        std::string resize_type_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, resize_type_str, necessary);        
        if (parse_result)
        {
            if (gStrToResizeTypeMap.find(resize_type_str) != gStrToResizeTypeMap.end())
                resize_type = gStrToResizeTypeMap[resize_type_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} resize type", op_name.c_str(),
                    attr_name.c_str(), resize_type_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseDataLayoutType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, DataLayout& data_layout_type, bool necessary)
    {
        std::string data_layout_str;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, data_layout_str, necessary);        
        if (parse_result)
        {
            if (gStrToDataLayoutMap.find(data_layout_str) != gStrToDataLayoutMap.end())
                data_layout_type = gStrToDataLayoutMap[data_layout_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} not support {} data layout type", op_name.c_str(),
                    attr_name.c_str(), data_layout_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool TimVXOp::addCreator(std::string op_type, OpCreator* creator)
    {
        if (m_op_creator_map.find(op_type) != m_op_creator_map.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{} op_creator has already added to map", op_type.c_str());
            return false;
        }
        TIMVX_LOG(TIMVX_LEVEL_INFO, "add {} op_creator to map", op_type.c_str());
        m_op_creator_map.insert(std::make_pair(op_type, creator));
        return true;
    }

    extern void registerOps();
    OpCreator* TimVXOp::getOpCreator(std::string op_type)
    {
        registerOps();
        if (m_op_creator_map.find(op_type) != m_op_creator_map.end())
            return m_op_creator_map[op_type];
        else
            return nullptr;
    }

}  //namespace TimVX