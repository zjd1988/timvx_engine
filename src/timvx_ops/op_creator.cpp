/***********************************
******  op_creator.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "op_creator.h"

namespace TIMVXPY
{

    bool OpCreator::parsePoolType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, PoolType &pool_type, bool necessary)
    {
        std::string pool_type_str;
        std::map<std::string, PoolType>   pool_type_map;
        pool_type_map["MAX"]         = PoolType::MAX;
        pool_type_map["AVG"]         = PoolType::AVG;
        pool_type_map["L2"]          = PoolType::L2;
        pool_type_map["AVG_ANDROID"] = PoolType::AVG_ANDROID;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, pool_type_str, necessary);
        if (parse_result && necessary)
        {
            if (pool_type_map.find(pool_type_str) != pool_type_map.end())
                pool_type = pool_type_map[pool_type_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << pool_type_str << " pool type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parse_pad_type(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, PadType &pad_type, bool necessary)
    {
        std::string padding_type_str;
        std::map<std::string, PadType>   padding_map;
        padding_map["NONE"]  = PadType::NONE;
        padding_map["AUTO"]  = PadType::AUTO;
        padding_map["VALID"] = PadType::VALID;
        padding_map["SAME"]  = PadType::SAME;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, padding_type_str, necessary);        
        if (parse_result && necessary)
        {
            if (padding_map.find(padding_type_str) != padding_map.end())
                pad_type = padding_map[padding_type_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << padding_type_str << " padding type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }
    
    bool OpCreator::parseRoundType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, RoundType &round_type, bool necessary)
    {
        std::string round_type_str;
        std::map<std::string, RoundType>   round_type_map;
        round_type_map["CEILING"]    = RoundType::CEILING;
        round_type_map["FLOOR"]      = RoundType::FLOOR;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, round_type_str, necessary);        
        if (parse_result && necessary)
        {
            if (round_type_map.find(round_type_str) != round_type_map.end())
                round_type = round_type_map[round_type_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << round_type_str << " round type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }


    bool OpCreator::parseOverflowPolicyType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, OverflowPolicy &overflow_policy_type, bool necessary)
    {
        std::string overflow_policy_str;
        std::map<std::string, OverflowPolicy>   overflow_policy_map;
        overflow_policy_map["WRAP"]       = OverflowPolicy::WRAP;
        overflow_policy_map["SATURATE"]   = OverflowPolicy::SATURATE;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, overflow_policy_str, necessary);        
        if (parse_result && necessary)
        {
            if (overflow_policy_map.find(overflow_policy_str) != overflow_policy_map.end())
                overflow_policy_type = overflow_policy_map[overflow_policy_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << overflow_policy_str << " overflow policy type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseRoundingPolicyType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, RoundingPolicy &rounding_policy_type, bool necessary)
    {
        std::string rounding_policy_str;
        std::map<std::string, RoundingPolicy>   rounding_policy_map;
        rounding_policy_map["TO_ZERO"]    = RoundingPolicy::TO_ZERO;
        rounding_policy_map["RTNE"]       = RoundingPolicy::RTNE;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, rounding_policy_str, necessary);        
        if (parse_result && necessary)
        {
            if (rounding_policy_map.find(rounding_policy_str) != rounding_policy_map.end())
                rounding_policy_type = rounding_policy_map[rounding_policy_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << rounding_policy_str << " rounding policy type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseResizeType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, ResizeType &resize_type, bool necessary)
    {
        std::string resize_type_str;
        std::map<std::string, ResizeType>   resize_type_map;
        resize_type_map["NEAREST_NEIGHBOR"]    = ResizeType::NEAREST_NEIGHBOR;
        resize_type_map["BILINEAR"]            = ResizeType::BILINEAR;
        resize_type_map["AREA"]                = ResizeType::AREA;
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, resize_type_str, necessary);        
        if (parse_result && necessary)
        {
            if (resize_type_map.find(resize_type_str) != resize_type_map.end())
                resize_type = resize_type_map[resize_type_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << resize_type_str << " resize type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool OpCreator::parseDataLayoutType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, DataLayout &data_layout_type, bool necessary)
    {
        std::string data_layout_str;
        std::map<std::string, DataLayout>   data_layout_map;
        data_layout_map["ANY"]    = DataLayout::ANY;
        data_layout_map["WHCN"]   = DataLayout::WHCN;
        data_layout_map["CWHN"]   = DataLayout::CWHN;
        data_layout_map["IcWHOc"] = DataLayout::IcWHOc;    /*TF*/
        data_layout_map["OcIcWH"] = DataLayout::OcIcWH;    /*TVM for classic conv2d in tflite model*/
        data_layout_map["IcOcWH"] = DataLayout::IcOcWH;    /*TVM for depthwise conv2d in tflite model*/
        data_layout_map["WHIcOc"] = DataLayout::WHIcOc;    /*TIM-VX default*/
        data_layout_map["WCN"]    = DataLayout::WCN;       /*for conv1d*/
        data_layout_map["WIcOc"]  = DataLayout::WIcOc;     /*for conv1d*/        
        const char* attr_c_name = attr_name.c_str();
        bool parse_result = parseValue<std::string>(op_info, op_name, attr_name, data_layout_str, necessary);        
        if (parse_result && necessary)
        {
            if (data_layout_map.find(data_layout_str) != data_layout_map.end())
                data_layout_type = data_layout_map[data_layout_str];
            else
            {
                std::cout << op_name << " op's attr " << attr_name << " not support " 
                    << data_layout_str << " data layout type!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool TimVXOp::addCreator(std::string op_type, OpCreator* creator)
    {
        if (op_creator.find(op_type) != op_creator.end())
        {
            std::cout << op_type << " type has be added to op_creator!" << std::endl;
            return false;
        }
        std::cout << "add " << op_type << " op_creator to map!" << std::endl;
        op_creator.insert(std::make_pair(op_type, creator));
        return true;
    }

    OpCreator* TimVXOp::getCreator(std::string op_type)
    {
        if (op_creator.find(op_type) != op_creator.end())
            return op_creator[op_type];
        else
            return nullptr;
    }

    static std::map<std::string, const OpCreator*>& getOpCreatorMap()
    {
        static std::once_flag gInitFlag;
        static std::map<std::string, const OpCreator*>* gOpCreator;
        std::call_once(gInitFlag,
                    [&]() { gOpCreator = new std::map<std::string, const OpCreator*>; });
        return *gOpCreator;
    }

    extern void registerOps();
    const ModelCreator* getOpCreator(std::string op_type)
    {
        registerModel();
        auto& gOpCreator = getOpCreatorMap();
        auto iter           = gOpCreator.find(op_type);
        if (iter == gOpCreator.end())
        {
            return nullptr;
        }
        if (iter->second)
        {
            return iter->second;
        }
        return nullptr;
    }

    bool insertModelCreator(ModelType model_type, const ModelCreator* creator)
    {
        auto& gModelCreator = getModelCreatorMap();
        std::string model_type_str = gModelTypeStrMap[model_type];
        if (gModelCreator.find(model_type) != gModelCreator.end())
        {
            CUSTOM_LOG(CUSTOM_LOG_LEVEL_ERROR, "duplicate register {} creator", model_type_str);
            return false;
        }
        CUSTOM_LOG(CUSTOM_LOG_LEVEL_INFO, "insert {} creator success", model_type_str);
        gModelCreator.insert(std::make_pair(model_type, creator));
        return true;
    }

}  //namespace TIMVXPY