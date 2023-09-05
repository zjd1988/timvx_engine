/***********************************
******  op_creator.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include <map>
#include <iostream>
#include <mutex>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/types.h"
#include "common/timvx_log.h"
#include "nlohmann/json.hpp"
using namespace tim::vx;
using namespace nlohmann;
using namespace std;

namespace TimVX
{

#define TIMVX_LOG_BASE_DATATYPE_ATTR(LEVEL, ATTR_NAME) \
    TIMVX_LOG(LEVEL, "{:>20}: {}", #ATTR_NAME, ATTR_NAME)

#define TIMVX_LOG_STL_DATATYPE_ATTR(LEVEL, ATTR_NAME) \
    TIMVX_LOG(LEVEL, "{:>20}: {}", #ATTR_NAME, spdlog::fmt_lib::join(ATTR_NAME, ","))

#define TIMVX_LOG_MAP_DATATYPE_ATTR(LEVEL, ATTR_NAME, ATTR_VALUE) \
    TIMVX_LOG(LEVEL, "{:>20}: {}", #ATTR_NAME, ATTR_VALUE)

    // string <---> PoolType map
    extern std::map<std::string, PoolType> gStrToPoolTypeMap;
    extern std::map<PoolType, std::string> gPoolTypeToStrMap;

    // string <---> PadType map
    extern std::map<std::string, PadType> gStrToPadTypeMap;
    extern std::map<PadType, std::string> gPadTypeToStrMap;

    // string <---> RoundType map
    extern std::map<std::string, RoundType> gStrToRoundTypeMap;
    extern std::map<RoundType, std::string> gRoundTypeToStrMap;

    // string <---> OverflowPolicy map
    extern std::map<std::string, OverflowPolicy> gStrToOverflowPolicyMap;
    extern std::map<OverflowPolicy, std::string> gOverflowPolicyToStrMap;

    // string <---> RoundingPolicy map
    extern std::map<std::string, RoundingPolicy> gStrToRoundingPolicyMap;
    extern std::map<RoundingPolicy, std::string> gRoundingPolicyToStrMap;

    // string <---> ResizeType map
    extern std::map<std::string, ResizeType> gStrToResizeTypeMap;
    extern std::map<ResizeType, std::string> gResizeTypeToStrMap;

    // string <---> DataLayout map
    extern std::map<std::string, DataLayout> gStrToDataLayoutMap;
    extern std::map<DataLayout, std::string> gDataLayoutToStrMap;

    class OpCreator
    {
    public:
        OpCreator(std::string op_name) : m_op_name(op_name) {}

    public:
        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) = 0;

        bool parsePoolType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, PoolType& pool_type, bool necessary = true);
        bool parsePadType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, PadType& pad_type, bool necessary = true);
        bool parseRoundType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, RoundType& round_type, bool necessary = true);
        bool parseOverflowPolicyType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, OverflowPolicy& overflow_policy_type, bool necessary = true);
        bool parseRoundingPolicyType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, RoundingPolicy& rounding_policy_type, bool necessary = true);
        bool parseResizeType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, ResizeType& resize_type, bool necessary = true);
        bool parseDataLayoutType(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, DataLayout& data_layout_type, bool necessary = true);
        
        template <class T>
        bool checkObjType(const json& item)
        {
            bool ret = true;
            try
            {
                T temp = item.get<T>();
            }
            catch(const std::exception& e)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "exception occur: {}", e.what());
                ret = false;
            }
            return ret;
        }

        template <class T>
        bool parseValue(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, T& parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} should contain {} attr, please check", op_name.c_str(), attr_c_name);
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (checkObjType<T>(op_info[attr_c_name]))
                {
                    parsed_value = op_info[attr_c_name].get<T>();
                }
                else
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} parse {} attr fail, please check", op_name.c_str(), attr_c_name);
                    return false;
                }
            }
            return true;
        }

        template <class T>
        bool checkListItemType(const json& list_value)
        {
            bool ret = true;
            try
            {
                for (int i = 0; i < list_value.size(); i++)
                {
                    T temp = list_value[i].get<T>();
                }
            }
            catch(const std::exception& e)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "exception occur: {}", e.what());
                ret = false;
            }
            return ret;
        }

        template <class T, int list_num>
        bool parseFixList(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, std::array<T, list_num>& parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} should contain {} attr, please check", op_name.c_str(), attr_c_name);
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!op_info[attr_c_name].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} is not list", op_name.c_str(), attr_c_name);
                    return false;
                }
                json list_value = op_info[attr_c_name];
                if (list_value.size() != list_num)
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} len should be {}", op_name.c_str(), attr_c_name, list_num);
                    return false;
                }
                if (!checkListItemType<T>(list_value))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} item type wrong", op_name.c_str(), attr_c_name);
                    return false;
                }
                for (int i = 0; i < list_value.size(); i++)
                {
                    T temp = list_value[i].get<T>();
                    parsed_value[i] = temp;
                }
            }
            return true;
        }

        template <class T>
        bool parseDynamicList(const json& op_info, const std::string& op_name, 
            const std::string& attr_name, std::vector<T>& parsed_value, bool necessary = true)
        {
            parsed_value.clear();
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} should contain {} attr, please check", op_name.c_str(), attr_c_name);
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!op_info[attr_c_name].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} is not list", op_name.c_str(), attr_c_name);
                    return false;
                }
                json list_value = op_info[attr_c_name];
                if (!checkListItemType<T>(list_value))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {}'s attr {} item type wrong", op_name.c_str(), attr_c_name);
                    return false;
                }
                for (int i = 0; i < list_value.size(); i++)
                {
                    T temp = list_value[i].get<T>();
                    parsed_value.push_back(temp);
                }
            }
            return true;
        }

    protected:
        std::string              m_op_name;
    };


    class TimVXOp
    {
    private:
        TimVXOp() = default;
    public:
        bool addCreator(std::string op_type, OpCreator* creator);
        OpCreator* getOpCreator(std::string op_type);
        static TimVXOp* getInstance()
        {
            static TimVXOp instance;
            return &instance;
        }

    private:
        std::map<std::string, OpCreator*> m_op_creator_map;
    };


    #define REGISTER_OP_CREATOR(name, op_type)                       \
        void register##op_type##OpCreator() {                        \
            static name _temp(#op_type);                             \
            TimVXOp::getInstance()->addCreator(#op_type, &_temp);    \
        }

}  //namespace TimVX
