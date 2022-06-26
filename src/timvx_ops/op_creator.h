/***********************************
******  op_creator.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include <iostream>
#include <map>
#include <mutex>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/types.h"
#include "nlohmann/json.hpp"
#include "timvx_define.h"
using namespace tim::vx;
using namespace nlohmann;

namespace TIMVX
{

    class OpCreator 
    {
    public:
        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) = 0;

        bool parsePoolType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, PoolType &pool_type, bool necessary = true);
        bool parsePadType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, PadType &pad_type, bool necessary = true);
        bool parseRoundType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, RoundType &round_type, bool necessary = true);
        bool parseOverflowPolicyType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, OverflowPolicy &overflow_policy_type, bool necessary = true);
        bool parseRoundingPolicyType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, RoundingPolicy &rounding_policy_type, bool necessary = true);
        bool parseResizeType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, ResizeType &resize_type, bool necessary = true);
        bool parseDataLayoutType(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, DataLayout &data_layout_type, bool necessary = true);
        
        template <class T>
        bool checkObjType(const json &item)
        {
            bool ret = true;
            try
            {
                T temp = item.get<T>();
            }
            catch(const std::exception& e)
            {
                TIMVX_ERROR("exception occur: %s\n", e.what());
                ret = false;
            }
            return ret;
        }

        template <class T>
        bool parseValue(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, T &parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                TIMVX_ERROR("op %s should contain %s attr, please check", op_name.c_str(), attr_c_name);
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
                    TIMVX_ERROR("op %s parse %s attr fail, please check", op_name.c_str(), attr_c_name);
                    return false;
                }
            }
            return true;
        }

        template <class T>
        bool checkListItemType(const json &list_value)
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
                TIMVX_ERROR("exception occur: %s\n", e.what());
                ret = false;
            }
            return ret;
        }

        template <class T, int list_num>
        bool parseFixList(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, std::array<T, list_num> &parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                TIMVX_ERROR("op %s should contain %s attr, please check", op_name.c_str(), attr_c_name);
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!op_info[attr_c_name].is_array())
                {
                    TIMVX_ERROR("op %s's attr %s is not list", op_name.c_str(), attr_c_name);
                    return false;
                }
                json list_value = op_info[attr_c_name];
                if (list_value.size() != list_num)
                {
                    TIMVX_ERROR("op %s's attr %s len should be %d", op_name.c_str(), attr_c_name, list_num);
                    return false;
                }
                if (!checkListItemType<T>(list_value))
                {
                    TIMVX_ERROR("op %s's attr %s item type wrong", op_name.c_str(), attr_c_name);
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
        bool parseDynamicList(const json &op_info, const std::string &op_name, 
            const std::string &attr_name, std::vector<T> &parsed_value, bool necessary = true)
        {
            parsed_value.clear();
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                TIMVX_ERROR("op %s should contain %s attr, please check", op_name.c_str(), attr_c_name);
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!op_info[attr_c_name].is_array())
                {
                    TIMVX_ERROR("op %s's attr %s is not list", op_name.c_str(), attr_c_name);
                    return false;
                }
                json list_value = op_info[attr_c_name];
                if (!checkListItemType<T>(list_value))
                {
                    TIMVX_ERROR("op %s's attr %s item type wrong", op_name.c_str(), attr_c_name);
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
            static name _temp;                                       \
            TimVXOp::getInstance()->addCreator(#op_type, &_temp);    \
        }

}  //namespace TIMVX
