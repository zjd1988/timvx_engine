/***********************************
******  json_parse.h
******
******  Created by zhaojd on 2022/05/04.
***********************************/
#pragma once
#include <vector>
#include <array>
#include "nlohmann/json.hpp"
#include "common/timvx_log.h"

using namespace nlohmann;

namespace TimVX
{

    template <class T>
    static bool checkObjType(const json &item)
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
    static bool checkListItemType(const json &list_value)
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
    static bool parseFixList(const json &json_info, const std::string &object_name, 
        const std::string &attr_name, std::array<T, list_num> parsed_value, bool necessary = true)
    {
        const char* attr_c_name = attr_name.c_str();
        if (necessary && !json_info.contains(attr_c_name))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {} should contain {} attr, please check", object_name.c_str(), attr_name.c_str());
            return false;
        }
        if (json_info.contains(attr_c_name))
        {
            if (!json_info[attr_c_name].is_array())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {}'s attr {} is not list", object_name.c_str(), attr_name.c_str());
                return false;
            }
            json list_value = json_info[attr_c_name];
            if (list_value.size() != list_num)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {}'s attr {} len should be %d", object_name.c_str(), attr_name.c_str(), list_num);
                return false;
            }
            if (!checkListItemType<T>(list_value))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {}'s attr {} item type wrong", object_name.c_str(), attr_name.c_str());
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
    static bool parseDynamicList(const json &json_info, const std::string &object_name, 
        const std::string &attr_name, std::vector<T> &parsed_value, bool necessary = true)
    {
        parsed_value.clear();
        const char* attr_c_name = attr_name.c_str();
        if (necessary && !json_info.contains(attr_c_name))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {} should contain {} attr, please check", object_name.c_str(), attr_name.c_str());
            return false;
        }
        if (json_info.contains(attr_c_name))
        {
            if (!json_info[attr_c_name].is_array())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {}'s attr {} is not list", object_name.c_str(), attr_name.c_str());
                return false;
            }
            json list_value = json_info[attr_c_name];
            if (!checkListItemType<T>(list_value))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {}'s attr {} item type wrong", object_name.c_str(), attr_name.c_str());
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

    template <class T>
    static bool parseValue(const json &json_info, const std::string &object_name, 
        const std::string &attr_name, T &parsed_value, bool necessary = true)
    {
        const char* attr_c_name = attr_name.c_str();
        if (necessary && !json_info.contains(attr_c_name))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {} should contain {} attr, please check", object_name.c_str(), attr_name.c_str());
            return false;
        }
        if (json_info.contains(attr_c_name))
        {
            if (checkObjType<T>(json_info[attr_c_name]))
            {
                parsed_value = json_info[attr_c_name].get<T>();
            }
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "object {} parse {} attr fail, please check", object_name.c_str(), attr_name.c_str());
                return false;
            }
        }
        return true;
    }

} // namespace TimVX