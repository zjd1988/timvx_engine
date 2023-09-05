/***********************************
******  tensor_info.h
******
******  Created by zhaojd on 2022/05/04.
***********************************/
#pragma once
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "common/timvx_log.h"
#include "nlohmann/json.hpp"
using namespace nlohmann;
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class TensorSpecConstruct
    {
    public:
        static bool constructTensorspec(const json& tensor_info, const std::string& tensor_name, 
            TensorSpec& tensorspec);

    private:
        static bool parseTensorDataType(const json &tensor_info, const std::string &tensor_name, 
            const std::string &key_name, DataType &data_type);
        static bool parseTensorAttr(const json &tensor_info, const std::string &tensor_name, 
            const std::string &key_name, TensorAttribute &tensor_attr);
        static bool parseTensorQuantType(const json &tensor_info, const std::string &tensor_name, 
            const std::string &key_name, QuantType &quant_type);

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
        static bool parseFixList(const json &tensor_info, const std::string &tensor_name, 
            const std::string &attr_name, std::array<T, list_num> parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !tensor_info.contains(attr_c_name))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} should contain {} attr, please check", tensor_name.c_str(), attr_name.c_str());
                return false;
            }
            if (tensor_info.contains(attr_c_name))
            {
                if (!tensor_info[attr_c_name].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} is not list", tensor_name.c_str(), attr_name.c_str());
                    return false;
                }
                json list_value = tensor_info[attr_c_name];
                if (list_value.size() != list_num)
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} len should be %d", tensor_name.c_str(), attr_name.c_str(), list_num);
                    return false;
                }
                if (!checkListItemType<T>(list_value))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} item type wrong", tensor_name.c_str(), attr_name.c_str());
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
        static bool parseDynamicList(const json &tensor_info, const std::string &tensor_name, 
            const std::string &attr_name, std::vector<T> &parsed_value, bool necessary = true)
        {
            parsed_value.clear();
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !tensor_info.contains(attr_c_name))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} should contain {} attr, please check", tensor_name.c_str(), attr_name.c_str());
                return false;
            }
            if (tensor_info.contains(attr_c_name))
            {
                if (!tensor_info[attr_c_name].is_array())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} is not list", tensor_name.c_str(), attr_name.c_str());
                    return false;
                }
                json list_value = tensor_info[attr_c_name];
                if (!checkListItemType<T>(list_value))
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} item type wrong", tensor_name.c_str(), attr_name.c_str());
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
        static bool parseValue(const json &tensor_info, const std::string &tensor_name, 
            const std::string &attr_name, T &parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !tensor_info.contains(attr_c_name))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} should contain {} attr, please check", tensor_name.c_str(), attr_name.c_str());
                return false;
            }
            if (tensor_info.contains(attr_c_name))
            {
                if (checkObjType<T>(tensor_info[attr_c_name]))
                {
                    parsed_value = tensor_info[attr_c_name].get<T>();
                }
                else
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} parse {} attr fail, please check", tensor_name.c_str(), attr_name.c_str());
                    return false;
                }
            }
            return true;
        }
    };

} // namespace TimVX

