/***********************************
******  tensor_info.cpp
******
******  Created by zhaojd on 2022/05/04.
***********************************/
#include "tensor_info.h"
#include <iostream>
#include <map>
using namespace std;

namespace TimVX
{

    std::map<DataType, std::string> gDataTypeToStrMap = {
        {DataType::INT8,                  "INT8"},
        {DataType::UINT8,                 "UINT8"},
        {DataType::INT16,                 "INT16"},
        {DataType::UINT16,                "UINT16"},
        {DataType::INT32,                 "INT32"},
        {DataType::UINT32,                "UINT32"},
        {DataType::FLOAT16,               "FLOAT16"},
        {DataType::FLOAT32,               "FLOAT32"},
        {DataType::BOOL8,                 "BOOL8"},
    };

    std::map<std::string, DataType> gStrToDataTypeMap = {
        {"INT8",                           DataType::INT8},
        {"UINT8",                          DataType::UINT8},
        {"INT16",                          DataType::INT16},
        {"UINT16",                         DataType::UINT16},
        {"INT32",                          DataType::INT32},
        {"UINT32",                         DataType::UINT32},
        {"FLOAT16",                        DataType::FLOAT16},
        {"FLOAT32",                        DataType::FLOAT32},
        {"BOOL8",                          DataType::BOOL8},
    };

    std::map<std::string, QuantType> gStrToQuantTypeMap = {
        {"NONE",                           QuantType::NONE},
        {"ASYMMETRIC",                     QuantType::ASYMMETRIC},
        {"SYMMETRIC_PER_CHANNEL",          QuantType::SYMMETRIC_PER_CHANNEL},
    };

    std::map<QuantType, std::string> gQuantTypeToStrMap = {
        {QuantType::NONE,                  "NONE"},
        {QuantType::ASYMMETRIC,            "ASYMMETRIC"},
        {QuantType::SYMMETRIC_PER_CHANNEL, "SYMMETRIC_PER_CHANNEL"},
    };

    std::map<std::string, TensorAttribute> gStrToTensorAttrMap = {
        {"CONSTANT",                      TensorAttribute::CONSTANT},
        {"TRANSIENT",                     TensorAttribute::TRANSIENT},
        {"VARIABLE",                      TensorAttribute::VARIABLE},
        {"INPUT",                         TensorAttribute::INPUT},
        {"OUTPUT",                        TensorAttribute::OUTPUT},
    };

    std::map<TensorAttribute, std::string> gTensorAttrToStrMap = {
        {TensorAttribute::CONSTANT,       "CONSTANT"},
        {TensorAttribute::TRANSIENT,      "TRANSIENT"},
        {TensorAttribute::VARIABLE,       "VARIABLE"},
        {TensorAttribute::INPUT,          "INPUT"},
        {TensorAttribute::OUTPUT,         "OUTPUT"},
    };

    bool TensorSpecConstruct::parseTensorDataType(const json &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, DataType &data_type)
    {
        std::string data_type_str;
        bool parse_result = parseValue<std::string>(tensor_info, tensor_name, key_name, data_type_str);
        if (parse_result)
        {
            if (gStrToDataTypeMap.find(data_type_str) != gStrToDataTypeMap.end())
                data_type = gStrToDataTypeMap[data_type_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} contains invalid datatype {}", tensor_name.c_str(),
                    key_name.c_str(), data_type_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool TensorSpecConstruct::parseTensorAttr(const json &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, TensorAttribute &tensor_attr)
    {
        std::string tensor_attr_str;
        bool parse_result = parseValue<std::string>(tensor_info, tensor_name, key_name, tensor_attr_str);
        if (parse_result)
        {
            if (gStrToTensorAttrMap.find(tensor_attr_str) != gStrToTensorAttrMap.end())
                tensor_attr = gStrToTensorAttrMap[tensor_attr_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} contains attribute type {}", tensor_name.c_str(),
                    key_name.c_str(), tensor_attr_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool TensorSpecConstruct::parseTensorQuantType(const json &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, QuantType &quant_type)
    {
        std::string quant_type_str;
        bool parse_result = parseValue<std::string>(tensor_info, tensor_name, key_name, quant_type_str);
        if (parse_result)
        {
            if (gStrToQuantTypeMap.find(quant_type_str) != gStrToQuantTypeMap.end())
                quant_type = gStrToQuantTypeMap[quant_type_str];
            else
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s attr {} contains quant type {}", tensor_name.c_str(),
                    key_name.c_str(), quant_type_str.c_str());
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool TensorSpecConstruct::constructTensorspec(const json &tensor_info, 
        const std::string &tensor_name, TensorSpec& tensorspec)
    {
        float scale = 1.0;
        int32_t zero_point = 0;
        int channel_dim = -1;
        std::vector<float> scales;
        std::vector<int32_t> zero_points;
        std::vector<uint32_t> shape;
        QuantType quant_type;
        TensorAttribute tensor_attr;
        DataType data_type;
        if (!parseDynamicList<uint32_t>(tensor_info, tensor_name, "shape", shape)
            || !parseTensorAttr(tensor_info, tensor_name, "attribute", tensor_attr)
            || !parseTensorDataType(tensor_info, tensor_name, "data_type", data_type))
            return false;
        if (tensor_info.contains("quant_info") && tensor_info["quant_info"].size())
        {
            if (!tensor_info["quant_info"].is_object())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s quant_info should be a dict item", tensor_name.c_str());
                return false;
            }

            json quant_info = tensor_info["quant_info"];
            if (quant_info.contains("scale") && quant_info.contains("scales"))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s quant_info both contain scale and scales", tensor_name.c_str());
                return false;
            }

            if (quant_info.contains("zero_point") && quant_info.contains("zero_points"))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s quant_info both contain zero_point and zero_points", tensor_name.c_str());
                return false;
            }

            if (!(quant_info.contains("scale") && quant_info.contains("zero_point")) &&
                !(quant_info.contains("scales") && quant_info.contains("zero_points")))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {}'s quant_info should both contain sacle/zero_point or scales/zero_points", tensor_name.c_str());
                return false;
            }

            if (!parseTensorQuantType(quant_info, tensor_name, "quant_type", quant_type))
                return false;

            if (!parseValue<int32_t>(quant_info, tensor_name, "channel_dim", channel_dim, false))
                return false;

            if (!parseDynamicList<int32_t>(quant_info, tensor_name, "zero_points", zero_points, false)
                || !parseDynamicList<float>(quant_info, tensor_name, "scales", scales, false))
                return false;

            if (!parseValue<float>(quant_info, tensor_name, "scale", scale, false)
                || !parseValue<int32_t>(quant_info, tensor_name, "zero_point", zero_point, false))
                return false;

            if (quant_info.contains("scale"))
                scales.push_back(scale);
            if (quant_info.contains("zero_point"))
                zero_points.push_back(zero_point);

            Quantization input_quant(quant_type, channel_dim, scales, zero_points);
            TensorSpec temp_tensor_spec(data_type, shape, tensor_attr, input_quant);
            tensorspec = temp_tensor_spec;
        }
        else
        {
            TensorSpec temp_tensor_spec(data_type, shape, tensor_attr);
            tensorspec = temp_tensor_spec;
        }
        return true;
    }

} // namespace TimVX