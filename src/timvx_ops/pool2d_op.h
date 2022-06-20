/***********************************
******  pool2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{
    class Pool2dCreator : public OpCreator
    {
    public:
        struct Pool2dOpAttr
        {
            // pool2d common
            PoolType type;
            RoundType round_type;
            DataLayout layout;
            // classic pool2d common
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            // Classic Pool2d 1
            PadType padding;
            // Classic Pool2d 2
            std::array<uint32_t, 4> pad;
            // global and adaptive pool2d
            std::array<uint32_t, 2> input_size;
            // adaptive pool2d
            std::array<uint32_t, 2> output_size;
        };

        enum Pool2dCfgType
        {
            None,
            Classic_Pool2d_1,
            Classic_Pool2d_2,
            Global_Pool2d,
            Adaptive_Pool2d,
        };

        virtual Operation* onCreate(std::shared_ptr<Graph> &graph, const json &op_info) override;

    private:
        Pool2dCfgType getPool2dType(const json &op_info);
        bool parsePad(const json &op_info, Pool2dOpAttr &op_attr);
        bool parsePadding(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseType(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseKsize(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseStride(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseInputSize(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseOutputSize(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseRoundType(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseLayout(const json &op_info, Pool2dOpAttr &op_attr);
        bool parseOpAttr(const json &op_info, Pool2dOpAttr &op_attr, Pool2dCfgType pool_type);

    private:
        std::string m_op_name = "Pool2d";
    };

} // namespace TIMVXPY
