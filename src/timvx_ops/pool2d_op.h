/***********************************
******  pool2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class Pool2dOpCreator : public OpCreator
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

        Pool2dOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        Pool2dCfgType getPool2dType(const json& op_info);
        bool parsePadAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parsePaddingAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseTypeAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseKsizeAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseStrideAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseInputSizeAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseOutputSizeAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseRoundTypeAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseLayoutAttr(const json& op_info, Pool2dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Pool2dOpAttr& op_attr, Pool2dCfgType pool_type);

    };

} // namespace TimVX
