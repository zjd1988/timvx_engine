/***********************************
******  slice_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class SliceOpCreator : public OpCreator
    {
    public:
        struct SliceOpAttr
        {
            uint32_t             dims;
            std::vector<int32_t> start;
            std::vector<int32_t> length;
        };

        SliceOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseDimsAttr(const json& op_info, SliceOpAttr& op_attr);
        bool parseStartAttr(const json& op_info, SliceOpAttr& op_attr);
        bool parseLengthAttr(const json& op_info, SliceOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, SliceOpAttr& op_attr);

    };

} // namespace TimVX