/***********************************
******  stridedslice_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class StridedSliceOpCreator : public OpCreator
    {
    public:
        struct StridedSliceOpAttr
        {
            std::vector<int32_t> begin_dims;
            std::vector<int32_t> end_dims;
            std::vector<int32_t> stride_dims;
            int32_t              begin_mask;
            int32_t              end_mask;
            int32_t              shrink_axis_mask;
        };

        StridedSliceOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseBeginDimsAttr(const json& op_info, StridedSliceOpAttr& op_attr);
        bool parseEndDimsAttr(const json& op_info, StridedSliceOpAttr& op_attr);
        bool parseStrideDimsAttr(const json& op_info, StridedSliceOpAttr& op_attr);
        bool parseBeginMaskAttr(const json& op_info, StridedSliceOpAttr& op_attr);
        bool parseEndMaskAttr(const json& op_info, StridedSliceOpAttr& op_attr);
        bool parseShrinkAxisMaskAttr(const json& op_info, StridedSliceOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, StridedSliceOpAttr& op_attr);

    };

} // namespace TimVX