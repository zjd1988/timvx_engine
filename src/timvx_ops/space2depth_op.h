/***********************************
******  space2depth_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class Space2DepthOpCreator : public OpCreator
    {
    public:
        struct Space2DepthOpAttr
        {
            std::vector<int32_t> block_size;
            DataLayout           layout;
        };

        Space2DepthOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseBlockSizeAttr(const json& op_info, Space2DepthOpAttr& op_attr);
        bool parseLayoutAttr(const json& op_info, Space2DepthOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Space2DepthOpAttr& op_attr);

    };

} // namespace TimVX