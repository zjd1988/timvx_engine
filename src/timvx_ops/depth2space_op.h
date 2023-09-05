/***********************************
******  depth2space_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class Depth2SpaceOpCreator : public OpCreator
    {
    public:
        struct Depth2SpaceOpAttr
        {
            int32_t    block_size;
            DataLayout layout;
        };

        Depth2SpaceOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseBlockSizeAttr(const json& op_info, Depth2SpaceOpAttr& op_attr);
        bool parseLayoutAttr(const json& op_info, Depth2SpaceOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Depth2SpaceOpAttr& op_attr);

    };

} // namespace TimVX
