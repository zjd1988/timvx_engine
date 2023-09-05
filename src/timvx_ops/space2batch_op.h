/***********************************
******  space2batch_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class Space2BatchOpCreator : public OpCreator
    {
    public:
        struct Space2BatchOpAttr
        {
            std::vector<int32_t> block_size;
            std::vector<int32_t> pad;
            DataLayout           layout;
        };

        Space2BatchOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseBlockSizeAttr(const json& op_info, Space2BatchOpAttr& op_attr);
        bool parsePadAttr(const json& op_info, Space2BatchOpAttr& op_attr);
        bool parseLayoutAttr(const json& op_info, Space2BatchOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Space2BatchOpAttr& op_attr);

    };

} // namespace TimVX