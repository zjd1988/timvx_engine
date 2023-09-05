/***********************************
******  localresponsenormalization_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class LocalResponseNormalizationOpCreator : public OpCreator
    {
    public:
        struct LocalResponseNormalizationOpAttr
        {
            uint32_t size;
            float    alpha;
            float    beta;
            float    bias;
            int32_t  axis;
        };

        LocalResponseNormalizationOpCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseSizeAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr);
        bool parseAlphaAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr);
        bool parseBetaAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr);
        bool parseBiasAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr);
        bool parseAxisAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, LocalResponseNormalizationOpAttr& op_attr);

    };

} // namespace TimVX
