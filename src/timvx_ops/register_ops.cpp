/***********************************
******  register_ops.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include <mutex>

namespace TimVX
{

    extern void registerActivationOpCreator();
    extern void registerAddNOpCreator();
    extern void registerArgOpCreator();
    extern void registerBatch2SpaceOpCreator();
    extern void registerBatchNormOpCreator();
    extern void registerClipOpCreator();
    extern void registerConcatOpCreator();
    extern void registerConv1dOpCreator();
    extern void registerConv2dOpCreator();
    extern void registerDeConv1dOpCreator();
    extern void registerDeConv2dOpCreator();
    extern void registerDepth2SpaceOpCreator();
    extern void registerDropoutOpCreator();
    extern void registerEltwiseOpCreator();
    extern void registerFullyConnectedOpCreator();
    extern void registerGatherOpCreator();
    extern void registerGatherNdOpCreator();
    extern void registerGroupedConv2dOpCreator();
    extern void registerInstanceNormalizationOpCreator();
    extern void registerL2NormalizationOpCreator();
    extern void registerLayerNormalizationOpCreator();
    extern void registerLocalResponseNormalizationOpCreator();
    extern void registerLogicalOpCreator();
    extern void registerLogSoftmaxOpCreator();
    extern void registerMatmulOpCreator();
    extern void registerMaxpoolWithArgmaxOpCreator();
    extern void registerMaxUnpool2dOpCreator();
    extern void registerMomentsOpCreator();
    extern void registerNBGOpCreator();
    extern void registerPadOpCreator();
    extern void registerPool2dOpCreator();
    extern void registerReduceOpCreator();
    extern void registerRelationalOperationsOpCreator();
    extern void registerReorgOpCreator();
    extern void registerReshapeOpCreator();
    extern void registerResizeOpCreator();
    extern void registerResize1dOpCreator();
    extern void registerReverseOpCreator();
    extern void registerScatterNDOpCreator();
    extern void registerSelectOpCreator();
    extern void registerSimpleOperationsOpCreator();
    extern void registerSliceOpCreator();
    extern void registerSoftmaxOpCreator();
    extern void registerSpace2BatchOpCreator();
    extern void registerSpace2DepthOpCreator();
    extern void registerSpatialTransformerOpCreator();
    extern void registerSplitOpCreator();
    extern void registerSqueezeOpCreator();
    extern void registerStackOpCreator();
    extern void registerStridedSliceOpCreator();
    extern void registerTileOpCreator();
    extern void registerTransposeOpCreator();
    extern void registerUnstackOpCreator();

    static std::once_flag s_flag;
    void registerOps()
    {
        std::call_once(s_flag, [&]() 
        {
            registerActivationOpCreator();
            registerAddNOpCreator();
            registerArgOpCreator();
            registerBatch2SpaceOpCreator();
            registerBatchNormOpCreator();
            registerClipOpCreator();
            registerConcatOpCreator();
            registerConv1dOpCreator();
            registerConv2dOpCreator();
            registerDeConv1dOpCreator();
            registerDeConv2dOpCreator();
            registerDepth2SpaceOpCreator();
            registerDropoutOpCreator();
            registerEltwiseOpCreator();
            registerFullyConnectedOpCreator();
            registerGatherOpCreator();
            registerGatherNdOpCreator();
            registerGroupedConv2dOpCreator();
            registerInstanceNormalizationOpCreator();
            registerL2NormalizationOpCreator();
            registerLayerNormalizationOpCreator();
            registerLocalResponseNormalizationOpCreator();
            registerLogicalOpCreator();
            registerLogSoftmaxOpCreator();
            registerMatmulOpCreator();
            registerMaxpoolWithArgmaxOpCreator();
            registerMaxUnpool2dOpCreator();
            registerMomentsOpCreator();
            registerNBGOpCreator();
            registerPadOpCreator();
            registerPool2dOpCreator();
            registerReduceOpCreator();
            registerRelationalOperationsOpCreator();
            registerReorgOpCreator();
            registerReshapeOpCreator();
            registerResizeOpCreator();
            registerResize1dOpCreator();
            registerReverseOpCreator();
            registerScatterNDOpCreator();
            registerSelectOpCreator();
            registerSimpleOperationsOpCreator();
            registerSliceOpCreator();
            registerSoftmaxOpCreator();
            registerSpace2BatchOpCreator();
            registerSpace2DepthOpCreator();
            registerSpatialTransformerOpCreator();
            registerSplitOpCreator();
            registerSqueezeOpCreator();
            registerStackOpCreator();
            registerStridedSliceOpCreator();
            registerTileOpCreator();
            registerTransposeOpCreator();
            registerUnstackOpCreator();
        });
    }

} // namespace TimVX