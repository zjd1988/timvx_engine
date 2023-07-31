/* ****************************************
 * @Author: zjd
 * @Date: 2021-08-27 
 * @LastEditTime: 2021-08-27 
 * @LastEditors: zjd
 *****************************************/
#include <mutex>

namespace TIMVX
{

    extern void registerActivationOpCreator();
    extern void registerEltwiseOpCreator();
    extern void registerConv2dOpCreator();
    extern void registerFullyConnectedOpCreator();
    extern void registerSoftmaxOpCreator();
    extern void registerPool2dOpCreator();
    extern void registerReshapeOpCreator();
    extern void registerResizeOpCreator();
    extern void registerTransposeOpCreator();
    extern void registerConcatOpCreator();

    static std::once_flag s_flag;
    void registerOps()
    {
        std::call_once(s_flag, [&]() 
        {
            registerActivationOpCreator();
            registerEltwiseOpCreator();
            registerConv2dOpCreator();
            registerFullyConnectedOpCreator();
            registerSoftmaxOpCreator();
            registerPool2dOpCreator();
            registerReshapeOpCreator();
            registerResizeOpCreator();
            registerTransposeOpCreator();
            registerConcatOpCreator();
        });
    }

} // namespace TIMVX

