
## 源码编译
### x64编译
./build_x64.sh -DTIM_VX_GIT=https://github.com/VeriSilicon/TIM-VX.git -DTIM_VX_VERSION=v1.1.32

### rk(rv1109/rv1126/rk1808)编译
rknpu-1.7.0  
./build_rk.sh arm32 -DTIM_VX_GIT=https://github.com/VeriSilicon/TIM-VX.git -DTIM_VX_VERSION=v1.1.32 -DEXTERNAL_VIV_SDK=$PWD/rk_1.7.0_sdk  

注:  
    (1) EXTERNAL_VIV_SDK可以从对应开发的sdk获取，该目录的文件构成如下  
    (2) 手动替换build_rk.sh 中交叉编译工具链的地址COMPILE_TOOL_CHAIN

```
sdk_dir/  
    drivers/  
        libArchModelSw.so  
        libCLC.so  
        libGAL.so  
        libNNArchPerf.so  
        ......  
    include/  
        CL/  
        VX/  
```
## x64测试
替换下面的localpath为本地真实路径  
1. export TIM_ENGINE_CODE=localpath/timvx_engine
2. export VIVANTE_SDK_DIR=$TIMVX_ENGINE_PATH/3rd_party/TIM_VX/src/TIM_VX/prebuilt-sdk/x86_64_linux
3. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIMVX_CODE_PATH/prebuilt-sdk/x86_64_linux/lib
4. 查看详细信息需要新增环境变量，export VSI_NN_LOG_LEVEL=5
5. python examples/api_test/lenet.py
