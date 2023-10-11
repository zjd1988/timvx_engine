
## 源码编译
### x64编译
```
./build_x64.sh -DTIM_VX_GIT=https://github.com/VeriSilicon/TIM-VX.git -DTIM_VX_VERSION=v1.1.32
```

### rk(rv1109/rv1126/rk1808)交叉编译
```
# rknpu-1.7.0 驱动为6.4.6
./build_rk.sh rv1126 -DTIM_VX_GIT=https://github.com/VeriSilicon/TIM-VX.git -DTIM_VX_VERSION=v1.1.32 -DEXTERNAL_VIV_SDK=$PWD/rk_1.7.0_sdk  
```

### aml(A311D/S905D3/C308X/C305X)交叉编译
```
# aml sdk 驱动为6.4.8
./build_aml.sh A311D -DTIM_VX_GIT=https://github.com/VeriSilicon/TIM-VX.git -DTIM_VX_VERSION=v1.1.34.fix -DEXTERNAL_VIV_SDK=$PWD/aml_sdk
```

### aml(A311D/S905D3/C308X/C305X)本地编译
```
# aml sdk 驱动为6.4.8
./build_aml_local.sh A311D -DTIM_VX_GIT=https://github.com/VeriSilicon/TIM-VX.git -DTIM_VX_VERSION=v1.1.34.fix -DEXTERNAL_VIV_SDK=$PWD/aml_sdk
```

注:  
    (1) EXTERNAL_VIV_SDK可以从对应开发的sdk获取，该目录的文件构成如下  
    (2) 手动替换build_rk.sh/build_aml.sh中交叉编译工具链的地址COMPILE_TOOL_CHAIN

```
rknpu-1.7.0目录下对应关系:
    linux-aarch64: RK1808
    linux-arm: RK1806
    linux-armhf-puma: RV1109/RV1126

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
1. export TIM_ENGINE_PATH=localpath/timvx_engine
2. export VIVANTE_SDK_DIR=$TIM_ENGINE_PATH/3rd_party/TIM_VX/src/TIM_VX/prebuilt-sdk/x86_64_linux
3. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIVANTE_SDK_DIR/lib
4. 查看详细信息需要新增环境变量，export VSI_NN_LOG_LEVEL=5
5. python examples/api_test/lenet.py

## arm64本地测试
替换下面的localpath为本地真实路径
1. sudo apt install libjemalloc2
2. cd localpath/build_for_A311D/
3. LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ./model_verify --weight ../example/api_test/lenet_weight.bin --para ../examples/api_test/lenet_graph.json --log_level 1