# timvx_engine
create timvx graph from json/weight files, and aim to deploy graph on rockchip/amlogic/nxp chip's npu

## 准备(PC-x64)
1. git clone https://github.com/zjd1988/TIM-VX-python
2. 按照readme进行编译安装libtim-vx.so, 记下对应的安装路径TIM_VX_INSTALL_PATH
3. 切换到TIM-VX-python-master/src/pytim目录下按照步骤编译得到python接口
4. git clone https://github.com/zjd1988/timvx_engine.git
5. 切换到timvx-engine代码路径, 替换CMakeLists.txt中的TIM_VX_INSTALL_PATH为真实路径
6. mkdir build & cmake .. & make

## 测试(PC-x64)
1. 使用现成的demo进行测试, 上一步完成后会在build目录生成对应的测试代码
2. 自行准备rknn模型使用timvx-python接口进行转换保存得到json和weight权重，  
    参照test.cpp编写测试代码，覆盖test.cpp
3. 重新进行make即可