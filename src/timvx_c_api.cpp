/***********************************
******  timvx_c_api.cpp
******
******  Created by zhaojd on 2023/08/02.
***********************************/
#include <string>
#include <memory>
#include "common/timvx_log.h"
#include "timvx_c_api.h"
#include "timvx_cxx_api.h"

std::map<TimvxQueryCmd, std::string> gTimvxQueryCmd2Str = {
    {TIMVX_QUERY_IN_OUT_NUM, "TIMVX_QUERY_IN_OUT_NUM"},
    {TIMVX_QUERY_INPUT_ATTR, "TIMVX_QUERY_INPUT_ATTR"},
    {TIMVX_QUERY_OUTPUT_ATTR, "TIMVX_QUERY_OUTPUT_ATTR"},
};

int timvxInit(TimvxContext* context, const char* model_para_path, const char* model_weight_path)
{
    std::unique_ptr<TimVX::EngineInterface> engine_ins(new TimVX::EngineInterface(model_para_path, model_weight_path));
    if (nullptr == engine_ins.get() || false == engine_ins->getEngineStatus())
        return -1;
    TimVX::EngineInterface* engine_ptr = engine_ins.release();
    *context = (TimvxContext)engine_ptr;
    return 0;
}

int timvxDestroy(TimvxContext context)
{
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    delete engine_ptr;
    return 0;
}

int timvxQuery(TimvxContext context, TimvxQueryCmd cmd, void* info, uint32_t size)
{
    if (nullptr == info)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input info is nullptr");
        return -1;
    }
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    TimvxInputOutputNum io_num;
    if (0 != engine_ptr->getInputOutputNum(io_num))
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "get io tensor num fail");
        return -1;
    }
    if (TIMVX_QUERY_IN_OUT_NUM == cmd)
    {
        if (size != sizeof(TimvxInputOutputNum))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor io_num need {} size bytes to store, but input size is {}", 
                sizeof(TimvxInputOutputNum), size);
            return -1;
        }
        TimvxInputOutputNum* dst_io_num = (TimvxInputOutputNum*)info;
        *dst_io_num = io_num;
    }
    else if (TIMVX_QUERY_INPUT_ATTR == cmd)
    {
        int tensor_attr_size = sizeof(TimvxTensorAttr) * io_num.n_input;
        if (size != tensor_attr_size)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}th input tensor need {} size bytes to store attr, but input size is {}", 
                io_num.n_input, tensor_attr_size, size);
            return -1;
        }
        for (int index = 0; index < io_num.n_input; index++)
        {
            TimvxTensorAttr* tensor_attr = (TimvxTensorAttr*)info + index;
            if (engine_ptr->getInputTensorAttr(index, *tensor_attr))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "get {}th input tensor attr fail", index);
                return -1;
            }
        }
    }
    else if(TIMVX_QUERY_OUTPUT_ATTR == cmd)
    {
        int tensor_attr_size = sizeof(TimvxTensorAttr) * io_num.n_output;
        if (size != tensor_attr_size)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}th output tensor need {} size bytes to store attr, but input size is {}", 
                io_num.n_output, tensor_attr_size, size);
            return -1;
        }
        for (int index = 0; index < io_num.n_output; index++)
        {
            TimvxTensorAttr* tensor_attr = (TimvxTensorAttr*)info + index;
            if (engine_ptr->getOutputTensorAttr(index, *tensor_attr))
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "get {}th output tensor attr fail", index);
                return -1;
            }
        }
    }
    else
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported query cmd: {} ", int(cmd));
        return -1;
    }
    return 0;
}

int timvxInputsSet(TimvxContext context, uint32_t n_inputs, TimvxInput inputs[])
{
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    std::vector<TimvxInput> input_datas;
    for (int i = 0; i < n_inputs; i++)
    {
        input_datas.push_back(inputs[i]);
    }
    return engine_ptr->setInputs(input_datas);
}

int timvxRun(TimvxContext context)
{
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    return engine_ptr->runEngine();
}

int timvxOutputsGet(TimvxContext context, uint32_t n_outputs, TimvxOutput outputs[])
{
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    std::vector<TimvxOutput> output_datas;
    for (int i = 0; i < n_outputs; i++)
    {
        output_datas.push_back(outputs[i]);
    }
    if (0 != engine_ptr->getOutputs(output_datas))
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "get engine output datas fail");
        return -1;
    }
    for (int i = 0; i < output_datas.size(); i++)
    {
        outputs[i] = output_datas[i];
    }
    return 0;
}

int timvxOutputsRelease(TimvxContext context, uint32_t n_ouputs, TimvxOutput outputs[])
{
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    return 0;
}

int timvxCompileModelAndSave(TimvxContext context, const char* weight_file, const char* para_file)
{
    TimVX::EngineInterface* engine_ptr = (TimVX::EngineInterface*)context;
    if (nullptr == engine_ptr)
    {
        TIMVX_LOG(TIMVX_LEVEL_ERROR, "input context is nullptr");
        return -1;
    }
    if (false == engine_ptr->compileModelAndSave(weight_file, para_file))
        return -1;
    return 0;
}