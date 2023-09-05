/***********************************
******  timvx_model.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include "timvx_model.h"
#include "common/timvx_log.h"
#include "common/pystring.h"

namespace TimVX
{

    TimVXModel::TimVXModel(CmdLineArgOption& opt)
    {
        m_cmd_opt = opt;
        // init logger
        TimVXLog::Instance().initTimVXLog("timvx_model", "", opt.log_level);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "init logger success");

        // init model
        std::string weight_file = opt.weight_file;
        std::string para_file = opt.para_file;
        if (0 != timvxInit(&m_model_context, para_file.c_str(), weight_file.c_str()))
            return;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "init model success");

        // get model input output number
        if (0 != timvxQuery(m_model_context, TIMVX_QUERY_IN_OUT_NUM, (void*)&m_io_num, sizeof(TimvxInputOutputNum)) ||
            0 == m_io_num.n_input || 0 == m_io_num.n_output)
        {
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "get model input output number fail or input:{} output:{} invalid", 
                m_io_num.n_input, m_io_num.n_output);
            return;
        }
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "get model input output number success, input:{} output:{}", 
            m_io_num.n_input, m_io_num.n_output);

        // get input tensor attr and init input tensor data
        m_input_attrs.resize(m_io_num.n_input);
        if (0 != timvxQuery(m_model_context, TIMVX_QUERY_INPUT_ATTR, (void*)&m_input_attrs[0], sizeof(TimvxTensorAttr)) || 
            0 != initModelInputTensors())
            return;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "get input tensor attr and init input tensor data success");

        // get output tensor attr and init output tensor data
        m_output_attrs.resize(m_io_num.n_output);
        if (0 != timvxQuery(m_model_context, TIMVX_QUERY_OUTPUT_ATTR, (void*)&m_output_attrs[0], sizeof(TimvxTensorAttr)) ||
            0 != initModelOutputTensors())
            return;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "get output tensor attr and init output tensor data success");

        // print model info
        printModelInfo();

        // reset timer
        m_time_sum = 0;
        m_timer.Reset();

        // set model status
        m_status = true;
    }

    TimVXModel::~TimVXModel()
    {
        if (m_model_context)
            timvxDestroy(m_model_context);
    }

    void TimVXModel::printModelInfo()
    {
        // print Model Input tensor Info
        TIMVX_LOG(TIMVX_LEVEL_INFO, "input tensors:");
        for (int i = 0; i < m_input_attrs.size(); i++)
        {
            TimvxTensorAttr attr = m_input_attrs[i];
            std::vector<int> tensor_shape(&attr.dims[0], &attr.dims[0] + attr.n_dims);
            TIMVX_LOG(TIMVX_LEVEL_INFO, "index={}, name={}, ", attr.index, attr.name);
            TIMVX_LOG(TIMVX_LEVEL_INFO, "n_dims={}, dims=[{}], ", attr.n_dims, spdlog::fmt_lib::join(tensor_shape, ","));
            TIMVX_LOG(TIMVX_LEVEL_INFO, "n_elems={}, size={}, ", attr.n_elems, attr.size);
            TIMVX_LOG(TIMVX_LEVEL_INFO, "fmt={}, type={}, qnt_type={}", 
                getFormatString(attr.fmt), getTypeString(attr.type), getQntTypeString(attr.qnt_type));
        }

        // print Model Output tensor Info
        TIMVX_LOG(TIMVX_LEVEL_INFO, "output tensors:");
        for (int i = 0; i < m_output_attrs.size(); i++)
        {
            TimvxTensorAttr attr = m_output_attrs[i];
            std::vector<int> tensor_shape(&attr.dims[0], &attr.dims[0] + attr.n_dims);
            TIMVX_LOG(TIMVX_LEVEL_INFO, "index={}, name={}, ", attr.index, attr.name);
            TIMVX_LOG(TIMVX_LEVEL_INFO, "n_dims={}, dims=[{}], ", attr.n_dims, spdlog::fmt_lib::join(tensor_shape, ","));
            TIMVX_LOG(TIMVX_LEVEL_INFO, "n_elems={}, size={}, ", attr.n_elems, attr.size);
            TIMVX_LOG(TIMVX_LEVEL_INFO, "fmt={}, type={}, qnt_type={}", 
                getFormatString(attr.fmt), getTypeString(attr.type), getQntTypeString(attr.qnt_type));
        }
        return;
    }

    int TimVXModel::initModelInputTensors()
    {
        // parse input files
        std::map<std::string, std::string> inputs_file_map;
        std::vector<std::string> input_files = pystring::split(m_cmd_opt.input_file, ";");
        if (1 == input_files.size() && 1 == m_input_attrs.size())
        {
            std::string tensor_name = m_input_attrs[0].name;
            inputs_file_map[tensor_name] = input_files[0];
        }
        else
        {
            for (int i = 0; i < input_files.size(); i++)
            {
                std::vector<std::string> split_strs = pystring::split(input_files[i], ":");
                if (2 != split_strs.size())
                {
                    TIMVX_LOG(TIMVX_LEVEL_ERROR, "parse input {} format is invalid, should be name:file_path", input_files[i]);
                    return -1;
                }
                std::string tensor_name = split_strs[0];
                std::string tensor_file = split_strs[1];
                inputs_file_map[tensor_name] = tensor_file;
            }
        }

        // parpare input tensors
        for (int i = 0; i < m_input_attrs.size(); i++)
        {
            std::shared_ptr<ModelTensorData> tensor_data;
            std::string tensor_name = m_input_attrs[i].name;
            if (inputs_file_map.end() != inputs_file_map.find(tensor_name) && "" != inputs_file_map[tensor_name])
            {
                std::string file_name = inputs_file_map[tensor_name];
                tensor_data.reset(new ModelTensorData(file_name.c_str()));
            }
            else
            {
                TimvxTensorType tensor_type = m_input_attrs[i].type;
                TimvxTensorFormat tensor_format = m_input_attrs[i].fmt;
                std::vector<int> tensor_shape(&m_input_attrs[i].dims[0], &m_input_attrs[i].dims[0] + m_input_attrs[i].n_dims);
                tensor_data.reset(new ModelTensorData(tensor_shape, tensor_type, tensor_format, true));
            }
            if (nullptr == tensor_data.get() || nullptr == tensor_data->tensorData())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "prepare model input tensor {}'s data fail", tensor_name);
                return -1;
            }
            m_input_tensors[tensor_name] = tensor_data;
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "prepare model input tensor {}'s data success", tensor_name);
        }
        return 0;
    }

    int TimVXModel::initModelOutputTensors()
    {
        // parpare output tensors
        for (int i = 0; i < m_output_attrs.size(); i++)
        {
            std::shared_ptr<ModelTensorData> tensor_data;
            std::string tensor_name = m_output_attrs[i].name;
            TimvxTensorType tensor_type = m_output_attrs[i].type;
            TimvxTensorFormat tensor_format = m_output_attrs[i].fmt;
            std::vector<int> tensor_shape(&m_output_attrs[i].dims[0], &m_output_attrs[i].dims[0] + m_output_attrs[i].n_dims);
            tensor_data.reset(new ModelTensorData(tensor_shape, tensor_type, tensor_format, true));
            if (nullptr == tensor_data.get() || nullptr == tensor_data->tensorData())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "prepare model output tensor {}'s data fail", tensor_name);
                return -1;
            }
            m_output_tensors[tensor_name] = tensor_data;
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "prepare model output tensor {}'s data success", tensor_name);
        }
        return 0;
    }

    int TimVXModel::modelInfer()
    {
        if (false == m_status || 0 == m_model_context)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "model status or context not avaliable");
            return -1;
        }
        // set model inputs
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "1 set model inputs .......");
        uint32_t n_inputs = m_input_attrs.size();
        std::vector<TimvxInput> model_inputs(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            std::string tensor_name = m_input_attrs[i].name;
            std::shared_ptr<ModelTensorData>& tensor = m_input_tensors[tensor_name];
            model_inputs[i].index = i;
            model_inputs[i].type = tensor->tensorType();
            model_inputs[i].fmt = tensor->tensorFormat();
            model_inputs[i].size = tensor->tensorLength();
            model_inputs[i].buf = tensor->tensorData();
            model_inputs[i].pass_through = (m_cmd_opt.pass_through == true ? 1 : 0);
        }
        if (0 != timvxInputsSet(m_model_context, n_inputs, &model_inputs[0]))
            return -1;

        // run model
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "2 run model .......");
        m_timer.Restart();
        if (0 != timvxRun(m_model_context))
            return -1;
        m_time_sum += m_timer.ElapsedMilliSeconds();

        // get model outputs
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "3 get model outputs .......");
        uint32_t n_outputs = m_output_attrs.size();
        std::vector<TimvxOutput> model_outputs(n_outputs);
        for (int i = 0; i < n_outputs; i++)
        {
            std::string tensor_name = m_output_attrs[i].name;
            std::shared_ptr<ModelTensorData>& tensor = m_output_tensors[tensor_name];
            model_outputs[i].index = i;
            model_outputs[i].is_prealloc = (m_cmd_opt.is_prealloc == true ? 1 : 0);
            if (0 == m_cmd_opt.is_prealloc)
            {
                model_outputs[i].size = tensor->tensorLength();
                model_outputs[i].buf = tensor->tensorData();
            }
            else
            {
                model_outputs[i].buf = nullptr;
                model_outputs[i].size = 0;
            }    
            model_outputs[i].want_float = (m_cmd_opt.want_float == true ? 1 : 0);
        }
        if (0 != timvxOutputsGet(m_model_context, n_outputs, &model_outputs[0]))
            return -1;

        // dump output to npy
        if (m_cmd_opt.output_flag)
        {
            TIMVX_LOG(TIMVX_LEVEL_DEBUG, "4 save model outputs .......");
            for (auto iter = m_output_tensors.begin(); iter != m_output_tensors.end(); iter++)
            {
                std::string tensor_name = iter->first;
                std::string file_name = tensor_name + ".npy";
                std::shared_ptr<ModelTensorData>& tensor = iter->second;
                if (0 != tensor->saveDataToNpy(file_name.c_str()))
                    break;
                TIMVX_LOG(TIMVX_LEVEL_DEBUG, "save model output tensor:{} to file {}", tensor_name, file_name);
            }
        }

        return 0;
    }

    int TimVXModel::modelBenchmark()
    {
        if (false == m_status || 0 == m_model_context)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "model status or context not avaliable");
            return -1;
        }
        int benchmark_times = m_cmd_opt.benchmark_times;
        int infer_count = 0;
        for (infer_count = 0; infer_count < benchmark_times; infer_count++)
        {
            if (0 != modelInfer())
                break;
        }
        if (infer_count != benchmark_times)
        {
            TIMVX_LOG(TIMVX_LEVEL_INFO, "run {}th model fail, total eplase time is {} ms", infer_count, m_time_sum);
            return -1;
        }
        TIMVX_LOG(TIMVX_LEVEL_INFO, "model infer {} times, average infer time is {} ms", 
            infer_count, m_time_sum / infer_count);
        return 0;
    }

    int TimVXModel::modelCompile()
    {
        if (false == m_status || 0 == m_model_context)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "model status or context not avaliable");
            return -1;
        }

        return timvxCompileModelAndSave(m_model_context, m_cmd_opt.compile_weight_file.c_str(), 
            m_cmd_opt.compile_para_file.c_str());
    }

} // namespace TimVX