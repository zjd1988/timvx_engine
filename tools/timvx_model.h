/***********************************
******  timvx_model.h
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#pragma once
#include <memory>
#include "tool_utils.h"
#include "timvx_c_api.h"
#include "common/timer.h"

namespace TimVX
{

    class TimVXModel
    {
    public:
        TimVXModel(CmdLineArgOption& opt);
        ~TimVXModel();
        int modelInfer();
        int modelBenchmark();
        int modelCompile();
        bool modelStatus() { return m_status; };
        void printModelInfo();

    private:
        int initModelInputTensors();
        int initModelOutputTensors();

    private:
        // cmd option
        CmdLineArgOption                                                m_cmd_opt;
        // timvx model attr
        TimvxContext                                                    m_model_context = 0;
        TimvxInputOutputNum                                             m_io_num;
        std::vector<TimvxTensorAttr>                                    m_input_attrs;
        std::vector<TimvxTensorAttr>                                    m_output_attrs;
        std::map<std::string, std::shared_ptr<ModelTensorData>>         m_input_tensors;
        std::map<std::string, std::shared_ptr<ModelTensorData>>         m_output_tensors;
        // timer
        Timer                                                           m_timer;
        double                                                          m_time_sum;

        bool                                                            m_status = false;
    };

} // namespace TimVX