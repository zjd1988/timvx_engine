/***********************************
******  engine_common.h
******
******  Created by zhaojd on 2022/06/12.
***********************************/
#include <iostream>
#include <string>
#include <vector>


namespace TIMVX
{
    /*
        Definition for tensor
    */
    #define TIMVX_MAX_DIMS                           16      /* maximum dimension of tensor. */
    #define TIMVX_MAX_NAME_LEN                       256     /* maximum name lenth of tensor. */
    #define TIMVX_MAX_NUM_CHANNEL                    128     /* maximum channel number of graph input tensor. */

    /*
        the tensor data format.
    */
    typedef enum TimvxTensorFormat 
    {
        TIMVX_TENSOR_NCHW = 0,                               /* data format is NCHW. */
        TIMVX_TENSOR_NHWC,                                   /* data format is NHWC. */
        TIMVX_TENSOR_FORMAT_MAX
    } TimvxTensorFormat;

    /*
        the information for TIMVX_QUERY_IN_OUT_NUM.
    */
    typedef struct TimvxInputOutputNum
    {
        uint32_t n_input;                                   /* the number of input. */
        uint32_t n_output;                                  /* the number of output. */
    } TimvxInputOutputNum;

    /*
        the tensor data type.
    */
    typedef enum TimvxTensorType
    {
        TIMVX_TENSOR_FLOAT32 = 0,                            /* data type is float32. */
        TIMVX_TENSOR_FLOAT16,                                /* data type is float16. */
        TIMVX_TENSOR_INT8,                                   /* data type is int8. */
        TIMVX_TENSOR_UINT8,                                  /* data type is uint8. */
        TIMVX_TENSOR_INT16,                                  /* data type is int16. */

        TIMVX_TENSOR_TYPE_MAX
    } TimvxTensorType;

    /*
        the quantitative type.
    */
    typedef enum TimvxTensorQntType
    {
        RKNN_TENSOR_QNT_NONE = 0,                           /* none. */
        RKNN_TENSOR_QNT_DFP,                                /* dynamic fixed point. */
        RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,                  /* asymmetric affine. */

        RKNN_TENSOR_QNT_MAX
    } TimvxTensorQntType;

    /*
        the information for RKNN_QUERY_INPUT_ATTR / RKNN_QUERY_OUTPUT_ATTR.
    */
    typedef struct TimvxTensorAttr
    {
        uint32_t index;                                     /* input parameter, the index of input/output tensor,
                                                            need set before call rknn_query. */

        uint32_t n_dims;                                    /* the number of dimensions. */
        uint32_t dims[TIMVX_MAX_DIMS];                       /* the dimensions array. */
        char name[TIMVX_MAX_NAME_LEN];                       /* the name of tensor. */

        uint32_t n_elems;                                   /* the number of elements. */
        uint32_t size;                                      /* the bytes size of tensor. */

        TimvxTensorFormat fmt;                              /* the data format of tensor. */
        TimvxTensorType type;                               /* the data type of tensor. */
        TimvxTensorQntType qnt_type;                        /* the quantitative type of tensor. */
        int8_t fl;                                          /* fractional length for RKNN_TENSOR_QNT_DFP. */
        uint32_t zp;                                        /* zero point for RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC. */
        float scale;                                        /* scale for RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC. */
    } TimvxTensorAttr;

    /*
        the input information for timvx_input_set.
    */
    typedef struct TimvxInput 
    {
        uint32_t index;                                     /* the input index. */
        void* buf;                                          /* the input buf for index. */
        uint32_t size;                                      /* the size of input buf. */
        uint8_t pass_through;                               /* pass through mode.
                                                            if TRUE, the buf data is passed directly to the input node of the rknn model
                                                                        without any conversion. the following variables do not need to be set.
                                                            if FALSE, the buf data is converted into an input consistent with the model
                                                                        according to the following type and fmt. so the following variables
                                                                        need to be set.*/
        TimvxTensorType type;                              /* the data type of input buf. */
        TimvxTensorFormat fmt;                             /* the data format of input buf.
                                                            currently the internal input format of NPU is NCHW by default.
                                                            so entering NCHW data can avoid the format conversion in the driver. */
    } TimvxInput;

    /*
        the output information for timvx_outputs_get.
    */
    typedef struct TimvxOutput
    {
        uint8_t want_float;                                 /* want transfer output data to float */
        uint8_t is_prealloc;                                /* whether buf is pre-allocated.
                                                            if TRUE, the following variables need to be set.
                                                            if FALSE, the following variables do not need to be set. */
        uint32_t index;                                     /* the output index. */
        void* buf;                                          /* the output buf for index.
                                                            when is_prealloc = FALSE and rknn_outputs_release called,
                                                            this buf pointer will be free and don't use it anymore. */
        uint32_t size;                                      /* the size of output buf. */
    } TimvxOutput;

} // namespace TIMVX