# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from examples.ops_test.activations_test import test_activations_op
from examples.ops_test.elementwise_test import test_elementwise_op
from examples.ops_test.conv1d_test import test_conv1d_op
from examples.ops_test.conv2d_test import test_conv2d_op
from examples.ops_test.deconv1d_test import test_deconv1d_op
from examples.ops_test.deconv2d_test import test_deconv2d_op
from examples.ops_test.groupedconv2d_test import test_groupedconv2d_op
from examples.ops_test.instancenormalization_test import test_instancenormalization_op
from examples.ops_test.layernormalization_test import test_layernormalization_op
from examples.ops_test.logsoftmax_test import test_logsoftmax_op
from examples.ops_test.matmul_test import test_matmul_op
from examples.ops_test.maxpoolwithargmax_test import test_maxpoolwithargmax_op
from examples.ops_test.maxunpool2d_test import test_maxunpool2d_op
from examples.ops_test.moments_test import test_moments_op
from examples.ops_test.relational_operations_test import test_relational_operations_op
from examples.ops_test.reorg_test import test_reorg_op
from examples.ops_test.resize1d_test import test_resize1d_op
from examples.ops_test.scatternd_test import test_scatternd_op
from examples.ops_test.simple_operations_test import test_simple_operations_op
from examples.ops_test.spatial_transformer_test import test_spatial_transformer_op
from examples.ops_test.tile_test import test_tile_op
from examples.ops_test.transpose_conv_test import test_transpose_conv_op
from examples.ops_test.unstack_test import test_unstack_op

# setLogLevel("DEBUG")

if __name__ == "__main__":
    test_result = {}
    test_result.update(test_activations_op())
    test_result.update(test_elementwise_op())
    test_result.update(test_conv1d_op())
    test_result.update(test_conv2d_op())
    test_result.update(test_deconv1d_op())
    test_result.update(test_deconv2d_op())
    test_result.update(test_groupedconv2d_op())
    test_result.update(test_instancenormalization_op())
    test_result.update(test_layernormalization_op())
    test_result.update(test_logsoftmax_op())
    test_result.update(test_matmul_op())
    test_result.update(test_maxpoolwithargmax_op())
    test_result.update(test_maxunpool2d_op())
    test_result.update(test_moments_op())
    test_result.update(test_relational_operations_op())
    test_result.update(test_reorg_op())
    test_result.update(test_resize1d_op())
    test_result.update(test_scatternd_op())
    test_result.update(test_simple_operations_op())
    test_result.update(test_spatial_transformer_op())
    test_result.update(test_tile_op())
    test_result.update(test_transpose_conv_op())
    test_result.update(test_unstack_op())

    print("ops_test summary: ")
    for key, value in test_result.items():
        print("{}: {}".format(key, value))