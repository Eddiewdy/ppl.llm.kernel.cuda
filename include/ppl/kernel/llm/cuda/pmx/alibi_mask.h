// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_ALIBI_MASK_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_ALIBI_MASK_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

// #include "ppl/kernel/llm/cuda/cublas/gemm.h"
// #include "ppl/common/cuda/nccl_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode alibi_mask(
    cudaStream_t stream,
    const ppl::common::TensorShape* seqstarts_shape,
    const void* seqstarts,
    const ppl::common::TensorShape* kvstarts_shape,
    const void* kvstarts,
    const ppl::common::TensorShape* attention_mask_shape,
    const void* attention_mask,
    const ppl::common::TensorShape* alibi_output_shape,
    const int64_t num_heads,
    void* alibi_output);

}}}}}

#endif
