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

#include "ppl/kernel/llm/cuda/pmx/alibi_mask.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template<int32_t total_heads>
__device__ float get_slope(int32_t head) {
    int32_t closest_power_of_2 = 1 << static_cast<int32_t>(floorf(log2f(static_cast<float>(total_heads))));
    if (head <= closest_power_of_2) {
        return powf(2.0, -8.0 * head / closest_power_of_2);
    } else {
        int32_t adjusted_head = 2 * (head - closest_power_of_2) - 1;
        return powf(2.0, -4.0 * adjusted_head / closest_power_of_2);
    }
}


template<int32_t VPT, int32_t TPB, bool MASK, int32_t total_heads>
__global__ void alibi_mask_kernel(
    const int64_t *seqstarts,
    const int64_t *kvstarts,
    const half *attention_mask,
    half *alibi_mask,
    int64_t seqdim, 
    int64_t kvdim
) 
{
    const int32_t bidx = blockIdx.y;
    const int32_t tid = threadIdx.x;
    const int32_t seqsbeg = seqstarts[bidx];
    const int32_t seqsend = seqstarts[bidx + 1];
    const int32_t kvbeg = kvstarts[bidx];
    const int32_t kvend = kvstarts[bidx + 1];
    const int32_t seqlen = seqsend - seqsbeg;
    const int32_t kvlen = kvend - kvbeg;
    
    int32_t total_elements = seqlen * kvlen;

    #pragma unroll
    for (int idx = threadIdx.x; idx < total_elements; idx += TPB) {
        int32_t seqpos = idx / kvlen;
        int32_t kvpos = idx % kvlen;

        float value = kvpos - kvlen - seqpos + seqlen;
        if (value <= 0.0f) {
            float slop = get_slope<total_heads>(blockIdx.x + 1);
            value = value * slop;
        } 
        else {
            value = -INFINITY;
        }
        int64_t write_idx = blockIdx.x * kvdim * seqdim +(seqsbeg + seqpos) * kvdim + kvbeg + kvpos;
        alibi_mask[write_idx] = __float2half(value);
    }
    // int32_t total_elements = seqlen * kvdim;
    // #pragma unroll
    // for (int idx = threadIdx.x; idx < total_elements; idx += TPB) {
    //     int32_t seqpos = idx / kvdim;
    //     int32_t kvpos = idx % kvdim;
    //     float value = 0.0f;
    //     int64_t write_idx = blockIdx.x * kvdim * seqdim +(seqsbeg + seqpos) * kvdim + kvpos;
    //     if (kvpos >= kvbeg && kvpos < kvend) {
    //         int32_t mask_pos = kvpos - kvbeg;
    //         value = mask_pos - kvlen - seqpos + seqlen;
    //         if (value <= 0.0f) {
    //             float slop = get_slope<total_heads>(blockIdx.x + 1);
    //             value = value * slop;
    //         } 
    //         else {
    //             value = -INFINITY;
    //         }
    //     }
    //     alibi_mask[write_idx] = __float2half(value);
    // }
    
    if (MASK) {
        int64_t total_mask_elements = seqlen * kvdim;
        const int64_t num_pack = total_mask_elements / VPT;

        half mask_local[VPT];
        half alibi_local[VPT];
        for(int32_t pack_id = tid; pack_id < num_pack; pack_id += TPB) {
            int64_t mask_idx = seqsbeg * kvdim + pack_id * VPT;
            int64_t alibi_idx = blockIdx.x * kvdim * seqdim + seqsbeg * kvdim + pack_id * VPT;
            copy<sizeof(half) * VPT>(&attention_mask[mask_idx], mask_local);
            copy<sizeof(half) * VPT>(&alibi_mask[alibi_idx], alibi_local);
            #pragma unroll
            for(int32_t it = 0; it < VPT; it++) {
                alibi_local[it] = mask_local[it] + alibi_local[it];
            }
            copy<sizeof(half) * VPT>(alibi_local, &alibi_mask[alibi_idx]);
        }
    }
}


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
    void* alibi_output
)
{
    // seqstarts (batch + 1)
    // kvstarts (batch + 1)
    // attention_mask (seqlen, kvlen)

    if (alibi_output_shape->GetDimCount() != 3) {
        LOG(ERROR) << "alibi_output's dim should be 3, however get " << alibi_output_shape->GetDimCount() << " dim 0: " << alibi_output_shape->GetDim(0);
        return ppl::common::RC_INVALID_VALUE;
    }

    const int32_t TPB = 256;
    constexpr int32_t VPT = 16 / sizeof(half);
    const int64_t batch = seqstarts_shape->GetDim(0) - 1;
    const int64_t kv_last_dim = alibi_output_shape->GetDim(2);
    const int64_t seq_last_dim = alibi_output_shape->GetDim(1);
    LOG(INFO) << "kv_last_dim " << kv_last_dim << " seq_last_dim " << seq_last_dim << " batch " << batch << " num heads " << num_heads;
    dim3 gridDim(num_heads, batch);

    if (attention_mask != nullptr) {
        switch(num_heads)
        {
            case 40:
                alibi_mask_kernel<VPT, TPB, true, 40>
                <<<gridDim, TPB, 0, stream>>>(
                    (const int64_t*)seqstarts, 
                    (const int64_t*)kvstarts, 
                    (const half*)attention_mask, 
                    (half*)alibi_output, 
                    seq_last_dim, 
                    kv_last_dim
                );
                break;
            default:
                LOG(ERROR) << "alibi mask do not support heads " << num_heads;
                return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        switch(num_heads)
        {
            case 40:
                alibi_mask_kernel<VPT, TPB, false, 40>
                <<<gridDim, TPB, 0, stream>>>(
                    (const int64_t*)seqstarts, 
                    (const int64_t*)kvstarts, 
                    nullptr, 
                    (half*)alibi_output, 
                    seq_last_dim, 
                    kv_last_dim
                );
                break;
            default:
                LOG(ERROR) << "alibi mask do not support heads " << num_heads;
                return ppl::common::RC_UNSUPPORTED;
        }
    }
    return ppl::common::RC_SUCCESS;

}


}}}}}