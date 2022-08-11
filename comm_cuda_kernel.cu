#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void comm_cuda_kernel(
    scalar_t* __restrict__ embed_dst,
    const scalar_t* __restrict__ embed_local,
    const scalar_t* __restrict__ embed_remote,
    size_t tensor_size
)
{
    size_t th_idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(th_idx<tensor_size)
    {
        embed_dst[th_idx]=embed_local[th_idx]+embed_remote[th_idx];
    }
}

}

std::vector<at::Tensor> comm_on_GPU(torch::Tensor embed_A,torch::Tensor embed_B)
{
    cudaSetDevice(0);
    auto embed_C = torch::zeros_like(embed_A);
    cudaSetDevice(1);
    auto embed_D = torch::zeros_like(embed_B);
    const int threads = 1024;
    const int blocks_0 = (embed_C.numel() + threads - 1) / threads;
    const int blocks_1 = (embed_D.numel() + threads - 1) / threads;
    const auto tensor_size = embed_A.numel();
    cudaStream_t streamtable[2];
    for(int i=0;i<2;++i)
    {
        cudaSetDevice(i);
        cudaStreamCreate(&streamtable[i]);
    }
    cudaSetDevice(0);
    AT_DISPATCH_FLOATING_TYPES(embed_A.type(), "comm", ([&] {
        comm_cuda_kernel<scalar_t><<<blocks_0, threads, 0, streamtable[0]>>>(
            embed_C.data<scalar_t>(),
            embed_A.data<scalar_t>(),
            embed_B.data<scalar_t>(),
            tensor_size);
    }));
    cudaSetDevice(1);
    AT_DISPATCH_FLOATING_TYPES(embed_B.type(), "comm", ([&] {
        comm_cuda_kernel<scalar_t><<<blocks_1, threads, 0, streamtable[1]>>>(
            embed_D.data<scalar_t>(),
            embed_B.data<scalar_t>(),
            embed_A.data<scalar_t>(),
            tensor_size);
    }));
    for(int i=0;i<2;++i)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    return {embed_C,embed_D};
}
