#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> comm_on_GPU(torch::Tensor embed_A,torch::Tensor embed_B);

std::vector<at::Tensor> comm(torch::Tensor embed_A,torch::Tensor embed_B)
{
    return comm_on_GPU(embed_A,embed_B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("comm_cuda", &comm, "COMM (CUDA)");
 // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
