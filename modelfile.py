import torch

import comm_cuda

def communication(embA,embB):
    outputs = comm_cuda.comm_cuda(embA,embB)
    print(outputs)
    embC,embD = outputs
    return embC,embD

def main():
    batch_size = 16
    state_size = 128
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")

    embedding_A = torch.randn(batch_size, state_size, device=device_0)
    embedding_B = torch.randn(batch_size, state_size, device=device_1)
    print(embedding_A)
    print(embedding_B)

    embedding_C,embedding_D=communication(embedding_A,embedding_B)

    #print(embedding_C)
    #print(embedding_D)
    embedding_C_cpu=embedding_C.cpu()
    embedding_D_cpu=embedding_D.cpu()
    print(embedding_C_cpu)
    print(embedding_D_cpu)



if __name__ == '__main__':
    main()
