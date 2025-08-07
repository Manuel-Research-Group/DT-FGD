import torch

# test cuda
def test_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

test_cuda()