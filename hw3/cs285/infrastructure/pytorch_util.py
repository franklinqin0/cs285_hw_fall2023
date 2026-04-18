from typing import Union

import torch
from torch import nn
import numpy as np

Activation = Union[str, nn.Module]


_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}

device = None


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
):
    """
    Builds a feedforward neural network

    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        scope: variable scope of the network

        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    mlp.to(device)
    return mlp


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(data: Union[np.ndarray, dict], **kwargs):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = np.asarray(data).copy()
        data = torch.frombuffer(data, dtype=_numpy_to_torch_dtype(data.dtype)).reshape(data.shape).clone()
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(device)


_NUMPY_TO_TORCH_DTYPE = {
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('bool'): torch.bool,
    np.dtype('uint8'): torch.uint8,
    np.dtype('int16'): torch.int16,
    np.dtype('int8'): torch.int8,
}

def _numpy_to_torch_dtype(np_dtype):
    return _NUMPY_TO_TORCH_DTYPE.get(np_dtype, torch.float32)


def to_numpy(tensor: Union[torch.Tensor, dict]):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.to("cpu").detach().numpy()
