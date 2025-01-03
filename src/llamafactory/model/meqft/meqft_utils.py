from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available


def detect_outliers(data):
    mean_value = torch.mean(data)
    std_deviation = torch.std(data)
    threshold = 3 * std_deviation

    outliers = data[torch.abs(data - mean_value) > threshold]

    return outliers


class NFQuantizer:
    def __init__(self, num_bits=2, device="cuda", method="normal", block_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        if self.method == "normal":  # 如果是 "normal"，调用 create_normal_map 方法生成查找表
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == "uniform":
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError("Other quantization methods not supported yet.")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            # print("symmetric uniform quantization")
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            # print("asymmetric uniform quantization")
            table = torch.linspace(-1, 1, 2**num_bits)
        return table

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")

        variations = 2**num_bits
        if symmetric:
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            v2 = [0]
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        return values

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs

        weight_normed_expanded = weight_normed.unsqueeze(-1)

        # Reshape L to have the same number of dimensions as X_expanded
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)

        # Calculate the absolute difference between X_expanded and L_reshaped
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)

        # Find the index of the minimum absolute difference for each element
        qweight = torch.argmin(abs_diff, dim=-1)
        return qweight, max_abs

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()

        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs

        weight = weight.reshape(qweight.shape)

        return weight

    def quantize_block(self, weight):
        if len(weight.shape) != 2:
            raise ValueError(f"Only support 2D matrix, but your input has {len(weight.shape)} dimensions.")
        if weight.shape[0] * weight.shape[1] % self.block_size != 0:
            raise ValueError(
                f"Weight with shape ({weight.shape[0]} x {weight.shape[1]}) "
                f"is not divisible by block size {self.block_size}."
            )

        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B

        if self.method == "normal":
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L)
        elif self.method == "uniform":
            weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")

        weight_max = weight_max.unsqueeze(-1)  # (L, 1)
        weight_divabs = weight_block / weight_max  # (L, B)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs.unsqueeze(-1) - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        return qweight, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, original_shape):
        L_reshaped = self.norm_lookup_table.reshape(-1)

        # Map indices back to values
        qweight_real = L_reshaped[qweight]

        # Rescale using the max values
        weight = qweight_real * weight_max

        return weight.reshape(original_shape)


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}


@torch.no_grad()
def meqft_init(weight: Union[torch.Tensor, torch.nn.Parameter], num_bits: int, ratio: float, reduced_rank: int, num_iter=2):
    device = weight.device
    dtype = weight.dtype

    quantizer1 = NFQuantizer(num_bits=3, device=device, method="normal", block_size=32)
    quantizer2 = NFQuantizer(num_bits=2, device=device, method="normal", block_size=32)
    compute_device = device

    reduced_rank = int(reduced_rank)
    weight = weight.to(device=compute_device, dtype=torch.float32)
    V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
    Vr = V[:, reduced_rank:]
    Sr = S[reduced_rank:]
    Uhr = Uh[reduced_rank:, :]
    L = Vr @ torch.diag(torch.sqrt(Sr))
    R = torch.diag(torch.sqrt(Sr)) @ Uhr
    residual_ = weight - L @ R

    detect_outliers_column = []
    for column in range(residual_.shape[1]):
        outliers = detect_outliers(residual_[:, column])
        detect_outliers_column.append(len(outliers))

    sorted_indices = sorted(range(len(detect_outliers_column)), key=lambda x: detect_outliers_column[x],
                            reverse=True)
    extract_rate = sorted_indices[:int(ratio * len(sorted_indices))]
    extract_columns = sorted(extract_rate, reverse=False)
    remaining_columns = [i for i in range(weight.shape[1]) if i not in extract_columns]

    for i in range(num_iter):
        residual = weight - L @ R

        residua_H = residual[:, extract_columns]
        quantized_weight_H, max_abs_H, shape_H = quantizer1.quantize_block(residua_H)
        dequantized_weight_H = quantizer1.dequantize_block(quantized_weight_H, max_abs_H, shape_H)

        residua_L = residual[:, remaining_columns]
        quantized_weight_L, max_abs_L, shape_L = quantizer2.quantize_block(residua_L)
        dequantized_weight_L = quantizer2.dequantize_block(quantized_weight_L, max_abs_L, shape_L)

        dequantized_weight = torch.zeros_like(residual)
        dequantized_weight[:, extract_columns] = dequantized_weight_H
        dequantized_weight[:, remaining_columns] = dequantized_weight_L

        # Decompose the residual by SVD
        output = _low_rank_decomposition(weight - dequantized_weight, reduced_rank=reduced_rank)
        L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]

    lora_A, lora_B = R, L

    return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B
