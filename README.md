## EQFT-量化微调算法

### Quickstart

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen_eqft.yaml
```

### 支持2/3/4-bit量化微调

### 函数说明

1、adapter.py增加函数_setup_eqft_tuning

```
def _setup_eqft_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))
    from .eqft.quantize_save import quantize_and_save
    base_dir, lora_dir = quantize_and_save(model_args, finetuning_args)
    if lora_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_dir,
            is_trainable=True if is_trainable else False,
            token=finetuning_args.token,
        )
    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model
```

2、在model下创建文件夹eqft，包含quantize_save.py以及eqft_utils.py

3、新增部分参数：

```
    token: Optional[str] = field(
        default=None,
        metadata={"help": "The access token to download model from HuggingFace Hub."}
    )
    bits: int = field(
        default=2,
        metadata={"help": "The quantized bits"}
    )
    iter: int = field(
        default=5,
        metadata={"help": "The alternating steps in EQFT"}
    )
    rank: int = field(
        default=16,
        metadata={"help": "The rank of the LoRA adapter"}
    )
    save_dir: str = field(
        default="./model_zoo/eqft/",
        metadata={"help": "Directory to save the quantized model"}
    )
```

### Results

|          |      | WIKI         |            |            | C4           |            |            |
| -------- | ---- | ------------ | ---------- | ---------- | ------------ | ---------- | ---------- |
|          | Bits | Qwen2.5-0.5B | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-0.5B | Qwen2.5-3B | Qwen2.5-7B |
| LoRA     | 16   | 13.41        | 7.56       | 5.67       | 22.62        | 14.28      | 12.54      |
| QPiSSA   | 4    | 13.92        | 7.75       | 5.83       | 23.61        | 14.57      | 12.81      |
| **EQFT** | 4    | **13.61**    | **7.70**   | **5.77**   | **23.01**    | **14.40**  | **12.65**  |
| QPiSSA   | 3    | 20.85        | 9.98       | 7.16       | N.A.         | 37.08      | 16.06      |
| **EQFT** | 3    | **16.09**    | **8.77**   | **6.57**   | **29.78**    | **15.98**  | **13.60**  |
| QPiSSA   | 2    | 21.70        | 10.42      | 7.53       | N.A.         | 39.75      | 16.15      |
| **EQFT** | 2    | **16.99**    | **9.09**   | **6.83**   | **30.57**    | **16.68**  | **13.85**  |

## MEQFT-混合精度量化微调算法

### Quickstart

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen_meqft.yaml
```

### 支持2/3-bit混合精度，通过ratio参数调整混合比例

### 函数说明

1、adapter.py增加函数_setup_meqft_tuning

```
def _setup_meqft_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))
    from .meqft.quantize_save_mix import quantize_and_save_mix
    base_dir, lora_dir = quantize_and_save_mix(model_args, finetuning_args)
    if lora_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_dir,
            is_trainable=True if is_trainable else False,
            token=finetuning_args.token,
        )
    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model
```

2、在model下创建文件夹meqft，包含quantize_save_mix.py以及meqft_utils.py

3、新增部分参数：

```
    token: Optional[str] = field(
        default=None,
        metadata={"help": "The access token to download model from HuggingFace Hub."}
    )
    bits: int = field(
        default=2,
        metadata={"help": "The quantized bits"}
    )
    iter: int = field(
        default=5,
        metadata={"help": "The alternating steps in EQFT"}
    )
    rank: int = field(
        default=16,
        metadata={"help": "The rank of the LoRA adapter"}
    )
    ratio: float = field(
        default=0.5,
        metadata={"help": "The ratio of the mixed-precision QAF"}
    )
    save_dir: str = field(
        default="./model_zoo/eqft/",
        metadata={"help": "Directory to save the quantized model"}
    )
```

### Results

|       |      | WIKI         |            |            | C4           |            |            |
| ----- | ---- | ------------ | ---------- | ---------- | ------------ | ---------- | ---------- |
|       | Bits | Qwen2.5-0.5B | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-0.5B | Qwen2.5-3B | Qwen2.5-7B |
| EQFT  | 3    | 16.09        | 8.77       | 6.57       | 29.78        | 15.98      | 13.60      |
| MEQFT | 2.5  | 16.19        | 8.86       | 6.69       | 30.42        | 16.45      | 13.67      |
| MEQFT | 2.25 | 16.79        | 8.99       | 6.77       | 30.50        | 16.74      | 13.89      |
| EQFT  | 2    | 16.99        | 9.09       | 6.83       | 30.57        | 16.68      | 13.85      |

## SNN-DLFT

### Quickstart

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3_snndlft.yaml
```

**SNN-DLFT** 是一种基于 **AdaLoRA** 的脉冲神经网络（SNN）PEFT 方法。遵循 **AdaLoRA**，我们也使用自适应秩分配来修改不同类型 LoRA 模块的秩（通过掩蔽 LoRA-E 矩阵实现）。

此外，我们引入了两个机制：

1. 我们采用动态层级策略，根据新的混合层重要性计算方法，在训练阶段的热身后自适应选择训练 **single** 层还是 **half** 层。
2. 我们将脉冲神经元引入网络，以实现低功耗的微调。

### 函数说明

1. SNNDLFT微调的主要代码在 ./src/llamafactory/model/snndlft

   - /src/llamafactory/mode/snndlft
     - layer.py  # snn_svdLinear
     - model.py  # SNNAdaLoRAModel
     - trainer.py # SNNAdaLoRA-Adapted Trainer

2. 在原有代码上修改了部分代码以适配SNNDLFT

   - /src/llamafactory/model/adapter.py

   1. 增加了_setup_adalora_tuning():设置AdaLoRA有关的config和获取模型

   2. 在init_adapter()中增加了调用AdaLoRA和SNNDLFT的判断

   3. 新增一些类的引入

      ```python
      from .snndlft import ada_get_peft_model, AdaLoraConfig
      ```

3. 新增部分参数：

   - 

     ```bash
     	target_r : Optional[int] = field(
             default=8,
             metadata={"help":"The target average rank of incremental matrix for AdaLoRA fine-tuning"}
         )
         init_r : Optional[int] = field(
             default=12,
             metadata={"help":"The initial rank for each incremental matrix for AdaLoRA fine tuning."}
         )
         deltaT : Optional[int] = field(
             default=1,
             metadata={"help":"The default Time step for update rank parten and layer-wise training."}
         ),
         tfinal: Optional[int] = field(
             default=0,
             metadata={"help":"Final Steps Training in a concrete rank parten and layer"}
         ),
         tinit: Optional[int] = field(
             default=0,
             metadata={"help":"Initial training step"}
         ),
         layer_wise: Literal["single", "half"] = field(
             default="half",
             metadata={"help": "Training strategy during SNNDLFT."},
         )
     ```

     ```python
     assert self.finetuning_type in ["lora", "freeze", "full", "adalora", "eqft", "meqft", "snndlft"]
     ```

4. 新增Adatrainer的引用

   - /src/llamafactory/train/pt/workflow.py

     ```python
     if finetuning_args.finetuning_type == 'adalora' or finetuning_args.finetuning_type == 'snndlft':
             # Initialize our Trainer
             trainer = AdaTrainer()
     ```

### 实验效果

提高模型的鲁棒性，在对抗数据集AdvGLUE上，SNNDLFT性能高于AdaLoRA和它的SNN版本

|                 | MNLI（m/mm-Acc） | QQP（Acc/F1） | QNLI（Acc） | SST2（Acc/F1) | RTE(Acc) |
| --------------- | ---------------- | ------------- | ----------- | ------------- | -------- |
| Ada             | 44.63/35.19      | 58.97/0.0     | 50.00       | 57.43/68.97   | 56.79    |
| Ada-SNN         | 47.11/30.25      | 51.28/48.65   | 61.49       | 41.22/28.1    | 69.14    |
| SNNDLFT（half） | 44.62/35.18      | 57.69/52.17   | 60.14       | 40.54/12.0    | 71.6     |
| SNNDLFT（one）  | 56.2/47.53       | 57.69/56.0    | 59.46       | 52.03/52.98   | 76.54    |

在普通数据集上，性能略微下降，但微调参数量减少，在debertav3模型上的效果

| Method           | Param     | MNLI（m/mm） | QQP（Acc/F1） | QNLI（Acc） | SST2（Acc) | COLA(Matthews) | STSB(score) | MRPC（Acc/F1） | RTE(Acc) |
| ---------------- | --------- | ------------ | ------------- | ----------- | ---------- | -------------- | ----------- | -------------- | -------- |
| Ada              | 90.6/90.5 | 91.2/88.44   | 94.42         | 95.64       | 69.35      | 91.66          | 89.95/92.79 | 89.17          |          |
| Ada-SNN          | 0.33M     | 89.79/90.3   | 89.99/86.89   | 94.01       | 95.18      | 66.79          | 91.28       | 89.22/92.28    | 83.39    |
| DLFT-SNN（half） | 0.16M     | 89.88/90.08  | 89.57/86.32   | 93.63       | 95.3       | 63.77          | 91.11       | 88.97/92.17    | 86.64    |
| DLFT-SNN（one）  | 0.027M    | 89.27/89.32  | 88.69/85.15   | 92.86       | 95.07      | 66.46          | 91.57       | 88.48/91.65    | 86.64    |

在LLaMA-3-1B模型上的效果：

| method          | Param   | wikitext2(ppl) |
| --------------- | ------- | -------------- |
| AdaLoRA-SNN     | 0.213M  | 9.35           |
| SNNDLFT-top-L/2 | 0.107M  | 9.39           |
| SNNDLFT-top-1   | 0.0134M | 9.64           |
| AdaLoRA         | 0.213M  | 8.88           |

