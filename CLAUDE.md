# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch learning playground covering foundational deep learning concepts — tensor ops, autograd, model building, training loops, transfer learning, ONNX export, and a credit card fraud detection end-to-end example. Uses Apple Silicon MPS backend.

## Commands

```bash
# Run any script directly (no build step, no package.json)
python <script.py>

# Check PyTorch MPS availability
python who_is_best_accerator.py
```

Python 3.12+ is expected. Dependencies: `torch`, `torchvision`, `numpy`, `scikit-learn`, `onnxruntime`, `onnx`, `matplotlib`, `pandas`.

## Project Structure

```
├── basic/                          # Foundational PyTorch concepts
│   ├── tensor.py                   # Tensor creation, indexing, reshape, math ops
│   ├── auto_grad.py                # Automatic differentiation, requires_grad, backward()
│   ├── dataset.py                  # Dataset/DataLoader, FashionMNIST, custom datasets
│   ├── classify_images.py          # nn.Module subclass, Softmax, named_parameters
│   ├── optimize_param.py           # Full train/test loop on FashionMNIST
│   ├── model_save_load.py          # state_dict save/load, pretrained models
│   └── fine_tuning.py              # Transfer learning: freeze features, retrain classifier
├── credit_card_fraud_dataset.py    # FraudDataset (custom Dataset) + FraudModel (Sequential MLP)
├── credit_card_fraud_detection.py  # End-to-end: synthetic data, train, eval, ONNX export
├── load_onnx_file.py               # ONNX Runtime inference on exported model
├── main.py                         # Embedding layer vs one-hot demonstration
├── cnn_linear.py                   # Minimal Linear layer creation example
├── who_is_best_accerator.py        # MPS / accelerator detection utility
├── assets/                         # Trained ONNX models, downloaded datasets
└── 202601_CV/                      # Computer vision learning notes (date-organized subdirs)
```

## Key Architecture Patterns

- **Device handling**: Use `torch.accelerator.current_accelerator()` for device detection (MPS on Apple Silicon), move model and tensors with `.to(device)`.
- **Standard training loop**: forward → loss → `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`, with `model.train()`/`model.eval()` and `torch.no_grad()` for evaluation.
- **Model definition**: Inherit `nn.Module`, define layers in `__init__`, implement `forward()`. `nn.Sequential` for simple stacking.
- **Dataset**: Subclass `torch.utils.data.Dataset` with `__len__` and `__getitem__`; wrap with `DataLoader` for batching.
- **Transfer learning**: Freeze feature layers (`param.requires_grad = False`), replace the classifier head, train only unfrozen params.
- **Model export**: `torch.onnx.export()` with dynamic axes for variable batch sizes; inference via `onnxruntime.InferenceSession`.
- **Comments are in Chinese** throughout the codebase.

## Common Tasks

- **Check available device**: `torch.accelerator.current_accelerator()` or `torch.backends.mps.is_available()`
- **ONNX export**: Pass a `dummy_input` matching the model's expected shape to `torch.onnx.export()` (see `credit_card_fraud_detection.py:120-134`)
- **Early stopping**: Track best validation metric with patience counter (see `credit_card_fraud_detection.py:61-62,106-116`)
