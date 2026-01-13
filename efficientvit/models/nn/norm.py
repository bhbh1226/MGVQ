from typing import Optional
import os

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "RMSNorm2d", "build_norm", "reset_bn", "set_norm_eps"]

# Triton 사용 가능 여부 확인 (환경변수로 강제 비활성화 가능)
TRITON_AVAILABLE = False
if os.environ.get("DISABLE_TRITON", "0") != "1":
    try:
        from efficientvit.models.nn.triton_rms_norm import TritonRMSNorm2dFunc
        # 실제로 Triton이 작동하는지 테스트
        _test_tensor = torch.randn(1, 4, 2, 2, device='cuda' if torch.cuda.is_available() else 'cpu')
        _test_weight = torch.ones(4, device=_test_tensor.device)
        _test_bias = torch.zeros(4, device=_test_tensor.device)
        _ = TritonRMSNorm2dFunc.apply(_test_tensor, _test_weight, _test_bias, 1e-6)
        TRITON_AVAILABLE = True
        print("[norm.py] Triton RMSNorm2d is available and working.")
    except Exception as e:
        print(f"[norm.py] Triton not available, using PyTorch fallback. Reason: {type(e).__name__}")
        TRITON_AVAILABLE = False
else:
    print("[norm.py] Triton disabled via DISABLE_TRITON environment variable.")


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class RMSNorm2d(nn.LayerNorm):
    """Pure PyTorch implementation of RMSNorm2d (fallback for Triton)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight + bias
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        out = x / rms
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    """RMSNorm2d with Triton acceleration (falls back to PyTorch if unavailable)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TRITON_AVAILABLE:
            return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)
        else:
            # PyTorch fallback
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            out = x / rms
            if self.elementwise_affine:
                out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return out


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
    "rms2d": RMSNorm2d,  # Pure PyTorch fallback
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    import copy

    import torch.nn.functional as F
    from tqdm import tqdm

    from efficientvit.apps.utils import AverageMeter, is_master, sync_tensor
    from efficientvit.models.utils import get_device, list_join

    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter(is_distributed=False)
            bn_var[name] = AverageMeter(is_distributed=False)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_mean = sync_tensor(batch_mean, reduce="cat")
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                        batch_var = sync_tensor(batch_var, reduce="cat")
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="reset bn", disable=not progress_bar or not is_master()) as t:
            for images in data_loader:
                images = images.to(get_device(tmp_model))
                tmp_model(images)
                t.set_postfix(
                    {
                        "bs": images.size(0),
                        "res": list_join(images.shape[-2:], "x"),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
