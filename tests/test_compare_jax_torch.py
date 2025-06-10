import sys
import os

import numpy as np
import torch
import jax

# Enable highest precision
jax.config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH, ".."))

from prope_jax import prope_dot_product_attention as prope_jax
from prope_torch import prope_dot_product_attention as prope_torch


def test_compare_jax_torch():
    torch.manual_seed(42)
    cameras = 3
    patches_x = 8
    patches_y = 8
    image_width = 128
    image_height = 128

    batch = 2
    seqlen = cameras * patches_x * patches_y
    num_heads = 4
    head_dim = 16

    q = torch.randn(batch, num_heads, seqlen, head_dim)
    k = torch.randn(batch, num_heads, seqlen, head_dim)
    v = torch.randn(batch, num_heads, seqlen, head_dim)

    viewmats = torch.eye(4).repeat(batch, cameras, 1, 1)
    Ks = torch.rand(batch, cameras, 3, 3)

    out_jax = prope_jax(
        q.permute(0, 2, 1, 3).numpy(),
        k.permute(0, 2, 1, 3).numpy(),
        v.permute(0, 2, 1, 3).numpy(),
        viewmats=viewmats.numpy(),
        Ks=Ks.numpy(),
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    out_torch = prope_torch(
        q,
        k,
        v,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )

    np.testing.assert_allclose(
        out_jax, out_torch.permute(0, 2, 1, 3).numpy(), atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
    test_compare_jax_torch()