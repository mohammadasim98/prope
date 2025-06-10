import sys
import os

import torch
import tqdm

# Enable highest precision
torch.set_default_dtype(torch.float64)

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH, ".."))

from prope_torch import prope_dot_product_attention as prope_torch, PropeDotProductAttention


def test_prope_torch():
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

    for _ in tqdm.tqdm(range(100), desc="prope as function"):
        out_torch_0 = prope_torch(
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

    prope = PropeDotProductAttention(
        head_dim=head_dim,
        cameras=cameras,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    for _ in tqdm.tqdm(range(10000), desc="prope as module"):
        out_torch_1 = prope(q, k, v, viewmats, Ks)

    torch.testing.assert_close(out_torch_0, out_torch_1)



if __name__ == "__main__":
    test_prope_torch()