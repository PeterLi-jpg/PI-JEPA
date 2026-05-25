import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    """2D decoder: token sequence -> (B, C, H, W) physical-space field."""

    def __init__(self, embed_dim, out_channels, image_size, patch_size):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size

        self.proj = nn.Linear(
            embed_dim,
            out_channels * patch_size * patch_size
        )

    def forward(self, z_full):
        """
        z_full: (B, N, D) — N = (image_size/patch_size)^2
        Returns: (B, C, image_size, image_size)
        """
        B, N, D = z_full.shape

        n = int(math.sqrt(N))

        if n * n != N:
            raise ValueError(f"Cannot reshape {N} tokens into square 2D grid")

        P = self.patch_size
        C = self.out_channels

        x = self.proj(z_full)

        x = x.view(B, n, n, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)

        x = x.contiguous().view(
            B,
            C,
            n * P,
            n * P
        )

        return x


def _as_triple(value):
    """Coerce scalar or 3-element list into a 3-tuple of ints."""
    if isinstance(value, int):
        return (value, value, value)
    if hasattr(value, "__len__") and len(value) == 3:
        return tuple(int(v) for v in value)
    raise ValueError(f"Expected int or 3-tuple; got {value!r}")


class Decoder3D(nn.Module):
    """3D decoder: token sequence -> (B, C, D, H, W) volumetric field.

    Supports both cubic and rectangular patches. image_size and patch_size
    may each be an int (cubic) or a 3-tuple (rectangular: D, H, W).
    """

    def __init__(self, embed_dim, out_channels, image_size, patch_size):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.image_size_dhw = _as_triple(image_size)
        self.patch_size_dhw = _as_triple(patch_size)
        # Back-compat scalar attrs
        self.image_size = max(self.image_size_dhw)
        self.patch_size = min(self.patch_size_dhw)

        pd, ph, pw = self.patch_size_dhw
        self.proj = nn.Linear(
            embed_dim,
            out_channels * pd * ph * pw,
        )

        gd = self.image_size_dhw[0] // pd
        gh = self.image_size_dhw[1] // ph
        gw = self.image_size_dhw[2] // pw
        self.grid_size_dhw = (gd, gh, gw)
        self.n_patches = gd * gh * gw

    def forward(self, z_full):
        """
        z_full: (B, N, D)  with N = gd * gh * gw
        Returns: (B, C, gd*pd, gh*ph, gw*pw)
        """
        B, N, D = z_full.shape
        gd, gh, gw = self.grid_size_dhw
        pd, ph, pw = self.patch_size_dhw
        C = self.out_channels

        expected_N = gd * gh * gw
        if N != expected_N:
            raise ValueError(
                f"Decoder3D expects {expected_N} tokens (grid {self.grid_size_dhw}); got {N}"
            )

        x = self.proj(z_full)
        # (B, N, C * pd * ph * pw)

        x = x.view(B, gd, gh, gw, C, pd, ph, pw)
        # Permute to (B, C, gd, pd, gh, ph, gw, pw) so consecutive memory walks
        # produce the volumetric voxel ordering when we collapse.
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.contiguous().view(B, C, gd * pd, gh * ph, gw * pw)
        return x
