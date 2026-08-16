import torch
from monai.transforms import MapTransform

# (window_level, window_width) for each clinically-motivated window
WINDOWS = {
    "lung": (-600, 1500),
    "soft_tissue": (40, 400),
    "bone": (400, 1500),
}


def _window_to_unit_range(volume: torch.Tensor, wl: float, ww: float) -> torch.Tensor:
    lo, hi = wl - ww / 2, wl + ww / 2
    clipped = torch.clamp(volume, lo, hi)
    return (clipped - lo) / (hi - lo)


class MultiWindowIntensityd(MapTransform):

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vol = d[key]  # [1, D, H, W], raw HU
            channels = [
                _window_to_unit_range(vol[0], wl, ww)
                for wl, ww in WINDOWS.values()
            ]
            d[key] = torch.stack(channels, dim=0)  # [3, D, H, W]
        return d


def volume_to_multiwindow_slices(volume: torch.Tensor, num_slices: int, slice_size: int,
                                  imagenet_mean: torch.Tensor, imagenet_std: torch.Tensor) -> torch.Tensor:

    depth = volume.shape[1]
    idxs = torch.linspace(0, depth - 1, num_slices).long()
    slices = volume[:, idxs, :, :]           # [3, K, H, W]
    slices = slices.permute(1, 0, 2, 3)       # [K, 3, H, W]
    slices = torch.nn.functional.interpolate(
        slices, size=(slice_size, slice_size), mode="bilinear", align_corners=False
    )
    slices = (slices - imagenet_mean) / imagenet_std
    return slices