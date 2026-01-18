from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int, aug_cfg: Dict[str, Any]) -> T.Compose:
    scale_min = float(aug_cfg.get("random_resized_crop_scale_min", 0.7))
    scale_max = float(aug_cfg.get("random_resized_crop_scale_max", 1.0))
    hflip_prob = float(aug_cfg.get("hflip_prob", 0.5))
    color_jitter = bool(aug_cfg.get("color_jitter", True))

    t = [
        T.RandomResizedCrop(image_size, scale=(scale_min, scale_max)),
        T.RandomHorizontalFlip(p=hflip_prob),
    ]
    if color_jitter:
        t.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    t += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(t)


def build_eval_transforms(image_size: int, aug_cfg: Dict[str, Any]) -> T.Compose:
    resize_shorter = int(aug_cfg.get("resize_shorter", int(image_size * 1.07)))
    center_crop = int(aug_cfg.get("center_crop", image_size))
    return T.Compose(
        [
            T.Resize(resize_shorter),
            T.CenterCrop(center_crop),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
















