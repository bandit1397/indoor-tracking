"""무료 공개 AI 모델(MobileNetV3, ImageNet 학습)을 장소 인식용 ONNX 파일로 변환한다.

한 번만 실행하면 된다. 결과 파일 web/model/embed.onnx 는
PC의 지도 만들기(build_map.py)와 휴대폰 웹페이지가 같이 사용한다.

    python mapper/export_model.py            # small (약 4MB, 휴대폰에서 빠름)
    python mapper/export_model.py --large    # large (약 12MB, 더 정확)
"""
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from common import DEFAULT_MODEL, INPUT_SIZE

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PlaceEmbed(nn.Module):
    """RGB 0~255 이미지 → 길이 1로 정규화된 특징 벡터 (GeM 풀링)."""

    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        f = self.features((x - self.mean) / self.std).clamp(min=1e-6)
        g = f.pow(3).mean(dim=(2, 3)).pow(1.0 / 3)
        return F.normalize(g, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="더 크고 정확한 모델 사용")
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    if args.large:
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    else:
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model = PlaceEmbed(net.features).eval()

    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE) * 255
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"저장 완료: {args.out} (벡터 크기 {model(dummy).shape[1]})")


if __name__ == "__main__":
    main()
