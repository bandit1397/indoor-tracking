"""PC(파이썬)와 휴대폰(웹)이 똑같이 쓰는 이미지 → 특징 벡터 변환.

웹의 app.js `preprocess()`와 반드시 같은 방식이어야 위치 매칭이 맞는다:
가운데 정사각형으로 자르기 → 224x224 → RGB 0~255 (정규화는 모델 안에서 처리).
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 224
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "web" / "model" / "embed.onnx"


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    crop = img_bgr[y:y + s, x:x + s]
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb.transpose(2, 0, 1)  # CHW


class Embedder:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        if not Path(model_path).exists():
            raise SystemExit(f"모델이 없습니다: {model_path}\n먼저 실행하세요: python mapper/export_model.py")
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, imgs_bgr: list[np.ndarray], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(imgs_bgr), batch):
            x = np.stack([preprocess(im) for im in imgs_bgr[i:i + batch]])
            out.append(self.sess.run(None, {self.input_name: x})[0])
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)
