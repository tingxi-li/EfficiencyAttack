"""Input-level defenses applied at the producer (c1_img).

All defenses operate on the numpy array as it comes off `ultra_fast_load`:
shape (1, 3, H, W), dtype float32, value range [0, 1].

`apply(img_np)` returns (transformed_or_None, label):
- transformed_or_None: the transformed image, or None to drop the input
- label: short string describing what happened (passed/drop)
"""
import os
import numpy as np


def _featurize_for_svm(img_np):
    """Lightweight features tuned to expose teaspoon high-frequency noise.

    - 16x16 downsample (768 dims): coarse content
    - per-channel mean/std (6 dims)
    - per-channel total variation L1 (3 dims): sensitive to adversarial high-freq
    - per-channel high-pass-energy (residual after 3x3 box blur, mean abs) (3 dims)
    - per-channel histogram (8 bins) of pixel values (24 dims)
    - per-channel histogram of |grad_x|+|grad_y| (8 bins) (24 dims)
    Total ~830 dims. Still O(ms) to compute and a single LinearSVC dot product.
    """
    x = img_np[0]  # (3, H, W) float32 in [0,1]
    H, W = x.shape[1], x.shape[2]

    stride = max(1, H // 16)
    ds = x[:, ::stride, ::stride][:, :16, :16].reshape(-1).astype(np.float32)

    mean = x.mean(axis=(1, 2)).astype(np.float32)
    std = x.std(axis=(1, 2)).astype(np.float32)

    dx = np.abs(np.diff(x, axis=2))
    dy = np.abs(np.diff(x, axis=1))
    tv = (dx.mean(axis=(1, 2)) + dy.mean(axis=(1, 2))).astype(np.float32)

    # high-pass via box-blur residual: blur ~ uniform 3x3
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    blur = (
        pad[:, :-2, :-2] + pad[:, :-2, 1:-1] + pad[:, :-2, 2:] +
        pad[:, 1:-1, :-2] + pad[:, 1:-1, 1:-1] + pad[:, 1:-1, 2:] +
        pad[:, 2:, :-2] + pad[:, 2:, 1:-1] + pad[:, 2:, 2:]
    ) / 9.0
    hp = np.abs(x - blur).mean(axis=(1, 2)).astype(np.float32)

    bins = 8
    pix_hist = np.stack([np.histogram(x[c], bins=bins, range=(0, 1))[0] for c in range(3)])
    pix_hist = (pix_hist / max(H * W, 1)).astype(np.float32).reshape(-1)

    grad_mag = (
        np.pad(dx, ((0, 0), (0, 0), (0, 1)), mode="edge")
        + np.pad(dy, ((0, 0), (0, 1), (0, 0)), mode="edge")
    )
    gm_max = float(grad_mag.max()) if grad_mag.size else 1.0
    gm_max = gm_max if gm_max > 1e-6 else 1.0
    grad_hist = np.stack(
        [np.histogram(grad_mag[c], bins=bins, range=(0, gm_max))[0] for c in range(3)]
    )
    grad_hist = (grad_hist / max(H * W, 1)).astype(np.float32).reshape(-1)

    return np.concatenate([ds, mean, std, tv, hp, pix_hist, grad_hist]).astype(np.float32)


class InputDefense:
    name = "base"

    def apply(self, img_np):
        raise NotImplementedError


class GaussianNoiseDefense(InputDefense):
    name = "gaussian"

    def __init__(self, sigma=5.0 / 255.0, seed=0):
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def apply(self, img_np):
        noise = self.rng.normal(0.0, self.sigma, img_np.shape).astype(img_np.dtype)
        out = img_np + noise
        np.clip(out, 0.0, 1.0, out=out)
        return out, "passed"


class SpatialSmoothingDefense(InputDefense):
    name = "smoothing"

    def __init__(self, kernel=3, strength=0.8):
        """strength in [0,1]: 0 = no filtering, 1 = full median replacement.
        Output = strength * median + (1 - strength) * original.
        """
        from scipy.ndimage import median_filter
        self._median_filter = median_filter
        self.kernel = int(kernel)
        self.strength = float(strength)

    def apply(self, img_np):
        x = img_np[0]
        med = np.empty_like(x)
        for c in range(x.shape[0]):
            med[c] = self._median_filter(x[c], size=self.kernel, mode="reflect")
        out = (self.strength * med + (1.0 - self.strength) * x).astype(x.dtype)
        return out[None], "passed"


class SVMDefense(InputDefense):
    name = "svm"

    def __init__(self, model_path):
        import joblib
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SVM defense model not found at {model_path}")
        bundle = joblib.load(model_path)
        # Backward compat: if old artifact (raw classifier), wrap with tau=0
        if isinstance(bundle, dict) and "clf" in bundle:
            self.clf = bundle["clf"]
            self.tau = float(bundle.get("tau", 0.0))
        else:
            self.clf = bundle
            self.tau = 0.0

    def apply(self, img_np):
        feat = _featurize_for_svm(img_np).reshape(1, -1)
        score = float(self.clf.decision_function(feat)[0])
        if score > self.tau:
            return None, "drop"
        return img_np, "passed"


def build_defense(name, **kwargs):
    if name in (None, "none", ""):
        return None
    if name == "gaussian":
        return GaussianNoiseDefense(
            sigma=float(kwargs.get("sigma", 5.0 / 255.0)),
            seed=int(kwargs.get("seed", 0)),
        )
    if name == "smoothing":
        return SpatialSmoothingDefense(kernel=int(kwargs.get("kernel", 3)))
    if name == "svm":
        return SVMDefense(model_path=kwargs.get("model_path", "svm_defense.joblib"))
    raise ValueError(f"unknown defense: {name}")
