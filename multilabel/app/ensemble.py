"""app_v2 core: odir7 3-model DINOv2 residual-attention ensemble (final_20261030)."""
from __future__ import annotations
import json, sys
import numpy as np, torch
from PIL import Image

from app import settings

ROOT = settings.PROJECT_ROOT
CKPT_DIR = settings.CHECKPOINT_DIR
MANIFEST = json.loads(settings.MANIFEST_PATH.read_text(encoding="utf-8"))
for p in (ROOT / "src", ROOT / "training" / "dual_tax"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import retina_multilabel.model as rm

def _nopre(name, img_size=None, *, dynamic_img_size=False):
    import timm
    kw = {"pretrained": False, "num_classes": 0}
    if img_size is not None: kw["img_size"] = img_size
    if dynamic_img_size: kw["dynamic_img_size"] = True
    return timm.create_model(name, **kw)
rm._create_timm_backbone = _nopre  # full weights come from the checkpoint; no hub download
from dual_tax.vit_residual_attn import build_vit_residual_attn

LABELS = MANIFEST["labels"]; THRESHOLDS = np.array(MANIFEST["thresholds"], dtype=np.float32)
DISPLAY = MANIFEST["label_display"]
_MEAN = torch.tensor(MANIFEST["preprocessing"]["mean"]).view(1, 3, 1, 1)
_STD = torch.tensor(MANIFEST["preprocessing"]["std"]).view(1, 3, 1, 1)

def preprocess(img: Image.Image, size: int):
    pil = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.array(pil, dtype=np.uint8)
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0
    return arr, (x - _MEAN) / _STD

class Ensemble:
    def __init__(self):
        self.members = []
        for m in MANIFEST["members"]:
            ck = torch.load(CKPT_DIR / m["checkpoint"], map_location="cpu", weights_only=False)
            model = build_vit_residual_attn(len(LABELS), img_size=m["img_size"])
            sd = ck.get("model_state_dict", ck.get("model", ck))
            mi, un = model.load_state_dict(sd, strict=False)
            if mi or un: raise RuntimeError(f"{m['run']}: missing={mi[:3]} unexpected={un[:3]}")
            for p in model.parameters(): p.requires_grad_(False)
            self.members.append((m["run"], m["img_size"], model.eval()))

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict:
        per = {}
        for run, size, model in self.members:
            per[run] = torch.sigmoid(model(preprocess(img, size)[1]))[0].numpy()
        prob = np.mean(list(per.values()), 0)
        pos = prob >= THRESHOLDS
        return dict(labels=LABELS, prob=prob, thresholds=THRESHOLDS, positive=pos, per_member=per)

    def gradcam(self, img: Image.Image, cls: int, member: int = 0):
        """Token Grad-CAM on norm1 of the last block of one member (default v2_518)."""
        run, size, model = self.members[member]
        arr, x = preprocess(img, size); store = {}
        def hook(mod, i, o): o.retain_grad(); store["a"] = o
        h = model.backbone.blocks[-1].norm1.register_forward_hook(hook)
        with torch.enable_grad():
            logits = model(x.requires_grad_(True)); h.remove(); logits[0, cls].backward()
        npre = getattr(model.backbone, "num_prefix_tokens", 1)
        a, g = store["a"][0, npre:], store["a"].grad[0, npre:]
        cam = torch.relu((a * g.mean(0)).sum(-1)); s = int(round(cam.numel() ** 0.5))
        cam = cam.reshape(s, s).detach().numpy(); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)) / 255.0
        return arr, cam

def overlay(arr, cam, alpha=0.45):
    import matplotlib.cm as cm
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.float32)
    return (arr * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
