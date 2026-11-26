"""The final model of the report: Branch 1 (fine-tuned BioCLIP, text tables re-fitted with synthetic embeddings)
ensembled with Branch 2 (MaPLe prompt learning on frozen BioCLIP, 3 seeds).

    S(k) = lam * S1(k) + (1 - lam) * S2(k) + bias * [k is unseen]
    S1(k) = w * log p(pair k) + log p(crop of k) + log p(disease of k) + beta * M(k)     (mean over image + flip)
    S2(k) = mean over 3 seeds of scale * cos(image, MaPLe text feature of k)

lam, w and bias were chosen on validation (MaPLe's split 0); the test set chose nothing. The scoring and the image
preprocessing are the same as in the evaluation code, so the app reproduces the reported test results.
"""
import copy
import io
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
BIOCLIP = 'hf-hub:imageomics/bioclip'
B1_NORM = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))   # Branch 1 training normalisation
B1_TF = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(*B1_NORM)])


def _open(data):
    img = Image.open(io.BytesIO(data))
    return img


def branch1_input(data):
    """As the evaluation cache: JPEG draft, short side 256 px (bicubic, only if larger), saved as JPEG q95;
    then resize 224, centre crop 224."""
    img = _open(data)
    img.draft('RGB', (512, 512))
    img = img.convert('RGB')
    s = 256 / min(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=95)
    return B1_TF(Image.open(buf).convert('RGB'))[None]


def maple_input(data):
    """As MaPLe's cache: short side 256 px (bicubic), centre crop 256; then centre crop 224, pixels / 255."""
    img = _open(data).convert('RGB')
    s = 256 / min(img.size)
    img = img.resize((max(256, round(img.width * s)), max(256, round(img.height * s))), Image.BICUBIC)
    left, top = (img.width - 256) // 2, (img.height - 256) // 2
    img = img.crop((left, top, left + 256, top + 256))
    return T.functional.to_tensor(T.functional.center_crop(img, 224))[None]


def _run_blocks(transformer, x, deep, put):
    for i, r in enumerate(transformer.resblocks):
        if 1 <= i <= len(deep):
            x = put(x, deep[i - 1])
        x = r(x) if getattr(transformer, 'batch_first', False) else r(x.transpose(0, 1)).transpose(0, 1)
    return x


def encode_image_prompted(visual, x, prompts):
    """BioCLIP image encoder with MaPLe image prompts: prompts[0] is appended to the input tokens, prompts[l] replaces
    them before block l (from the MaPLe notebook). x: [B, 3, 224, 224] normalised; prompts: [depth, B, k, 768]."""
    x = visual.conv1(x).flatten(2).transpose(1, 2)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], 1) + visual.positional_embedding.to(x.dtype)
    k = prompts.shape[2]
    x = visual.ln_pre(torch.cat([x, prompts[0]], 1))
    x = _run_blocks(visual.transformer, x, prompts[1:], lambda x, p: torch.cat([x[:, :-k], p], 1))
    x = x[:, :-k]
    if getattr(visual, 'final_ln_after_pool', False):
        pooled = visual.ln_post(visual._global_pool(x)[0])
    else:
        pooled = visual._global_pool(visual.ln_post(x))[0]
    return pooled @ visual.proj


class FinalModel:
    def __init__(self, weights, root=HERE, device='cpu'):
        """weights: folder with branch1_a.pt, branch1_b.pt, heads.pt (downloaded from Hugging Face by the app)."""
        root, weights = Path(root), Path(weights)
        self.device = device
        self.labels = json.loads((root / 'labels.json').read_text(encoding='utf-8'))
        clip, _, _ = open_clip.create_model_and_transforms(BIOCLIP)
        self.bioclip = clip.visual.to(device).eval().requires_grad_(False)       # frozen BioCLIP (Branch 2)
        del clip                                                                 # text encoders are not needed
        self.branch1 = copy.deepcopy(self.bioclip)                               # + our fine-tuned layers (Branch 1)
        sd = self.branch1.state_dict()
        for f in ('branch1_a.pt', 'branch1_b.pt'):
            sd.update({k: v.float() for k, v in torch.load(weights / f, map_location='cpu').items()})
        self.branch1.load_state_dict(sd)
        h = torch.load(weights / 'heads.pt', map_location='cpu')
        dev = lambda t: t.float().to(device)
        b1, mp, self.ens = h['b1'], h['maple'], h['ens']
        self.TP, self.TC, self.TD = dev(b1['t_pair']), dev(b1['t_crop']), dev(b1['t_dis'])
        self.k1, self.w, self.beta, self.gamma = b1['scale'], b1['w'], b1['beta'], b1['gamma']
        self.keys = F.normalize(dev(h['memory']['keys']), dim=-1)
        self.kcrop, self.kdis = h['memory']['crop'].long().to(device), h['memory']['dis'].long().to(device)
        self.files = h['memory']['files']
        self.mtext, self.mprompts = dev(mp['text']), dev(mp['prompts'])          # [3, 333, 512], [3, depth, k, 768]
        self.k2, self.mmean, self.mstd = mp['scale'], dev(mp['mean']).view(1, 3, 1, 1), dev(mp['std']).view(1, 3, 1, 1)
        L = self.labels
        self.PC = torch.tensor(L['pair_crop'], device=device)
        self.PD = torch.tensor(L['pair_dis'], device=device)
        self.unseen = torch.zeros(len(L['pairs']), device=device)
        self.unseen[L['unseen']] = 1.0

    def _class_max(self, sim, labels, n):
        idx = labels[None].expand(len(sim), -1)
        return torch.full((len(sim), n), -1.0, device=sim.device).scatter_reduce(1, idx, sim, 'amax', include_self=True)

    @torch.no_grad()
    def branch1_parts(self, x):
        """x: [B, 3, 224, 224] (Branch 1 preprocessing). Returns S1 without bias, crop / disease probabilities, and
        the image embedding used by the memory and the nearest-neighbour panel."""
        x = x.to(self.device)
        V = [F.normalize(self.branch1(v).float(), dim=-1) for v in (x, torch.flip(x, [3]))]
        k = self.k1
        A = sum(F.log_softmax(k * v @ self.TP.T, 1) for v in V) / 2
        lc = [F.log_softmax(k * v @ self.TC.T, 1) for v in V]
        ld = [F.log_softmax(k * v @ self.TD.T, 1) for v in V]
        B = sum(c[:, self.PC] + d[:, self.PD] for c, d in zip(lc, ld)) / 2
        v = F.normalize(V[0] + V[1], dim=-1)
        sim = v @ self.keys.T
        M = (torch.exp(-self.gamma * (1 - self._class_max(sim, self.kcrop, len(self.labels['crops']))))[:, self.PC]
             + torch.exp(-self.gamma * (1 - self._class_max(sim, self.kdis, len(self.labels['diseases']))))[:, self.PD])
        pc, pd = (sum(t) / 2 for t in (lc, ld))
        return self.w * A + B + self.beta * M, pc.exp(), pd.exp(), sim

    @torch.no_grad()
    def branch2(self, x):
        """x: [B, 3, 224, 224] in [0, 1] (MaPLe preprocessing). Mean over the 3 seeds of the scaled cosine logits."""
        x = ((x.to(self.device) - self.mmean) / self.mstd)
        n, s = len(x), len(self.mtext)
        xs = x.repeat(s, 1, 1, 1)                                                 # seed-major: [s * n, ...]
        prm = self.mprompts[:, None].expand(-1, n, -1, -1, -1).reshape(s * n, *self.mprompts.shape[1:]).transpose(0, 1)
        z = F.normalize(encode_image_prompted(self.bioclip, xs, prm).float(), dim=-1).view(s, n, -1)
        return (self.k2 * torch.einsum('snd,skd->snk', z, self.mtext)).mean(0)

    @torch.no_grad()
    def scores(self, x1, x2, bias=True):
        S1, pc, pd, sim = self.branch1_parts(x1)
        S = self.ens['lam'] * S1 + (1 - self.ens['lam']) * self.branch2(x2)
        if bias:
            S = S + self.ens['bias'] * self.unseen
        return S, pc, pd, sim

    def predict(self, data):
        """data: image file bytes -> (display probabilities over 333 pairs, crop probs, disease probs, memory similarities)."""
        S, pc, pd, sim = self.scores(branch1_input(data), maple_input(data))
        return (S[0] / self.ens.get('T', 1.0)).softmax(0).cpu(), pc[0].cpu(), pd[0].cpu(), sim[0].cpu()
