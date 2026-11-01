"""Nonnegative top-K visual relation with a fixed self edge."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class GraphConfig:
    graph_topk: int = 64
    graph_temperature: float = .07
    graph_min_similarity: float = 0.
    graph_chunk: int = 128
    eps: float = 1e-8
    def __post_init__(self):
        if self.graph_topk < 1 or self.graph_temperature <= 0:
            raise ValueError('K and graph temperature must be positive')

@dataclass
class SparseRelation:
    indices: torch.Tensor  # [B,N,K], self is column zero
    weights: torch.Tensor  # [B,N,K], nonnegative, row sum one
    chunk: int = 128

    @classmethod
    def from_features(cls, features, cfg):
        d = F.normalize(features.float(), dim=-1, eps=cfg.eps)
        if d.ndim != 3 or d.shape[1] == 0:
            raise ValueError('DINO features must be [B,N,D] with N>0')
        b, n, _ = d.shape
        informative = features.float().norm(dim=-1) > 1e-6
        k = min(cfg.graph_topk, n)
        index_chunks, weight_chunks = [], []
        for start in range(0, n, cfg.graph_chunk):
            end = min(start + cfg.graph_chunk, n)
            q = d[:, start:end]
            sim = q @ d.transpose(1, 2)
            own = torch.arange(start, end, device=d.device)[None, :, None].expand(b, -1, -1)
            self_sim = sim.gather(2, own)
            sim.masked_fill_(~informative[:, None, :], torch.finfo(sim.dtype).min)
            sim.scatter_(2, own, -torch.inf)
            other_sim, other_idx = sim.topk(k - 1, dim=-1, sorted=True)
            values = torch.cat((self_sim, other_sim), -1)
            indices = torch.cat((own, other_idx), -1)
            valid = (values >= cfg.graph_min_similarity) & informative[:, start:end, None]
            valid[..., 0] = True  # no neighbours -> exact self fallback
            weights = (values / cfg.graph_temperature).masked_fill(~valid, -torch.inf).softmax(-1)
            index_chunks.append(indices)
            weight_chunks.append(weights)
        return cls(torch.cat(index_chunks, 1), torch.cat(weight_chunks, 1), cfg.graph_chunk)

    def apply(self, features):
        if features.shape[:2] != self.indices.shape[:2]:
            raise ValueError('Relation and features must share batch and patch dimensions')
        x = features.float()
        batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
        chunks = []
        for start in range(0, x.shape[1], self.chunk):
            idx = self.indices[:, start:start + self.chunk]
            neighbours = x[batch, idx]
            chunks.append((neighbours * self.weights[:, start:start + self.chunk, :, None]).sum(2))
        return torch.cat(chunks, 1)
