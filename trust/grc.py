"""Graph Residual Correction, manuscript Eqs. (11)--(21).

Only the complete quadratic objective is implemented. There is no group
sparsity term, alternative objective, ablation branch or experimental sweep.
"""
import numpy as np
import torch


def correction_graph(indices, weights):
    """Remove self edges, symmetrize and scale by maximum weighted degree."""
    from scipy.sparse import csr_matrix
    idx=np.asarray(indices,np.int64)
    weights=np.asarray(weights,np.float64)
    n,k=idx.shape
    rows=np.broadcast_to(np.arange(n)[:,None],(n,k))
    keep=(idx!=rows)&(weights!=0)
    # Preserve the native auto layout: small crop graphs use dense reductions.
    if n<1024:
        b=np.zeros((n,n),np.float64)
        b[rows[keep],idx[keep]]=weights[keep]
        symmetric=(b+b.T)*.5
        maximum=float(symmetric.sum(1).max())
        adjacency=symmetric/maximum if maximum else symmetric
        return adjacency,adjacency.sum(1)
    b=csr_matrix((weights[keep],(rows[keep],idx[keep])),shape=(n,n))
    symmetric=((b+b.T)*.5).tocsr()
    maximum=float(np.asarray(symmetric.sum(1)).max(initial=0.))
    adjacency=symmetric/maximum if maximum else symmetric
    degree=np.asarray(adjacency.sum(1)).ravel()
    return adjacency,degree


def grc(query_responses, query_to_class, indices, weights, strength=16., steps=5,
        device='cuda'):
    """Estimate the residual on a fixed graph; restore native query units.

    Graph preparation and query-coordinate conversion use float64. The CUDA
    deployment iterations use float32, as in the supplied native CUDA solver.
    ``device='cpu'`` is a float64 reference path, never a silent fallback.
    """
    y=np.asarray(query_responses)
    q=np.asarray(query_to_class,np.int64)
    if y.ndim!=2 or q.shape!=(y.shape[1],):
        raise ValueError('Require responses [N,Q] and category map [Q]')
    if strength<0 or int(steps)!=steps or steps<1 or not np.isfinite(y).all():
        raise ValueError('Invalid GRC strength/steps/responses')
    counts=np.bincount(q,minlength=int(q.max())+1)
    if (counts==0).any():
        raise ValueError('Category indices must be contiguous')
    scale=np.sqrt(1./counts[q])
    mean=(y.astype(np.float64)*(1./counts[q])[None]).sum(1)/len(counts)
    initial=(y.astype(np.float64)-mean[:,None])*scale[None]
    adjacency,degree=correction_graph(indices,weights)
    eta=1./(1.+2.*strength)
    sparse_layout=hasattr(adjacency,'nnz')
    if not (adjacency.nnz if sparse_layout else np.count_nonzero(adjacency)):
        return y.copy()
    if device=='cpu':
        delta=np.zeros_like(initial)
        for _ in range(int(steps)):
            x=initial+delta
            gradient=delta+strength*(degree[:,None]*x-adjacency@x)
            delta=delta-eta*gradient
    elif device=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GRC requested but CUDA is unavailable')
        x0=torch.as_tensor(initial,device=device,dtype=torch.float32)
        if sparse_layout:
            coo=adjacency.tocoo()
            graph=torch.sparse_coo_tensor(
                torch.as_tensor(np.stack([coo.row,coo.col]),device=device,dtype=torch.long),
                torch.as_tensor(coo.data,device=device,dtype=torch.float32),
                coo.shape,device=device).coalesce()
        else:
            graph=torch.as_tensor(adjacency,device=device,dtype=torch.float32)
        d=torch.as_tensor(degree,device=device,dtype=torch.float32)
        delta=torch.zeros_like(x0)
        for _ in range(int(steps)):
            x=x0+delta
            propagated=torch.sparse.mm(graph,x) if sparse_layout else graph@x
            gradient=delta+strength*(d[:,None]*x-propagated)
            delta=delta-eta*gradient
        delta=delta.cpu().numpy().astype(np.float64)
    else:
        raise ValueError('GRC device must be cuda or cpu')
    result=y.astype(np.float64)+delta/scale[None]
    unchanged=np.linalg.norm(delta,axis=1)==0
    result[unchanged]=y[unchanged]
    return result.astype(y.dtype)
