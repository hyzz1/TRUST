"""Complete TRUST: native host -> RGSS aggregation -> GRC -> native decode."""
import numpy as np
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from .dino import FrozenDINO
from .graph import GraphConfig, SparseRelation
from .rgss import collapse_queries, source_scores, select_sources
from .grc import grc

from base_model import NACLIP as NativeHost
HOST='naclip'


@MODELS.register_module()
class TRUST(NativeHost):
    """Both proposed modules are always enabled; no component switches."""
    def __init__(self,dino_repo,dino_weights,graph_topk=64,keep_fraction=.5,
                 graph_temperature=.07,grc_lambda=16.,grc_steps=5,
                 dino_family='dinov3',dino_model='dinov3_vitb16',
                 source_seed=20260911,**kwargs):
        super().__init__(**kwargs)
        if not 0<keep_fraction<=1:
            raise ValueError('Retention fraction must be in (0,1]')
        self.keep_fraction=float(keep_fraction)
        self.source_seed=int(source_seed)
        self.image_index=0
        self.crop_index=0
        self.grc_lambda=float(grc_lambda)
        self.grc_steps=int(grc_steps)
        self.graph_config=GraphConfig(graph_topk=graph_topk,graph_temperature=graph_temperature)
        device=self.query_features.device
        self.dino=FrozenDINO(dino_repo,dino_weights,family=dino_family,
                             model_name=dino_model,precision='fp32',device=device)
        self.register_buffer('pixel_mean',torch.tensor([122.771,116.746,104.094],device=device).view(1,3,1,1),persistent=False)
        self.register_buffer('pixel_std',torch.tensor([68.501,66.632,70.323],device=device).view(1,3,1,1),persistent=False)
        self.net.requires_grad_(False).eval()

    @torch.no_grad()
    def forward_feature(self,img,logit_size=None,**kwargs):
        if isinstance(img,list):
            img=img[0]
        if img.shape[0]!=1:
            raise ValueError('TRUST evaluates one image/crop per batch')
        patch=int(self.net.visual.patch_size)
        h,w=img.shape[-2]//patch,img.shape[-1]//patch
        features=self.net.encode_image(img,return_all=True)
        if HOST!='scclip':
            features=features[:,1:]
        features=F.normalize(features,dim=-1)
        z=features[0]
        if len(z)!=h*w:
            raise RuntimeError('Native CLIP tokens do not match the patch grid')
        rgb=(img.float()*self.pixel_std+self.pixel_mean)/255.
        if HOST=='pearl':
            rgb=rgb.clamp(0,1)
        else:
            rgb=rgb[:,:,:h*patch,:w*patch]
        visual,_=self.dino(rgb,(h,w))
        relation=SparseRelation.from_features(visual,self.graph_config)
        idx=relation.indices[0]
        original=relation.weights[0]
        if HOST=='pearl':
            raw=z.float()@self.query_features.float().T
        else:
            raw=z@self.query_features.T
        classes=int(self.query_idx.max())+1
        scores=collapse_queries(raw.cpu().numpy(),self.query_idx.cpu().numpy(),classes)
        reliability=source_scores(scores,idx.cpu().numpy(),original.cpu().numpy())['margin_agreement']
        seed=self.source_seed+self.image_index*17+self.crop_index*1000003
        selected,_,_=select_sources(idx.cpu().numpy(),original.cpu().numpy(),
                                    reliability,self.keep_fraction,seed)
        self.crop_index+=1
        selected_tensor=torch.from_numpy(selected).to(z.device)
        unit=F.normalize(z.float(),dim=-1,eps=1e-8)
        aggregate=torch.empty_like(unit)
        for start in range(0,len(z),128):
            edges=idx[start:start+128]
            aggregate[start:start+128]=(unit[edges]*selected_tensor[start:start+128,:,None]).sum(1)
        aggregate=F.normalize(aggregate,dim=-1,eps=1e-8)
        if HOST=='pearl':
            text=F.normalize(self.query_features,dim=-1).to(aggregate.dtype)
        else:
            aggregate=aggregate.to(z.dtype)
            text=self.query_features
        response=aggregate@text.T
        numpy_response=response.float().cpu().numpy() if HOST!='naclip' else response.cpu().numpy()
        corrected=grc(numpy_response,self.query_idx.cpu().numpy(),idx.cpu().numpy(),selected,
                      self.grc_lambda,self.grc_steps,device=img.device.type)
        out_dtype=torch.float32 if HOST=='scclip' else response.dtype
        logits=torch.as_tensor(corrected,device=img.device,dtype=out_dtype).T.reshape(1,self.num_queries,h,w)
        size=logit_size if logit_size is not None else img.shape[-2:]
        return F.interpolate(logits,size=size,mode='bilinear',align_corners=self.align_corners)

    def predict(self,inputs,data_samples):
        self.crop_index=0
        result=super().predict(inputs,data_samples)
        self.image_index+=1
        return result
