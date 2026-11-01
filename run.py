"""Evaluate the complete TRUST model on one full validation set."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['voc20','voc21','context59','context60','coco_object','coco_stuff164k','cityscapes','ade20k'])
    parser.add_argument('--data-root',default=os.environ.get('DATA_ROOT',''))
    parser.add_argument('--clip-weights',default=os.environ.get('CLIP_WEIGHTS','weights/ViT-B-16.pt'))
    parser.add_argument('--dino-repo',default=os.environ.get('DINO_REPO','third_party/dinov3'))
    parser.add_argument('--dino-weights',default=os.environ.get('DINO_WEIGHTS','weights/dinov3_vitb16.pth'))
    parser.add_argument('--output')
    args=parser.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    runtime=ROOT/'outputs/.runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    for key in ['TEMP','TMP','MPLCONFIGDIR','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH']:
        os.environ[key]=str(runtime)
    os.chdir(ROOT)
    weights=Path(args.clip_weights).expanduser().resolve()
    dino_weights=Path(args.dino_weights).expanduser().resolve()
    dino_repo=Path(args.dino_repo).expanduser().resolve()
    if not weights.is_file() or not dino_weights.is_file():
        parser.error('Provide the local CLIP and official DINO checkpoints; no weight download is performed.')
    if not (dino_repo/'hubconf.py').is_file():
        parser.error('DINO_REPO must point to an official DINO checkout containing hubconf.py.')
    if not args.data_root or not Path(args.data_root).is_dir():
        parser.error('Set --data-root or DATA_ROOT to the prepared dataset directory.')
    output=Path(args.output) if args.output else ROOT/'outputs'/(args.dataset+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        parser.error('Output already exists; choose a new directory.')
    output=output.resolve()
    output.mkdir(parents=True)
    import torch
    if not torch.cuda.is_available():
        parser.error('The complete model requires CUDA; no scientific precision/device fallback is used.')
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    import custom_datasets
    import trust.model
    from trust.export import PredictionExport
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    init_default_scope('mmseg')
    cfg=Config.fromfile(str(ROOT/'configs'/(args.dataset+'.py')))
    cfg.model.update(clip_path=str(weights),dino_repo=str(dino_repo),dino_weights=str(dino_weights))
    cfg.model.name_path=str((ROOT/cfg.model.name_path).resolve())
    cfg.test_dataloader.dataset.data_root=str(Path(args.data_root).resolve())
    cfg.work_dir=str(output)
    cfg.launcher='none'
    cfg.dump(str(output/'config.py'))
    runner=Runner.from_cfg(cfg)
    exporter=PredictionExport(output,cfg.expected_images)
    runner.register_hook(exporter,priority='LOW')
    metrics=runner.test()
    exporter.finish(metrics)
    print(json.dumps(dict(output=str(output),mIoU=metrics.get('mIoU')),indent=2))


if __name__=='__main__':main()
