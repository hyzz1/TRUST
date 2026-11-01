"""Save native predictions; GT is inspected only after model inference."""
import json
import os
import time
import uuid
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from mmengine.hooks import Hook


def save_json(path,payload):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05*(attempt+1))


class PredictionExport(Hook):
    def __init__(self,output,expected_images):
        self.output=Path(output)
        self.expected=int(expected_images)
        self.done=0
        self.cm=None
        self.identifiers=set()

    def before_test(self,runner):
        dataset=runner.test_dataloader.dataset
        if len(dataset)!=self.expected:
            raise ValueError(f'Expected full split {self.expected}, found {len(dataset)}')
        classes=dataset.metainfo['classes']
        self.cm=np.zeros((len(classes),len(classes)),np.int64)
        palette=np.asarray(dataset.metainfo['palette'],np.uint8)
        self.palette=np.zeros((256,3),np.uint8)
        self.palette[:len(palette)]=palette
        self.palette[255]=0
        for fmt in ['ids','color']:
            (self.output/'predictions'/fmt).mkdir(parents=True)
        (self.output/'palette.json').write_text(json.dumps(dict(classes=list(classes),palette=self.palette.tolist(),ignore=255)),encoding='utf-8')

    def after_test_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        sample=outputs[0]
        prediction=sample.pred_sem_seg.data.detach().cpu().numpy().squeeze().astype(np.uint8)
        identifier=Path(sample.metainfo['img_path']).stem
        if identifier in self.identifiers:
            raise ValueError('Duplicate image identifier')
        self.identifiers.add(identifier)
        # Inference has returned before any label is inspected here.
        ground_truth=sample.gt_sem_seg.data.detach().cpu().numpy().squeeze()
        valid=ground_truth!=255
        if prediction.shape!=ground_truth.shape:
            raise ValueError('Prediction must retain original image dimensions')
        if (prediction>=len(self.cm)).any():
            raise ValueError('Predicted class ID is outside the native category map')
        self.cm+=np.bincount(ground_truth[valid].astype(np.int64)*len(self.cm)+prediction[valid],
                            minlength=self.cm.size).reshape(self.cm.shape)
        Image.fromarray(prediction).save(self.output/'predictions/ids'/(identifier+'.png'))
        Image.fromarray(self.palette[prediction]).save(self.output/'predictions/color'/(identifier+'.png'))
        with (self.output/'predictions.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(index=self.done,id=identifier,shape=list(prediction.shape),
                prediction_pixel_sha256=hashlib.sha256(prediction.tobytes()).hexdigest()))+'\n')
        self.done+=1
        save_json(self.output/'progress.json',dict(images_done=self.done,total=self.expected))

    def finish(self,native_metrics):
        if self.done!=self.expected:
            raise ValueError('Full validation did not complete')
        diagonal=self.cm.diagonal().astype(np.float64)
        union=self.cm.sum(0)+self.cm.sum(1)-diagonal
        iou=np.divide(diagonal,union,out=np.full(len(diagonal),np.nan),where=union>0)*100
        score=float(np.nanmean(iou))
        if abs(score-float(native_metrics['mIoU']))>.011:
            raise ValueError('Independent confusion matrix and MMSeg score disagree')
        np.save(self.output/'confusion.npy',self.cm)
        for fmt in ['ids','color']:
            if len(list((self.output/'predictions'/fmt).glob('*.png')))!=self.expected:
                raise ValueError('Incomplete saved prediction count')
        save_json(self.output/'metrics.json',dict(complete=True,images=self.done,
            mIoU_percent=score,per_class_IoU_percent=iou.tolist(),native_metrics=native_metrics))
