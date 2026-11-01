# Dataset roots

Pass the root of the **prepared native dataset**, not a new raw-label format. No label preparation is silently performed. Native MMSeg datasets perform their declared zero-label reduction. Class 0 in the resulting evaluation label space remains valid; 255 is ignored.

| CLI dataset | Images | Image subdirectory | Label subdirectory | Split file |
|---|---:|---|---|---|
| voc20 | 1449 | `JPEGImages` | `SegmentationClass` | `ImageSets/Segmentation/val.txt` |
| context59 | 5105 | `JPEGImages` | `SegmentationClassContext` | `ImageSets/SegmentationContext/val.txt` |
| coco_stuff164k | 5000 | `val2017` | `annotations/val2017` | `` |
| cityscapes | 500 | `leftImg8bit/val` | `gtFine/val` | `` |
| ade20k | 2000 | `images/validation` | `annotations/validation` | `` |
| voc21 | 1449 | `JPEGImages` | `SegmentationClass` | `ImageSets/Segmentation/val.txt` |
| context60 | 5105 | `JPEGImages` | `SegmentationClassContext` | `ImageSets/SegmentationContext/val.txt` |
| coco_object | 5000 | `val2017` | `annotations/val2017` | `` |

The class/alias lists are in `configs/cls_*.txt`. Preserve them exactly. COCO Object uses preconverted `_instanceTrainIds.png` masks; COCO Stuff uses its native train-ID mapping. Cityscapes uses its native train-ID labels.

Example (Windows CMD): `python run.py voc21 --data-root E:/DLearn/Dataset/VOCdevkit/VOC2012`. The same command shape applies on Linux.
