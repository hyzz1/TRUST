default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn',opencv_num_threads=0),
               dist_cfg=dict(backend='gloo'))
test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='IoUMetric',iou_metrics=['mIoU'])
default_hooks = dict(timer=dict(type='IterTimerHook'),
                     logger=dict(type='LoggerHook',interval=25,log_metric_by_epoch=False))
log_processor = dict(by_epoch=False)
log_level = 'INFO'
randomness = dict(seed=20260911,deterministic=False)
load_from = None
resume = False
