# shallow-U
An example implementation of [U-Net](https://arxiv.org/abs/1505.04597) with less depth for semantic segmentation. The training loop is intended for use with the [Oxford Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) but is relatively easy to adapt to other simple segmentation targets.
## Model
The model file (zipped) was trained for 32 epochs with batch size 16, an Adam optimiser, and a piecewise learning scheduler.

This version was trained on the daffodil subset of the dataset and is not intended to generalise beyond images of the same type and perspective (see examples). Achieves 96% OA, 0.91 mIoU, and 0.93 weighted mIoU for this task.
