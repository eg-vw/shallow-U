# shallow-U
An example implementation of U-Net with less depth for the semantic segmentation of images. The training loop is intended for use with the [Oxford Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) but can be adapted to other simple segmentation tasks.
## Model
The model file (zipped) was trained for 32 epochs with batch size 16 using the Adam optimiser and a piecewise learning scheduler.

This version was trained on the daffodil subset of the dataset and is not intended to generalise beyond images of the same type and perspective (see examples). Achieves 96% OA, 0.91 mIoU, and 0.93 weighted mIoU for this task.
