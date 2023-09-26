# shallow-U
An example implementation of U-Net with less depth for the semantic segmentation of images. Created for use with the [Oxford Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) but applicable to simple segmentation tasks.
## Model
The model file was trained for 32 epochs with batch size 16 using the Adam optimiser and a piecewise learning scheduler.

This version was trained on the daffodil subset of the dataset and not intended to generalise beyond images of the same type and perspective (see examples). Achieves 96% OA, 0.91 mIoU, and 0.93 weighted mIoU for this task.
