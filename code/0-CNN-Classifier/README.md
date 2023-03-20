## The classifier

We will be implementing a classifier based on EfficientNet, and we'll be adapting it as we proceed with the project.
The first order of businesses is to get the model up and working on generic images.
Then we will fine-tune it on the images of our TBD dataset.

We did some tests with the pretrained efficientnet model provided by pytorch, and it seems to work well.
That's why we decided to implement our own EfficientNet model.

### Sources

#### https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-1-an-overview-1830935e0c8b
This 4-part series on how to implement EfficientNet in pytorch was a great source for us.
It explains the architecture of the model, and how to implement it very nicely.

#### https://arxiv.org/abs/1905.11946
The original paper on EfficientNet. 
A good read to give us the general idea of the model since we aren't deeply familiar with ML theory.

