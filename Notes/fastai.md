# Super Resolution By fastai

[2018 Video Link](https://youtu.be/nG3tT31nPmQ)

[2019 Video Link](https://www.youtube.com/watch?v=9spwoDYwW_I&vl=en&ab_channel=JeremyHoward)

[Lecture Notes](https://github.com/hiromis/notes/blob/master/Lesson7.md)

[github notebooks](https://github.com/fastai/course-v3/tree/master/nbs/dl1)

More notes here:

- https://forums.fast.ai/t/lesson-7-official-resources/32553

VGG16:

- Long training time, memory consuming, lots of weights.

- Most models use 7x7 kernel for first conv layer

  =>stride 2=>wasted half pixels=>bad b.c. fine details matters

- Idea: attach vgg to a resnet. 



Residual (skip connection):

- keep details from previous layer

Dense Block(Net):

- Similar to Residual, instead of adding, it concatenates. The result get larger and larger but keeps info across many layers.
- Work well for segmentation

U-Net

- Similar to skip connection, as a U Shape. Skip connection concatenate the older layers to new layers
- 





How to increase resolution? Half stride.

<img src="fastai.assets/image-20201203012412075.png" alt="image-20201203012412075" style="zoom:33%;" />

Lots of redundent calculation.

What people do nowadays? Nearest Neighbor interpolation.

<img src="fastai.assets/image-20201203012525016.png" alt="image-20201203012525016" style="zoom:25%;" />

Do a stride 1 convolution after scaling up.



## Super Resolution

https://youtu.be/9spwoDYwW_I?t=2935

https://youtu.be/9spwoDYwW_I?t=3164 Crappify image

crappify: resize to low resolution img with linear interpolation, then draw a number to image and save as jpeg with different quality, lower quality get compressed more and have worse results.

```python
from fastai.vision import *
from PIL import Image, ImageDraw, ImageFont

class crappifier(object):
    def __init__(self, path_lr, path_hr):
        self.path_lr = path_lr
        self.path_hr = path_hr              
        
    def __call__(self, fn, i):       
        dest = self.path_lr/fn.relative_to(self.path_hr)    
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, 96, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
        w,h = img.size
        q = random.randint(10,70)
        ImageDraw.Draw(img).text((random.randint(0,w//2),random.randint(0,h//2)), str(q), fill=(255,255,255))
        img.save(dest, quality=q)
```

Add different artifacts, anything not in the images, the model wouldn't learn.

Code: https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-gan.ipynb

Use U-Net for the model.

U-Net Implementation: https://github.com/milesial/Pytorch-UNet

Use a pretrained model. 

```python
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
```

normalize imagenet_stats means use pretrained model. Why? To know how to remove noise/artifact, the model needs to know what should be there originally. Pretrained model knows these things because it has seen many data.

```python
def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)
```

Can be used as a watermark remover.

MSE loss function doesn't work very well. Why?

- Mean Squared Error between low res and high res image is actually quite small as the pixel location are the same. Missing detailed texture but color still preserves.

What can be a better loss function?

- GAN (Generative Adverserial Network)

  https://youtu.be/9spwoDYwW_I?t=3714

  Train a loss function. 

  <img src="fastai.assets/image-20201203015500778.png" alt="image-20201203015500778" style="zoom: 25%;" />

  Want to fool discriminator with prediction. As generator gets better within a few epoch, now train discriminator with generator. Let them train each other.

Need some pretrained models for both of them otherwise they are both blind and is hard to learn anything.



Clean Memory

```python
learn_gen=None
gc.collect
```

 GAN doesn't work well with momentums.





## Good Results

https://youtu.be/9spwoDYwW_I?t=5584

https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb









