# Lyft Perception Challenge

The goal of this competition is pixel-wise identification of road and vehicles in an image

I used the same technique from Term 3 in my Self-Driving Car course to build the model. 

## Architecture
Every pixel will be classified as:
1) Vehicle
2) Road
3) Not vehicle nor road

The model is configured as follows:
1) Load the fcn vgg model and get layers 3, 4, and 7
2) Perform convolution on layer 7. Name it `fcn8`

| Parameters         | Value    |
| ------------------ |:--------:|
| name               | fcn8     |
| filters            | 3        |
| kernel_size        | 1        |
| kernel_regularizer | 1e-3     |

3) Perform convolutional tranpose on `fcn8`. Name it `fcn9`

| Parameters         | Value    |
| ------------------ |:--------:|
| name               | fcn9     |
| filters            | same shape as vgg layer4 |
| strides | 2, 2 |
| padding | same |
| kernel_size        | 4        |
| kernel_regularizer | 1e-3     |

4) Add `fcn9` and layer 4. Name it `fcn9_plus_vgg_layer4`
5) Perform convolutional tranpose on `fcn9_plus_vgg_layer4`. Name it `fcn10`

| Parameters         | Value    |
| ------------------ |:--------:|
| name               | fcn10     |
| filters            | same shape as vgg layer3 |
| strides | 2, 2 |
| padding | same |
| kernel_size        | 4        |
| kernel_regularizer | 1e-3     |

6) Add `fcn10` and layer 3. Name it `fcn10_plus_vgg_layer3`
7) Finally, perform convolutional tranpose on `fcn10_plus_vgg_layer3`. Name it `fcn11`

| Parameters         | Value    |
| ------------------ |:--------:|
| name               | fcn11     |
| filters            | 3 |
| strides | 8, 8 |
| padding | same |
| kernel_size        | 16        |
| kernel_regularizer | 1e-3     |

8) Reshape `fcn11` into (-1, 3). Name it `fcn_logits`
9) Use `Adam optimizer` for training

## Training
1) [Download](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) a VGG model that has been customized for FCN (Fully Convolutional Networks)
2) Crop both RGB and Seg image from a `800x600` to `800x320`
3) Filter the Seg image for vehicle, road, and lane lines. (The lane lines and road is merged)
4) Resize both RGB and Seg image to `480x192`
5) Invert both road and vehicle to get not vehicle nor road image
6) Flatten, vertical stack, invert, and reshape to `480x192x3`
7) Finally, train with `drop rate: 0.3` and `learning rate: 0.0001`

## Interfacing
1) Load trained model
2) Crop image to `800x320`
3) Resize cropped image to `480x192`
4) Pass image into the model
5) Probabilities greater then `0.1` is accepted (set to 1) for vehicle and `0.96` for road. Everything else is rejected (set to 0)
6) Since the image is only 0 and 1, we can consider this a black and white image. Resize the black and white image back to `800x320`
7) Pad the image back to its original size `800x600`