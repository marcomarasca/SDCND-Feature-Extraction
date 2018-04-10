# AlexNet Feature Extraction
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
We use feature extraction taking a pretrained neural network and replacing the final (classification) layer with a new classification layer. During training the weights in all the pre-trained layers are frozen, so only the weights for the new layer(s) are trained. In other words, the gradient doesn't flow backwards past the first new layer.

We use [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) as base network and in particular a Tensorflow implementation adapted from [Michael Guerhoy and Davi Frossard](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) (See [alexnet.py](./alexnet.py)).

AlexNet was originally trained on the [ImageNet](http://www.image-net.org/) database, we extract AlexNet's features and use them to classigy images from the [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The weights used in this project are provided by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu/).

## Getting Started

The project requires various libraries to be installed first, in particular:

* Python 3
* TensorFlow
* NumPy
* SciPy
* matplotlib

You'll also need the training data and the AlexNet weights (make sure the downloaded files are in the same directory as the code):

* [Training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p)
* [AlexNet weights](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy)

## Imagenet Inference

We can run the [imagenet_inference.py](./imagenet_inference.py) that runs using the original imagenet weights on the following images:

![Poodle](./poodle.png)
![Weasel](./weasel.png)

```bash
python imagenet_inference.py
```
The output should look something like:

```bash
Image 0
miniature poodle: 0.389
toy poodle: 0.223
Bedlington terrier: 0.173
standard poodle: 0.150
komondor: 0.026

Image 1
weasel: 0.331
polecat, fitch, foulmart, foumart, Mustela putorius: 0.280
black-footed ferret, ferret, Mustela nigripes: 0.210
mink: 0.081
Arctic fox, white fox, Alopex lagopus: 0.027

Time: 0.890 seconds
```

## Traffic Sign Inference

Now we can adapt the network to run on the [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The original AlexNet network expects 227x227x3 pixel image, whereas the traffic sign images are 32x32x3, we therefore resize the images.

![Construction](./construction.jpg)
![Stop](./stop.jpg)

```bash
python traffic_sign_inference.py
```

The output should look somehing like:

```bash
Image 0
screen, CRT screen: 0.051
digital clock: 0.041
laptop, laptop computer: 0.030
balance beam, beam: 0.027
parallel bars, bars: 0.023

Image 1
digital watch: 0.395
digital clock: 0.275
bottlecap: 0.115
stopwatch, stop watch: 0.104
combination lock: 0.086
```

Naturally the original dataset was trained on the ImageNet database, which has 1000 classes of image while the german traffic sign dataset only has 43 classes. Next we apply feature extraction so that we can use the 43 classes from the german traffic sign dataset.

## Feature Extraction

In order to successfully classify our traffic sign images, we need to remove the final, 1000-neuron classification layer and replace it with a new, 43-neuron classification layer.

This is called feature extraction, because we're basically extracting the image features inferred by the penultimate layer, and passing these features to a new classification layer.

In the [feature_extraction.py(./feature_extraction.py)] we simply replace the last layer of AlexNet with a dense layer outputting 43 classes:

```python
# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)

# Replace the last layer of AlexNet (e.g. from 1000 to 43)

shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

# Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs.

logits = logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)
```

Running 
```bash
python feature_extraction.py
```

Should output something like:

```
Image 0
Traffic signals: 0.045
End of all speed and passing limits: 0.044
Speed limit (50km/h): 0.040
Go straight or right: 0.035
Turn left ahead: 0.035

Image 1
Wild animals crossing: 0.069
Traffic signals: 0.068
Speed limit (60km/h): 0.058
End of speed limit (80km/h): 0.054
End of no passing by vechiles over 3.5 metric tons: 0.052

Time: 0.710 seconds
```

## Training the extracted features

The model kinda of works but it's yet not trained as we simply replace the last classification layer to adapt to the german traffic sign dataset, we can run now the [train_feature_extraction.py](./train_feature_extraction.py) in order to train the last layer:

```bash
python train_feature_extraction.py
```

The process will take a while and a GPU is strongly suggested. At the end of the process the model will be save in the model.ckpt file.

If we run now the feature_extraction.py again (will reuse the save moedl) we should better results :)




