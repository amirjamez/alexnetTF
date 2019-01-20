# AlexNet Pretrained
There are a number of recent pretrained models available in [TensorFlow-Slim Research](https://github.com/tensorflow/models/tree/master/research/slim#fine-tuning-a-model-from-an-existing-checkpoint) for which users can download and finetune to other datasets, or, evaluate for classification tasks. However, there was no `AlexNet`in the list and this repo helps you reproduce that for `ImageNet` dataset. I also provide a pretrained model captured at 9 million iterations with `Top-5 accuracy of 79.85%` for those who doesn't want to train it from scracth. 

If you have optimized the training hyperparameters and managed to converge to a higher accuracy, please share your code here in the comment section bellow, so other can benefit from it as well.

## AlexNet Model Architecture
Here is the Conv and FC layers of AlexNet with their corresponding parameter and shape size:

```
#AlexNet Layer's reference                            #Shape                         #Params
LAYER=['alexnet_v2/conv1/weights:0',       # shape=(11, 11, 3, 64)                    23 232
'alexnet_v2/conv2/weights:0',              # shape=(5, 5, 64, 192)                   307 200
'alexnet_v2/conv3/weights:0',              # shape=(3, 3, 192, 384)                  663 552
'alexnet_v2/conv4/weights:0',              # shape=(3, 3, 384, 384)                1 327 104
'alexnet_v2/conv5/weights:0',              # shape=(3, 3, 384, 256)                  884 736
'alexnet_v2/fc6/weights:0',                # shape=(5, 5, 256, 4096)              26 214 400
'alexnet_v2/fc7/weights:0',                # shape=(1, 1, 4096, 4096)             16 777 216
'alexnet_v2/fc8/weights:0']                # shape=(1, 1, 4096, 1001)              4 100 096
                                                                         #Total=  50 297 536
```

The definition of the AlexNet_v2 is available at [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py).


## Training 
In order to do a `tf.slim` way of training alexnet_v2 with imagenet, you need to have a preprocessing.py file located in `models/slim/preprocessing`. There is also a `preprocessing_factory.py` file that maps the specific preprocessing file for a specific model and you need to add a line to tell TF to use the one you want for AlexNet. 

I have tried to implement them from scracth, but empirically found lower inference accuracy with respect to using inception_preprocessing. Thus, for this gist we stick to that for alexNet. In order to do that, just add a line where `preprocessing_fn_map` defines as: 

```
preprocessing_fn_map = {
      'alexnet_v2': inception_preprocessing,    # Needs to be added
      'cifarnet': cifarnet_preprocessing,
      'inception': inception_preprocessing,
      'inception_v1': inception_preprocessing,
      'inception_v2': inception_preprocessing,
      'inception_v3': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'lenet': lenet_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'resnet_v1_50': vgg_preprocessing,
      'resnet_v1_101': vgg_preprocessing,
      'resnet_v1_152': vgg_preprocessing,
      'resnet_v1_200': vgg_preprocessing,
      'resnet_v2_50': vgg_preprocessing,
      'resnet_v2_101': vgg_preprocessing,
      'resnet_v2_152': vgg_preprocessing,
      'resnet_v2_200': vgg_preprocessing,
      'vgg': vgg_preprocessing,
      'vgg_a': vgg_preprocessing,
      'vgg_16': vgg_preprocessing,
      'vgg_19': vgg_preprocessing,
  }
```

At this point you are good to strat training alexnet_v2 as per another tf.slim model, since you both have the architecture definition and the preprocessing:

```
$ python train_image_classifier.py     --train_dir=amir-alexnet-v2-results-oct2018/trained     --dataset_name=imagenet     --dataset_split_name=train     --dataset_dir=${DATASET_DIR}     --model_name=alexnet_v2
```

This uses mostly default hyperparameters as:

```
---batch_size=32
--learning_rate=0.01
--end_learning_rate=0.0001
--num_epochs_per_decay=2.0

```

## Inference Accuracy
You can easily evaluate the accuracy by executing this:

```
$ python eval_image_classifier.py     --alsologtostderr     --checkpoint_path=${CHECKPOINT_FILE}     --dataset_dir=${DATASET_DIR}     --dataset_name=imagenet     --dataset_split_name=validation     --model_name=alexnet_v2
```
, and here is the result:

```
2019-01-09 11:22:00.396964: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.56828]
2019-01-09 11:22:00.403138: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_4[0.7755]
2019-01-09 11:22:00.403139: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_3[0.742]
2019-01-09 11:22:00.403139: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.79854]
INFO:tensorflow:Finished evaluation at 2019-01-09-16:22:00
```

## AlexNet Pretraind Model

For those who wants to use the preatrained model, I have uploaded the model files as [AlexNet_Pretrained](https://drive.google.com/file/d/1ICnwX2fgyPMkJ0DyjOdLDadEO0C9C_ll/view?usp=sharing). The .zip file contains:

```
model.ckpt-9048119.data-00000-of-00001
model.ckpt-9048119.index
model.ckpt-9048119.meta
```

