# Slack
*Join slack and enter CRF-as-RNN channel to discuss*

https://deep-learning-geeks-slack.herokuapp.com/

# Intro and Clarifications

This repo is just to 

- [X] show you how to train [CRF as RNN](https://github.com/torrvision/crfasrnn) with PASCAL VOC datasets (20 classes + background)

- [X] be a well maintained place to communicate with this methods

- [X] try to rewrite CRF as RNN with Caffe2 (join the slack team and let's discuss together)


# Step by Step


## Pick Up a Machine

1. Single GPU
	
    AWS `p2.xlarge` instance (spot instance ~0.2$/hour, with Tesla K80 GPU, 12G Memory) will be enough for the training purpose.

    Equivalent setup may work.

2. Multipile GPUs
    
    You need to make some changes to achieve that and I haven't succeeded doing this for once (tried with 3GPUs with 18G memory intotal but failed) but I will update if I see any.


## Prepare Environment

Please refer to this [repo](https://github.com/KleinYuan/easy-yolo#b-environment-gpu) for all the details and commands. And it may take you around half -> one hour.

- [X] OpenCV (You may do not need OpenCV)

- [X] NVIDIA Driver

- [X] CUDA

- [X] CUDNN 

## Build CRF as RNN Caffe


1. Get CRF as RNN and navigate to the correct branch

	```
	>$ git clone https://github.com/torrvision/caffe.git
	>$ git checkout crfrnn
	```

	You will have a repo called `caffe` lying on whatever you run the command above.


2. Change some source code to optimize memory consumption (!IMPORTANT)

	If you just begin to build, you probably will meet this [issue](https://github.com/torrvision/crfasrnn/issues/79), which sucks.

	Therefore, you need to change some Caffe source code with this [PR](https://github.com/BVLC/caffe/pull/2016).

	Details:

	- [X] Change `caffe/src/caffe/layers/base_conv_layer.cpp`:

		 ```
		 line 12: Blob<Dtype> BaseConvolutionLayer<Dtype>::col_buffer_;
	     line 13: template <typename Dtype>
		 ```

	- [X] Change `caffe/include/caffe/layers/base_conv_layer.hpp`:

		 ```
		 line168: static Blob<Dtype> col_buffer_;
		 ```

	This can reduce the GPU memory consumption via sharing memory but with a known but ignorable [bug](https://github.com/BVLC/caffe/pull/2016#issuecomment-77509575).


3.  Configure Make

	In the root folder of `caffe`, there's a file called `Makefile.config.example`.

	Copy and Paste and then rename it to `Makefile.config` or just run `cp Makefile.config.example Makefile.config`

	- [X] If you install OpenCV separately, uncomment [`USE_PKG_CONFIG := 1`](https://github.com/torrvision/caffe/blob/crfrnn/Makefile.config.example#L107)

	- [X] If you want to train with Multiple GPU, you need to uncomment [`USE_NCCL := 1`](https://github.com/torrvision/caffe/blob/crfrnn/Makefile.config.example#L103) and install [NCCL](https://github.com/KleinYuan/train-crfasrnn#trial-on-train-with-multiple-gpu)

	- [X] If you want to use `OpenBlas` which works more efficiently with multiple CPUs instead of `ATLAS`, you probably wanna change [`BLAS := atlas`](https://github.com/torrvision/caffe/blob/crfrnn/Makefile.config.example#L50) to `BLAS := open`, which I don't think is necessary

	- [X] Comment out all the `60, 61` [arch options](https://github.com/torrvision/caffe/blob/crfrnn/Makefile.config.example#L42) since the machine you are using is probably not going to support them unless you have a machine can


	Then in root folder of `caffe` just run :

	```
	make all
	```

	and in case you fucked up, you can run 

	```
	make clean
	```
	to clean everything and re-make


	The process of make may take a while, like 10~min.


## Prepare DataSets

The whole idea is that, we need to 

- [X] Download PASCAL VOC dataset (which is very large)

- [X] Label them

- [X] Create LMDB for Caffe to easy access

So, I found this [repo](https://github.com/remz1337/train-CRF-RNN) is doing good job on this step.

Therefore, you need to:

1. Clone this [repo](https://github.com/remz1337/train-CRF-RNN)

2. Prepare

	Follow the step from [Prepare dataset for training](https://github.com/remz1337/train-CRF-RNN#prepare-dataset-for-training) to [Create LMDB database](https://github.com/remz1337/train-CRF-RNN#create-lmdb-database) and stop here. 

	You will have trouble on executing last [step](https://github.com/remz1337/train-CRF-RNN#create-lmdb-database) because this script needs a very small [functionality](https://github.com/remz1337/train-CRF-RNN/blob/master/data2lmdb.py#L14) to [dump](https://github.com/remz1337/train-CRF-RNN/blob/master/data2lmdb.py#L156) img into datum.

	Therefore, you have two options:

	a. In the root folder of this repo, also clone a Caffe (can be any version) and just build it like above except that you don't do any changes, basically:

	- [X] Clone Caffe
	- [X] run:
		```
		cp Makefile.config.example Makefile.config
		make all
		make pycaffe #!IMPORTANT and we didn't do that above because we didn't need this while you need this here
		``` 

	b. Go to the caffe we built above and continue build pycaffe and change the file path:

	- [X] run command: make pycaffe
	- [X] add caffe root like [this example](https://github.com/remz1337/train-CRF-RNN/blob/master/crfasrnn.py#L6) after this[line](https://github.com/remz1337/train-CRF-RNN/blob/master/data2lmdb.py#L6) with the actual caffe relative/absolute(recommended) path

	After you've done those two above, you should be able to finish all the labeling step and thereby, in the folder of `train-CRF-RNN`, you will be able to see those folders: 
		
	- [X] train_images_20_lmdb
	- [X] train_labels_20_lmdb
	- [X] test_images_20_lmdb
	- [X] test_labels_20_lmdb


## Prepare Training Proto buffer files

1. Clone this repo in a different place


	```
	git clone https://github.com/KleinYuan/train-crfasrnn.git

	```

2. Edit`trainKit/CRFRNN_train.prototxt`

	Replace `${PATH}` in line7/19/31/41 with actual *absolute* path of those folders I listed above.

3. Edit Make file of this repo

	Replace `${CAFFE_PATH}` with the root path of the caffe we built above and replace `${TRAIN_CRF_RNN_PATH}` with root path of this repo.


## Download Pre-trained Models

If you check the Makefile, you will see I offer you four choices:

- [X] Train-single-gpu-from-0

- [X] Train-multiple-gpus-from-0

- [X] Train-single-gpu-fine-tuning

- [X] Train-multiple-gpus-fine-tuning

If you wanna train a model from sratch, you need to download the FCN-8s model and put it in the folder of Makefile by just run:

```
wget http://dl.caffe.berkeleyvision.org/fcn-8s-pascal.caffemodel	
```

If you wanna train a model based on the pre-trained model, you need to download the `TVG_CRFRNN_COCO_VOC.caffemodel` (be aware of the LICENCE of this model, it's not free for commercial usage):

```
wget http://goo.gl/j7PrPZ -O TVG_CRFRNN_COCO_VOC.caffemodel
```


## Train!

Finally, you can train the model based on your purpose with the Makefile by running one of following command:

```
make Train-single-gpu-from-0
```

or 

```
make Train-multiple-gpus-from-0
```

or

```
Train-single-gpu-fine-tuning
```

or 

```
Train-multiple-gpus-fine-tuning
```

Also, if you wanan do multiple GPUs and have many GPUs, just keep adding `2, 3, 4...` on `0, 1` after the `-gpu` flag.

## Trial on Train with Multiple GPU

So, for Multiple GPUs training with Caffe, it's very picky for environment.

Those dependencies／changes are necessary to achieve this:

- [X] [NCCL](https://github.com/NVIDIA/nccl), which is subjected to CUDA version and be aware of the branch

- [X] Caffe Makefile uncomment [`USE_NCCL := 1`](https://github.com/torrvision/caffe/blob/crfrnn/Makefile.config.example#L103)


## Debug and Validation

Potential problems you will meet are:

1. `Check failed: error == cudaSuccess (2 vs. 0) out of memory`  ----> it means what it says, your memory is not enough

2. `error==cudaSuccess (77 vs. 0) an illegal memory access was encountered` -----> it means that either the shape is not correct or your cuda version is not correct, check [here](https://github.com/BVLC/caffe/issues/4169)

3. `iteration 0` stuck for a long time ----> it's normal, just chill and drink Coffee
