# OpenFace Demo

### 1. Create raw image directory.
Create a directory for your raw images so that images from different people are in different subdirectories. The names of the labels or images do not matter, and each person can have a different amount of images. The images should be formatted as jpg or png and have a lowercase extension.

```sh
$ tree data/mydataset/raw
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png
```

### 2. Preprocess the raw images
We align the faces in the raw images that we prepared. Change 8 to however many separate processes you want to run: 

```sh
for N in {1..8}; do python ./demos/align-dlib.py <path-to-raw-dataset> align outerEyesAndNose <path-to-aligned-images> --size 96 --toGrayScale & done
```

Arguments:

- size: define the image size
- toGrayScale: convert the image to grayscale before alignment process.

If failed alignment attempts causes your directory to have too few images, you can use the utility script _openface/util/prune-dataset.py_ to deletes directories with less than a specified number of images.

```sh
python openface/util/prune-dataset.py <path-to-aligned-images> --numImagesThreshold 1
```

You can specify the minimum number of images in each class to be kept by using the argument _--numImagesThreshold_.

### 3. Generate Representations
In order to embed the aligned faces, we need to call the following command:

```sh
./openface/batch-represent/main.lua -outDir <path-to-output-features> -data <path-to-aligned-images>
```

The extracted features will be generated in the _<path-to-output-features>_.

### 4.Create the Classification Model
We perform training a classification model using the following command:
```sh
python ./demos/train.py train --classifier KNN <path-to-output-features>
```
We can choose among different classifiers such as KNN, LinearSVM, Softmax, ect. However, we recommend to use a KNN classifier with K=1.

### 5. Perform face recognition on Video.

We are now able to recognize people from video using the trained classifier from the previous step. The command that helps us to do this is:
```sh
python ./demos/recognize_online.py --classifierModel <path-to-features-directory> --multi --videoDir <path-to-video> --saveFaces --saveAllFrames --recognizeFace <person-name>
```
We can perform different tasks by specifying the following arguments:

- imgDim: image dimension used in the DNN model and in the alignment process. Default: 96
- videoDir: path to input video.
- outDir: path to output folder. Default: "./output/"
- classifierModel: path to the trained classifier (the generated .pkl file) from the previous step.
- saveFaces: save all detected faces in the video with their corresponding prediction, distance and frame number.
- saveAllFrames: save all frames with recognized faces on them.
- combineVideo: combine all the frames with recognized faces on them into a single video.
- multi: Detect multiple faces in each frame. With this turned off, the script only recognizes the largest face in each frame.
- threshold: Save only the frames that are predicted with confidence higher than this threshold. Type: float. Default: -1.0, this means frames are not saved.
- resizeVideoRatio: Resize the input video before processing by a ratio. Type: float. Default: 1.0
- recognizeFace: Specify the person needs to be recognized in the video. The name of the person must be same as the name specified in training set. Type: string. Default value: empty string "". If not specified, all the faces in the video will be recognized.

### 6. For Adding new persons (or images of existing persons) into an existing embedding:

First, setup the new person images as follow. The name of the person that is added more images has to be the same as the previously added.

```sh
$ tree data/new_person
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png
```

Then execute the following command:

```sh
python ./demos/add_new_person.py --inputDir data/new_person/ --featureDir <path-to-features-directory>
```

- featureDir: input the path to the feature directory that the new person should be added into.
- classifierModel: define the classifier to be re-trained. Type: string. Default: 'KNN'.

** Note: Please empty the data/new_person directory before adding other persons. **

