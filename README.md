# Animaloc

## Installation
Install the dependencies
```console
pip install -r requirements.txt
```

Install the code
```console
python setup.py install
```

Create a [Weights & Biases](https://wandb.ai/home) account and then log in
```console
wandb login
```

## Usage
### Starting a training session
A training session can easily be launched using the `train.py` tool. This tool uses [Hydra](https://hydra.cc/) framework. You simply need to modify the basic config file and then run:
```console
python tools/train.py
```

You can also create your own config file. Save it first into the `configs/train` folder and then run:
```console
python tools/train.py train=<your config name>
```
You can also make multiple different configurations runs or modify some parameters directly from the command line (see the [doc](https://hydra.cc/docs/intro)).

### Starting a testing session
A testing session can easily be launched using the `test.py` tool. This tool uses [Hydra](https://hydra.cc/) framework again. You simply need to modify the basic config file and then run:
```console
python tools/test.py
```


You can also create your own config file. Save it first into the `configs/test` folder and then run:
```console
python tools/test.py test=<your config name>
```

### Visualize ground truth (and detections)
You can view your ground truth and your model's detections by using the `view.py` tool. This tool uses [FiftyOne](https://voxel51.com/fiftyone/). You simply need to specify a root directory that contains your images (`root`), your CSV file containing the ground truth (`gt`) and optionaly a CSV file containing model's detections (`-dets`). See dataset format below for your CSV files format.
```console
python tools/view.py root gt [-dets]
```

## Dataset format
The `csv_file` and `root_dir` datasets parameters of the config file should point to a CSV file, containing your annotations, and the folder containing your images respectively.

The CSV file must contain the header **`images,x,y,labels`** (points) or **`images,x_min,y_min,x_max,y_max,y,labels`** (bounding boxes) and each row should represent one annotation, with at least, the image name (**images**), the object location within the image (**x**, **y**) for points, and (**x_min**, **y_min**, **x_max**, **y_max**) for bounding boxes and its label (**labels**):

Points dataset:
```csv
images,x,y,labels
Example.JPG,517,1653,2
Example.JPG,800,1253,1
Example.JPG,78,33,3
Example_2.JPG,896,742,1
...
```

Bounding boxes dataset:
```csv
images,x_min,y_min,x_max,y_max,labels
Example.JPG,530,1458,585,1750,4
Example.JPG,95,1321,152,1403,2
Example.JPG,895,478,992,658,1
Example_2.JPG,47,253,65,369,1
...
```

An image containing *n* objects is therefore spread over *n* lines.

Note that the names of the images in the CSV file must obviously match those in the folder pointed to by the `root-dir` parameter.