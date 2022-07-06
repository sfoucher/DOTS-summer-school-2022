import PIL
import pandas
import os
import torch
import albumentations
import numpy

from typing import *

from .csv import CSVDataset
from .register import DATASETS

from ..data.transforms import SampleToTensor

@DATASETS.register()
class ImageLevelDataset(CSVDataset):
    ''' Class to create an Image Level Dataset from a CSV file 
    
    This dataset is built on the basis of CSV files containing at least 'images' and
    'labels' columns. Any additional information (i.e. additional columns) will be 
    associated and returned by the dataset.
    '''

    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        albu_transforms: Optional[list] = None,
        end_transforms: Optional[list] = None
        ) -> None:
        ''' 
        Args:
            csv_file (str): absolute path to the csv file containing 
                annotations
            root_dir (str) : path to the images folder
            albu_transforms (list, optional): an albumentations' transformations 
                list that takes input sample as entry and returns a transformed 
                version. Defaults to None.
            end_transforms (list, optional): list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
        '''

        self.data = pandas.read_csv(csv_file)
        self.root_dir = root_dir
        self.albu_transforms = albu_transforms
        self.end_transforms = end_transforms

        self._store_end_params()

        self._img_names = self.data['images'].tolist()
        self._labels = self.data['labels'].tolist()
        self._others = self.data.to_dict('list')
        self._others.pop('images')
        self._others.pop('labels')
    
    def _load_target(self, index: int) -> Dict[str,List[Any]]:
        img_name = self._img_names[index]
        label = self._labels[index]

        target = {
            'image_id': [index], 
            'image_name': [img_name],
            'labels': [label]
            }
        
        for key, values in self._others.items():
            target.update({key: [values[index]]})
        
        return target
    
    def _transforms(
        self, 
        image: PIL.Image.Image, 
        target: dict
        ) -> Tuple[torch.Tensor, dict]:

        if self.albu_transforms:

            transform_pipeline = albumentations.Compose(self.albu_transforms)
            transformed = transform_pipeline(image = numpy.array(image))
            tr_image = numpy.asarray(transformed['image'])

            tr_image,  tr_target = SampleToTensor()(tr_image, target, '')

            if self.end_transforms is not None:
                for trans in self.end_transforms:
                    tr_image, tr_target = trans(tr_image, tr_target)

            return tr_image, tr_target