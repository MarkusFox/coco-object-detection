"""A set of functions that are used for custom dataset creation after using the 
    COCO Annotator by Justin Brooks (https://github.com/jsbroks/coco-annotator/).

These functions do not return a value, instead they create folders and copies of the input data.

"""

import json
import os
import random as rd
from shutil import copy


def create_dataset_from_annotations(json_file, output_dir):
    """Creates a dataset by copying only images with existing annotations.
    
    Args:
    - json_file: json data in form of COCO-Dataset format (http://cocodataset.org/#home).
    - output_dir: folder that will include the copies of all images and the json file.
    
    """
    # check json file existence
    assert (os.path.isfile(json_file)),"JSON file doesn't exist!"
    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir("%simages" % output_dir)
        print("Directory " , output_dir ,  " Created ")
        print("Directory " , "%simages" % output_dir ,  " Created ")
    else:
        print("Output folder already exists, please rename or choose a different name!")
        return

    with open(json_file) as json_f:
        data = json.load(json_f)
        for i in data['images']:
            src = i['path'][1:]
            dst = output_dir + 'images/' + i['file_name']
            copy(src,dst)
            
    copy(json_file, output_dir)


def create_dataset_split(json_file, output_dir, split=[0.7,0.15,0.15], supercategory=[False,""]):
    """Splits a dataset into up to three sets.
    
    Args:
    - json_file: json data in form of COCO-Dataset format (http://cocodataset.org/#home).
    - output_dir: folder that will include the three datasets and json files.
    - split: list of three float values, that define the split. First value is training, second validation
             and third test set. Must sum up to 1.
    - supercategory: list of [Boolean, String], that defines if all categories should be combined to
                     a single supercategory. Turned off by default.
                     
    
    """
    # check json file existence
    assert (os.path.isfile(json_file)),"JSON file doesn't exist!"
    # check if split is reasonable and sums to 1
    assert (sum(split) == 1),"Incorrect Split, please make [train,val,test] sum to 1"
    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir("%straining" % output_dir)
        os.mkdir("%svalidation" % output_dir)
        os.mkdir("%stest" % output_dir)
        print("Directory " , output_dir ,  " Created ")
        print("Directory " , "%straining" % output_dir ,  " Created ")
        print("Directory " , "%svalidation" % output_dir ,  " Created ")
        print("Directory " , "%stest" % output_dir ,  " Created ")
    else:
        print("Output folder already exists, please rename or choose a different name!")
        return
    
    # load annotation data from json
    with open(json_file) as json_data:
        data = json.load(json_data)
        # determine amount of images according to split
        num_images = len(data['images'])
        print("{} images found!".format(num_images))
        num_train = int(num_images * split[0])
        num_val = int(num_images * split[1])
        num_test = num_images - num_train - num_val
        print("{} Training, {} Validation, {} Test - Images".format(num_train, num_val, num_test))
        # create image split from shuffled data
        shuffled_imgs = rd.sample(data['images'], num_images)
        imgs_train = shuffled_imgs[:num_train]
        img_ids_train = [img['id'] for img in imgs_train]
        imgs_val = shuffled_imgs[num_train:num_train+num_val]
        img_ids_val = [img['id'] for img in imgs_val]
        imgs_test = shuffled_imgs[num_train+num_val:num_images]
        img_ids_test = [img['id'] for img in imgs_test]
        
        if supercategory[0]==True:
            single_label = supercategory[1]
            categ_all = [{"id": 1, "name": single_label, "supercategory": "", "color": "#df3ccd", "metadata": {}}]
            # update all annotations to one category
            for anno in data['annotations']:
                anno['category_id'] = 1
        else:
            # categories will stay the same for all splits
            categ_all = data['categories']
        
        anno_train = [anno for anno in data['annotations'] if anno['image_id'] in img_ids_train]
        anno_val = [anno for anno in data['annotations'] if anno['image_id'] in img_ids_val]
        anno_test = [anno for anno in data['annotations'] if anno['image_id'] in img_ids_test]

        # create json dicts
        train_dict = {
            #"info": {...},
            #"licenses": [...],
            "images": imgs_train,
            "annotations": anno_train,
            "categories": categ_all
            #"segment_info": [...] <-- Only in Panoptic annotations
        }
        val_dict = {
            "images": imgs_val,
            "annotations": anno_val,
            "categories": categ_all
        }
        test_dict = {
            "images": imgs_test,
            "annotations": anno_test,
            "categories": categ_all
        }
        
        # write json files
        with open('{}training.json'.format(output_dir), 'w') as fp:
            json.dump(train_dict, fp)
        with open('{}validation.json'.format(output_dir), 'w') as fp:
            json.dump(val_dict, fp)
        with open('{}test.json'.format(output_dir), 'w') as fp:
            json.dump(test_dict, fp)
        
        # copy images to respective folders
        for i in imgs_train:
            src = i['path'][1:]
            dst = output_dir + "training/" + i['file_name']
            copy(src,dst)
        for i in imgs_val:
            src = i['path'][1:]
            dst = output_dir + "validation/" + i['file_name']
            copy(src,dst)
        for i in imgs_test:
            src = i['path'][1:]
            dst = output_dir + "test/" + i['file_name']
            copy(src,dst)

