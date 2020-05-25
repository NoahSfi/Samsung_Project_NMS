#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:43:21 2020

@author: noahsarfati
"""

###############################################################################

""" Please download in the cocoapi folder `2017 val images` 
and `2017 val/train annotations` under the names `val2017` and `annotations`"""

###############################################################################

# import the necessary packages


# patch tf1 into `utils.ops`
import glob
import numpy as np
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.core import post_processing
from tqdm import tqdm
from PIL import Image, ImageDraw
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
import copy
import os
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


###############################################################################

# Load object detection model


def loadModel(modelPath):
    """Load associate tf OD model"""
    model_dir = modelPath + "/saved_model"
    detection_model = tf.saved_model.load(str(model_dir))
    detection_model = detection_model.signatures['serving_default']
    return detection_model


def loadCocoApi(dataDir="cocoapi", dataType="val2017"):
    """Read annotations file and return the associate cocoApi"""
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    # initialize COCO api for instance annotations
    coco = COCO(annFile)
    return coco


def getCategories(coco):
    """Return list of categories one can study"""
    cats = coco.loadCats(coco.getCatIds())
    categories = [cat['name'] for cat in cats]
    return categories


def getImgClass(catStudied, coco):
    # get all images containing given categories, and the Index of categories studied
    """
    input:
        catStudied: name among the category list
        coco: coco class describing the image to study
        nbImageStudied: the number of image one want to work on, if it is above
        than the number of image in the given category, then it choses total number of images.
    output: 
        list img with dictionnary element of the form: 
        {'license': 5, 'file_name': '000000100274.jpg', 
        'coco_url': 'http://images.cocodataset.org/val2017/000000100274.jpg', 'height': 427, 
        'width': 640, 'date_captured': '2013-11-17 08:29:54', 
        'flickr_url': 'http://farm8.staticflickr.com/7460/9267226612_d9df1e1d14_z.jpg', 'id': 100274}

        catIds: List of index of categories stydied
    """
    catIds = coco.getCatIds(catNms=[catStudied])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds)
    return img, catIds[0]


def expand_image_to_4d(image):
    """image a numpy array representing a gray scale image"""
    # The function supports only grayscale images
    assert len(image.shape) == 2, "Not a grayscale input image"
    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3dims = np.expand_dims(image, last_axis)
    training_image = np.repeat(
        grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
    assert len(training_image.shape) == 3
    assert training_image.shape[-1] == 3
    return training_image
###############################################################################


def run_inference_for_single_image(model, image):
    """
    input:
        model : OD model
        image : np.array format
    output:
        Dictionnary:
            key = ['num_detections','detection_classes','detection_boxes','detection_scores']
            valuesType = [int,list of int, list of 4 integers, list int]
    """
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    # If image doesn't respect the right format ignore it
    try:
        output_dict = model(input_tensor)
    except:
        return None

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}

    key_of_interest = ['detection_scores',
                       'detection_classes', 'detection_boxes']
    output_dict = {key: list(output_dict[key]) for key in key_of_interest}
    output_dict["num_detections"] = int(num_detections)
    output_dict['detection_boxes'] = [[float(box) for box in output_dict['detection_boxes'][i]] for i in range(
        len(output_dict['detection_scores']))]
    output_dict['detection_scores'] = [
        float(box) for box in output_dict['detection_scores']]
    output_dict['detection_classes'] = [
        float(box) for box in output_dict['detection_classes']]

    if len(output_dict["detection_boxes"]) == 0:
        return None
    return output_dict


def computeInferenceBbox(model):
    """
    For all the images in the coco img, compute the output_dict with
    run_inference_for_single_image. Store them as a dictionnary with keys
    being the index of the image in our coco dataset.

    input:
    - model: OD model
    - img: coco class describing the images to study

    output:
    A dictionnary describing the inferences for each image id associated to 
    a dictionary:
    {id: keyDic = ['num_detections','detection_classes','detection_boxes',
                   'detection_scores']}
    """
    all_output_dict = dict()
    i = 0
    for image_path in tqdm(glob.glob("cocoapi/val2017/*.jpg")):
        # the array based representation of the image
        image = Image.open(image_path)
        image_np = np.array(image)
        """If image is gray_scale one need to reshape to dimension 4
        using the utility function defined above"""
        if len(image_np.shape) == 2:
            image_np = expand_image_to_4d(image_np)
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        idx = image_path.split("/")[-1]
        all_output_dict[idx] = output_dict
        i += 1
    return all_output_dict
###############################################################################


def computeNMS(output_dict, iou_threshold=0.6):
    """
    input:
    -output_dict: the dictionnary ouput of the inference computation
    keyDic = ['num_detections','detection_classes','detection_boxes','detection_scores']
    - iou_threshold: it is used for the nms that will be applied after the OD

    output:
    A 3D tuple in this order:
    - final_classes : list of int64 telling the category of each detected bbox
    - final_scores : list of float64 scoring each bbox
    - final_boxes : list of coordinates of each bbox (format : [ymin,xmin,ymax,xmax] )
    """

    if not output_dict:
        return None, None, None
    #box_selection = post_processing.multiclass_non_max_suppression([output_dict['detection_boxes']], output_dict['detection_scores'], score_thresh=.8, iou_thresh=.5, max_size_per_class=0)

    # Apply the nms

    box_selection = tf.image.non_max_suppression_with_scores(
        output_dict['detection_boxes'], output_dict['detection_scores'], 100,
        iou_threshold=iou_threshold, score_threshold=float('-inf'),
        soft_nms_sigma=0.0, name=None)

    # Index in the list output_dict['detection_boxes']
    final_boxes = list(box_selection[0].numpy())
    # Index in the list output_dict['detection_scores']
    final_scores = list(box_selection[1].numpy())
    final_classes = []
    for i in range(len(final_boxes)):
        index = final_boxes[i]
        # We want the actual bbox coordinate not the index
        final_boxes[i] = output_dict['detection_boxes'][index]
        final_classes.append(output_dict['detection_classes'][index])

    return final_classes, final_scores, final_boxes

###############################################################################


def putCOCOformat(boxes, im_width, im_height):
    """
    Transform a bbox in the OD format into cocoformat
    input:
        boxes: List of the form [ymin,xmin,ymax,xmax] in the percentage of the image scale
        im_width: real width of the associated image
        im_height: real height of the associated image
    output:
        List of the form [left,top,width,height] describing the bbox, in the image scale
    """
    # float to respect json format
    left = float(boxes[1]) * im_width
    right = float(boxes[3]) * im_width
    top = float(boxes[0]) * im_height
    bottom = float(boxes[2]) * im_height
    width = right - left
    height = bottom - top

    return [left, top, width, height]


def writeResJson(img, resFilePath, all_output_dict, catIds, iou_threshold,newFile = True):
    """
    Write a Json file in the coco annotations format for bbox detections

    input:
        img: coco class describing the images to study
        resFilePath: path of the file where the result annotations will be written
        all_output_dict: dict with key: image Id in the coco dataset and value the output_dict
        computed with the OD.
        iou_threshold: Paramaeter for the nms algorithm that will be applied to the OD
    output:
        Json file of the form:
        [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}...]
        The length of this list is lengthData

        imgIds: List of the image ids that are studied
    """
    result = []
    imgIds = set()  # set to avoid repetition
    key_of_interest = ['detection_scores',
                       'detection_classes', 'detection_boxes']
    for i in range(len(img)):
        imgId = img[i]["id"]
        imgIds.add(imgId)
        
        output_dict = copy.deepcopy(all_output_dict[img[i]['file_name']])
        if output_dict == None:
            continue
        # remove detection of other classes that are not studied
        idx_to_remove = [i for i, x in enumerate(
            output_dict["detection_classes"]) if x != catIds]
        for index in sorted(idx_to_remove, reverse=True):
            del output_dict['detection_scores'][index]
            del output_dict['detection_classes'][index]
            del output_dict['detection_boxes'][index]
            output_dict["num_detections"] -= 1
        num_detections = output_dict["num_detections"]
        output_dict = {key: np.array(
            output_dict[key]) for key in key_of_interest}
        output_dict["num_detections"] = num_detections
        if len(output_dict['detection_boxes']) == 0:
            output_dict = None
        # [ymin,xmin,ymax,xmax] normalized

        final_classes, final_scores, final_boxes = computeNMS(
            output_dict, iou_threshold=iou_threshold)

        if not final_classes:
            continue

        for j in range(len(final_classes)):

            #ex : {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
            properties = {}
            # json format doesnt support int64
            properties["category_id"] = int(final_classes[j])
            properties["image_id"] = imgId
            im_width = img[i]['width']
            im_height = img[i]['height']
            # we want [ymin,xmin,ymax,xmax] -> [xmin,ymin,width,height]
            properties["bbox"] = putCOCOformat(
                final_boxes[j], im_width, im_height)
            properties["score"] = float(final_scores[j])

            result.append(properties)
    if newFile:
        with open(resFilePath, 'w') as fs:
            json.dump(result, fs, indent=1)
    else:
        with open(resFilePath, 'r') as fs:
            data = json.load(fs)
            result += data
        with open(resFilePath, 'w') as fs:
            json.dump(result, fs, indent=1) 
        

    return list(imgIds)

def getAP05(all_output_dict, img, resFilePath, cocoApi, catIds, catStudied, modelPath, number_IoU_thresh=50):
    """
    input:
        img: coco class describing the images to study
        resFile: Json file describing your bbox OD in coco format
        cocoApi: coco loaded with annotations file
        catIds: list of class index that we are studying
        number_IoU_thresh: The number that will used to compute the different AP
    output:
        List of AP score associated to the list np.linspace(0.01,0.99,number_IoU_thresh)
    """

    iou_thresholdXaxis = np.linspace(0.2, 0.9, number_IoU_thresh)
    AP = []
    FN = []
    computeInstances = True
    for iou in tqdm(iou_thresholdXaxis, desc="progressbar IoU Threshold"):
        # Create the Json result file and read it.
        imgIds = writeResJson(img, resFilePath, all_output_dict,
                              catIds, iou_threshold=float(iou))
        try:
            cocoDt = cocoApi.loadRes(resFilePath)
        except:
            return 1
        cocoEval = COCOeval(cocoApi, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.params.catIds = catIds
        # Here we increase the maxDet to 1000 (same as in model config file)
        # Because we want to optimize the nms that is normally in charge of dealing with
        # bbox that detects the same object twice or detection that are not very precise
        # compared to the best one.
        cocoEval.params.maxDets = [1, 10, 1000]
        cocoEval.evaluate()
        number_FN = 0
        if computeInstances:
            instances_non_ignored = 0
        for evalImg in cocoEval.evalImgs:
            number_FN += sum(evalImg["FN"])
            if computeInstances:
                instances_non_ignored += sum(np.logical_not(evalImg['gtIgnore']))
        computeInstances = False
        FN.append(int(number_FN))
        cocoEval.accumulate(iou,withTrain=WITH_TRAIN,category=catStudied)
        
        cocoEval.summarize()
        # readDoc and find self.evals
        AP.append(cocoEval.stats[1])
        precisions = cocoEval.s.reshape((101,))
        if GRAPH_PRECISION_TO_RECALL:
            precisionToRecall(precisions,round(iou,3),catStudied,modelPath)
    
    general_folder = "{}/AP[IoU=0.5]/".format(modelPath)
    if not os.path.isdir(general_folder):
        os.mkdir(general_folder)
    
    if not WITH_TRAIN:
        general_folder += "validation/" 
    else:
        general_folder += "validation_train/"
    if not os.path.isdir(general_folder):
        os.mkdir(general_folder)
        
    with open(general_folder + "{}.json".format(catStudied), 'w') as fs:
            json.dump({"iou threshold": list(iou_thresholdXaxis), "AP[IoU:0.5]": AP, "False Negatives": FN,
                    "number of instances": int(instances_non_ignored) }, fs, indent=1)

    return 0

def getOverallAP(all_output_dict, resFilePath, cocoApi, modelPath,categories, number_IoU_thresh=50):
    """
    input:
        img: coco class describing the images to study
        resFile: Json file describing your bbox OD in coco format
        cocoApi: coco loaded with annotations file
        catIds: list of class index that we are studying
        number_IoU_thresh: The number that will used to compute the different AP
    output:
        List of AP score associated to the list np.linspace(0.01,0.99,number_IoU_thresh)
    """

    iou_thresholdXaxis = np.linspace(0.2, 0.9, number_IoU_thresh)
    AP = []
    FN = []
    computeInstances = True
    
    for iou in tqdm(iou_thresholdXaxis,desc="progressbar IoU Threshold"):
        # coco = loadCocoApi()
        # with open("{}/all_output_dict.json".format(modelPath), 'r') as fs:
        #     all_output_dict = json.load(fs)
        allCatIds = []
        allImgIds = []
        for i,category in tqdm(enumerate(categories),desc="category"):
            
            img, catIds = getImgClass(category, cocoApi)
            allCatIds += [catIds]
        # Create the Json result file and read it.
            if i == 0:
                imgIds = writeResJson(img, resFilePath, all_output_dict,
                                    catIds, iou_threshold=float(iou))
                
            else:
                imgIds = writeResJson(img, resFilePath, all_output_dict,
                                    catIds, iou_threshold=float(iou),newFile=False)
            allImgIds += imgIds
            
        try:
            cocoDt = cocoApi.loadRes(resFilePath)
        except:
            return 1
        cocoEval = COCOeval(cocoApi, cocoDt, 'bbox')
        cocoEval.params.imgIds = allImgIds
        # cocoEval.params.catIds = allCatIds
        # Here we increase the maxDet to 1000 (same as in model config file)
        # Because we want to optimize the nms that is normally in charge of dealing with
        # bbox that detects the same object twice or detection that are not very precise
        # compared to the best one.
        cocoEval.params.maxDets = [1, 10, 1000]
        cocoEval.evaluate()
        number_FN = 0
        if computeInstances:
            instances_non_ignored = 0
        
        for evalImg in cocoEval.evalImgs:
            if evalImg != None:
                number_FN += sum(evalImg["FN"])
                if computeInstances:
                    instances_non_ignored += sum(np.logical_not(evalImg['gtIgnore']))
        computeInstances = False
        FN.append(int(number_FN))
        cocoEval.accumulate(iou,withTrain=WITH_TRAIN,category='all')
        
        cocoEval.summarize()
        # readDoc and find self.evals
        AP.append(cocoEval.stats[1])
        print(AP)
    
    general_folder = "{}/AP[IoU=0.5]/".format(modelPath)
    if not os.path.isdir(general_folder):
        os.mkdir(general_folder)
    
    if not WITH_TRAIN:
        general_folder += "validation/" 
    else:
        general_folder += "validation_train/"
    if not os.path.isdir(general_folder):
        os.mkdir(general_folder)
        
    with open(general_folder + "all.json", 'w') as fs:
            json.dump({"iou threshold": list(iou_thresholdXaxis), "AP[IoU:0.5]": AP, "False Negatives": FN,
                    "number of instances": int(instances_non_ignored) }, fs, indent=1)

    return 0


def precisionToRecall(precision,iouThreshold, catStudied, modelPath):
    """
    precision -[all] P = 101. Precision for each recall
    catStudied: String describing the category of image studied
    modelPath: path to the model to save the graph
     """
    plt.figure(figsize=(18, 10))
    recall = np.linspace(0, 1, 101)
        
    # Plot the data
    plt.plot(recall, precision, label='Precision to recall for IoU = {}'.format(iouThreshold))
    # Add a legend
    plt.legend(loc="lower left")
    plt.title('Class = {}'.format(catStudied))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    #Create correct folder
    general_folder =  "{}/precision_to_recall/".format(modelPath)
    if not os.path.isdir(general_folder):
        os.mkdir(general_folder)
    
    if WITH_TRAIN:
        general_folder = "{}/precision_to_recall/train_MR/".format(modelPath)
    else:
        general_folder = "{}/precision_to_recall/validation_MR/".format(modelPath)
        
    if not os.path.isdir(general_folder):
        os.mkdir(general_folder)
        
    category_folder = general_folder + catStudied.replace(' ','_')
    if not os.path.isdir(category_folder):
        os.mkdir(category_folder)
        
    plt.savefig(category_folder + '/iou={}.png'.format(iouThreshold), bbox_inches='tight')
    
        
    # plt.clf()
    plt.close('all')


def main(modelPath, resFilePath, cocoDir, valType, number_IoU_thresh=50, catFocus=[]):
    """
    input:

    modelPath: Path to the model of OD
    resFilePath: Path to the json file where OD with nms will be written
    catStudied: String describing of the category of the coco dataset
    nbImageStudied:
    cocoDir: where is the cocoapi
    valType: name of the image folder inside cocoapi

    output:
    Graph of AP[IoU = 0.5] in function of different number_IoU_thresh
    """

    model = loadModel(modelPath)
    coco = loadCocoApi(dataDir=cocoDir, dataType=valType)
    is_all_output_dict = os.path.isfile("{}/all_output_dict.json".format(modelPath))
    if not is_all_output_dict:
        print("Compute all the inferences boxes in the validation set for the model {} and save it for faster computations of you reuse the interface in {}/all_output_dict.json".format(modelPath,modelPath))
        all_output_dict = computeInferenceBbox(model)
        with open("{}/all_output_dict.json".format(modelPath), 'w') as fs:
            json.dump(all_output_dict, fs, indent=1)
    else:
        with open("{}/all_output_dict.json".format(modelPath), 'r') as fs:
            all_output_dict = json.load(fs)

    categories = getCategories(coco) if catFocus == [] else catFocus
    # for catStudied in tqdm(categories, desc="Categories Processed", leave=False):
    #     img, catIds = getImgClass(catStudied, coco)

    #     getAP05(all_output_dict, img, resFilePath, coco,
    #                         catIds, catStudied, modelPath, number_IoU_thresh=number_IoU_thresh)
    getOverallAP(all_output_dict,resFilePath,coco,modelPath,categories)

        
            

WITH_TRAIN = False

#Chose if you want to graph the precision to recall if you want more precision
GRAPH_PRECISION_TO_RECALL = False
models = [
        "ssd_inception_v2",
        "ssd_mobilenet_v1_fpn",
        "ssd_mobilenet_v1",
        ]
# if __name__ == "__main__":
#     # execute only if run as a script
    
#     #Put other model, be aware that the script will insert results in the model folder
   
#     for modelPath in models:
#         for train in [True]:
            
#             WITH_TRAIN = train
#             main(modelPath=modelPath,
#                 resFilePath="cocoapi/results/iou.json",
#                 cocoDir="cocoapi",
#                 valType="val2017",
                # catFocus=[])


main(modelPath="ssd_mobilenet_v1",
        resFilePath="cocoapi/results/iou.json",
        cocoDir="cocoapi",
        valType="val2017",
        catFocus=[])
