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

import numpy as np
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.core import post_processing
from tqdm import tqdm
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

###############################################################################

#Load object detection model

def loadModel(modelPath):
    """Load associate tf OD model"""
    model_dir = modelPath + "/saved_model"
    detection_model = tf.saved_model.load(str(model_dir))
    detection_model = detection_model.signatures['serving_default']
    return detection_model


def loadCocoApi(dataDir = "cocoapi",dataType = "val2017"):
    """Read annotations file and return the associate cocoApi"""
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    return coco

def getCategories(coco):
    """Return list of categories one can study"""
    cats = coco.loadCats(coco.getCatIds())
    categories=[cat['name'] for cat in cats]
    return categories

def getImgClass(catStudied,coco,nbImageStudied):
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
    catIds = coco.getCatIds(catNms=catStudied)
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds)
    nbImg = min(len(img),nbImageStudied)
    img = np.random.choice(img,nbImg)
    return img,catIds

def expand_image_to_4d(image):
    """image a numpy array representing a gray scale image"""
    # The function supports only grayscale images
    assert len(image.shape) == 2, "Not a grayscale input image" 
    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3dims = np.expand_dims(image, last_axis)
    training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
    assert len(training_image.shape) == 3
    assert training_image.shape[-1] == 3
    return training_image
###############################################################################

def run_inference_for_single_image(model, image,catIds):
    
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
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    #If image doesn't respect the right format ignore it
    try:
       output_dict = model(input_tensor)
    except:
       return None
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
    
    
    key_of_interest = ['detection_scores','detection_classes','detection_boxes']
    output_dict = { key: list(output_dict[key]) for key in key_of_interest}
    output_dict["num_detections"] = num_detections
  
    #remove detection of other classes that are not studied
    idx_to_remove = [i for i,x in zip(range(len(output_dict["detection_classes"])),output_dict["detection_classes"]) if x not in catIds]
    for index in sorted(idx_to_remove, reverse=True):
        del output_dict['detection_scores'][index]
        del output_dict['detection_classes'][index]
        del output_dict['detection_boxes'][index]
        output_dict["num_detections"] -= 1
    num_detections = output_dict["num_detections"]
    output_dict = { key: np.array(output_dict[key]) for key in key_of_interest}
    output_dict["num_detections"] = num_detections

    if len(output_dict["detection_boxes"]) == 0:
        return None
    return output_dict
def computeInferenceBbox(model, img,catIds):
    
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
    for i in tqdm(range(len(img)),position= 0,leave=True,desc = "Images Processed"):
        image_path = 'cocoapi/val2017/' + img[i]['file_name']
        # the array based representation of the image 
        image = Image.open(image_path)
        image_np = np.array(image)
        """If image is gray_scale one need to reshape to dimension 4
        using the utility function defined above"""
        if len(image_np.shape) == 2:
            image_np = expand_image_to_4d(image_np)
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np,catIds)
        idx = img[i]['id']
        all_output_dict[idx] = output_dict
    return all_output_dict


def computeNMS(output_dict,iou_threshold = 0.6):
    
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
    
    if not output_dict : return None,None,None
    #box_selection = post_processing.multiclass_non_max_suppression([output_dict['detection_boxes']], output_dict['detection_scores'], score_thresh=.8, iou_thresh=.5, max_size_per_class=0)
    
    # Apply the nms
    
    box_selection = tf.image.non_max_suppression_with_scores(
        output_dict['detection_boxes'],output_dict['detection_scores'], 100, 
        iou_threshold= iou_threshold,score_threshold=float('-inf'), 
        soft_nms_sigma=0.0, name=None)
    
    
    #Index in the list output_dict['detection_boxes']
    final_boxes = list(box_selection[0].numpy())
    #Index in the list output_dict['detection_scores']
    final_scores = list(box_selection[1].numpy())
    final_classes = []
    for i in range(len(final_boxes)):
        index = final_boxes[i]
        #We want the actual bbox coordinate not the index
        final_boxes[i] = output_dict['detection_boxes'][index]
        final_classes.append(output_dict['detection_classes'][index])
    
    return final_classes,final_scores,final_boxes

###############################################################################

def putCOCOformat(boxes,im_width,im_height):
    """
    Transform a bbox in the OD format into cocoformat
    input:
        boxes: List of the form [ymin,xmin,ymax,xmax] in the percentage of the image scale
        im_width: real width of the associated image
        im_height: real height of the associated image
    output:
        List of the form [left,top,width,height] describing the bbox, in the image scale
    """
    #float to respect json format
    left = float(boxes[1]) * im_width
    right = float(boxes[3]) * im_width
    top = float(boxes[0]) * im_height
    bottom = float(boxes[2]) * im_height
    width = right - left
    height = bottom - top
    
    return [left,top,width,height]


def writeResJson(img,resFilePath, all_output_dict,catIds,iou_threshold):
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
    imgIds = set() #set to avoid repetition
    
    for i in range(len(img)):

        #[ymin,xmin,ymax,xmax] normalized
        final_classes,final_scores,final_boxes = computeNMS(all_output_dict[img[i]['id']],iou_threshold=iou_threshold)

        if not final_classes: 
            continue
        for j in range(len(final_classes)):
            
            #ex : {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
            properties = {}
            #json format doesnt support int64
            properties["category_id"] = int(final_classes[j])
            imgId = img[i]["id"]
            properties["image_id"] = imgId
            imgIds.add(imgId)
            im_width = img[i]['width']
            im_height = img[i]['height']

            #we want [ymin,xmin,ymax,xmax] -> [xmin,ymin,width,height]
            properties["bbox"] = putCOCOformat(final_boxes[j],im_width,im_height)
            properties["score"]= float(final_scores[j])

            result.append(properties)
    with open(resFilePath, 'w+') as fs:
        json.dump(result, fs, indent=1)
    return list(imgIds)



def getAP05(model,img,resFilePath,cocoApi,catIds,catStudied,number_IoU_thresh = 50):
    """
    input:
        model : OD detector
        img: coco class describing the images to study
        resFile: Json file describing your bbox OD in coco format
        cocoApi: coco loaded with annotations file
        catIds: list of class index that we are studying
        number_IoU_thresh: The number that will used to compute the different AP
    output:
        List of AP score associated to the list np.linspace(0.01,0.99,number_IoU_thresh)
    """
    
    iou_thresholdXaxis = np.linspace(0.1,0.99,number_IoU_thresh)
    AP = []
    all_output_dict = computeInferenceBbox(model,img,catIds)
    for iou in tqdm(iou_thresholdXaxis,desc = "progressbar IoU Threshold"):
        #Create the Json result file and read it.
        imgIds = writeResJson(img,resFilePath,all_output_dict,catIds,iou_threshold=float(iou))
        cocoDt=cocoApi.loadRes(resFilePath)
        cocoEval = COCOeval(cocoApi,cocoDt,'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = catIds
        #Here we increase the maxDet to 1000 (same as in model config file)
        #Because we want to optimize the nms that is normally in charge of dealing with
        #bbox that detects the same object twice or detection that are not very precise
        #compared to the best one.
        cocoEval.params.maxDets = [1,10,1000]
        cocoEval.evaluate()
        cocoEval.accumulate()
        # cocoEval.summarize()
        #readDoc and find self.evals
        AP.append(cocoEval.stats[1])
    with open("class_value_AP/{}.json".format(catStudied), 'w') as fs:
        json.dump([{"iou_threshold": list(iou_thresholdXaxis),"AP[IoU:0.5]":AP}], fs, indent=1)
    return np.array(AP),round(iou_thresholdXaxis[AP.index(max(AP))],4)

def plotAP(AP,catStudied,number_IoU_thresh = 50):
    """
    AP: List of score AP
    catStudied: String describing the category of image studied
     """
    plt.figure(figsize=(18,10))
    iou_thresholdXaxis = np.linspace(0.1,0.99,number_IoU_thresh)
    #Put an arrow on the max value
    IoU_max = iou_thresholdXaxis[np.argmax(AP)]
    AP_max = AP.max()
    text= "Best IoU_Thresh ={:.3f}".format(IoU_max)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    plt.annotate(text, xy=(IoU_max, AP_max), xytext=(0.94,0.96), **kw)

    # Plot the data
    plt.plot(iou_thresholdXaxis, AP, label='AP[IoU=0.5]')
    # Add a legend
    plt.legend(loc = "lower left")
    plt.title('Class = {}'.format(catStudied))
    plt.xlabel('iou threshold')
    plt.ylabel('AP[IoU=0.5]')
    plt.savefig('graph_result/{}.png'.format(catStudied), bbox_inches='tight')
    plt.clf()

def main(modelPath,resFilePath,nbImageStudied,cocoDir,valType,number_IoU_thresh = 50,catFocus = []):
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
    coco = loadCocoApi(dataDir=cocoDir,dataType=valType)
    
    categories = getCategories(coco) if catFocus == [] else catFocus
    end = '\n'
    s = ' '
    with open("string_float_iouThresh_map_pb2.pbtxt", 'w+') as f:
        for catStudied in tqdm(categories,desc="Categories Processed",leave=False):
            img,catIds = getImgClass(catStudied,coco,nbImageStudied)
            if len(img) == 0:
                #No image from the given category
                #Write it in order to know which one
                img = Image.new('RGB', (100, 30), color = (73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((10,10), "No Data to study", fill=(255,255,0))
                img.save('graph_result/{}.png'.format(catStudied))
                out = ''
                out += 'item' + s + '{' + end
                out += s*2 + 'id:' + ' ' + (str(catIds[0])) + end
                out += s*2 + 'display_name:' + s + catStudied  + end
                out += '}' + end*2
                f.write(out)
                continue
            AP05,iou = getAP05(model,img,resFilePath,coco,catIds,catStudied,number_IoU_thresh=number_IoU_thresh)
            
            out = ''
            out += 'item' + s + '{' + end
            out += s*2 + 'id:' + ' ' + (str(catIds[0])) + end
            out += s*2 + 'display_name:' + s + catStudied  + end
            out += s*2 + 'iou_threshold:' + s + str(iou) + end
            out += '}' + end*2
            f.write(out)
            
            plotAP(AP05,catStudied,number_IoU_thresh=number_IoU_thresh)


if __name__ == "__main__":
    # execute only if run as a script
    main(modelPath = "model",
        resFilePath = "cocoapi/results/iou.json",
        nbImageStudied = float("inf"),
        cocoDir = "cocoapi",
        valType = "val2017",
        catFocus= []) 







