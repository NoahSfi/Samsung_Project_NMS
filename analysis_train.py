#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:44:21 2020

@author: noahsarfati
"""

###############################################################################

""" Please download in the cocoapi folder d `2017 val/train annotations` under the names `annotations`"""

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
import random

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

###############################################################################
from NMS_project import loadCocoApi,getCategories,getImgClass




def getBbox(coco,image_Id,catId):
    """
    Input:
        [left,top,width,height]
    output:
        List of the form [xmin,ymin,xmax,ymax]  describing each bbox of the given image_Id0
    """
    annIds = coco.getAnnIds(imgIds=image_Id, catIds=catId, iscrowd=None)
    anns = coco.loadAnns(annIds)
    bbox = list()
    for annotation in anns:
        box_annotation = annotation["bbox"]
        bbox.append(box_annotation)
    return bbox

def IoU(box1,box2):
    """
        Input:
            box1 ~ box2 : [xmin,ymin,xmax,ymax]
        Output:
            Intersection over Union of the inputs
            return a negative value if there is no intersection
    """
    #check if there is an intersection
    #1- check if x-axis coincides:
    rightBox = max(box1,box2,key = lambda x : x[0])
    leftBox = box1 if rightBox is box2 else box2
    
    topBox = max(box1,box2,key = lambda x : x[3])
    bottomBox = box1 if topBox is box2 else box2
    
    if (leftBox[0] + leftBox[2] <= rightBox[0]) or (bottomBox[1]<= topBox[1] - topBox[3]) :
        return 0

    # determine the (x, y)-coordinates of the intersection rectangle
    xmin = max(box1[0],box2[0])
    ymin = max(box1[1] - box1[3], box2[1] -  box2[3])
    xmax = min(box1[0] + box1[2], box2[0] + box2[2])
    ymax = min(box1[1], box2[1])

    # compute the area of intersection rectangle
    interArea = (xmax - xmin) * (ymax - ymin)
    
    # compute the area of both boxes
    box1Area = box1[2] * box1[3] 
    box2Area = box2[2] * box2[3]
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    
    iou = interArea / (box1Area+box2Area-interArea)
    
    return iou
    

def pseudoNMS(bbox,iou_threshold,seed = 30):
    random.seed(seed)
    finalBbox = list()
    #won't modify the input
    bbox_to_study = list(bbox)
    while bbox_to_study:
        
        #nothing to compare with case
        if len(bbox) == 1:
            finalBbox.append(bbox_to_study[0])
            del bbox_to_study[0]
            continue
        
        idx = random.randint(0,len(bbox_to_study)-1)
        bboxToCompare = bbox_to_study[idx]
        finalBbox.append(bboxToCompare)
        del bbox_to_study[idx]
        #decreasing order to be able to remove object by their indexes
        for i in range(len(bbox_to_study)-1,-1,-1):
            iou = IoU(bboxToCompare,bbox_to_study[i])
            if iou > iou_threshold:
                del bbox_to_study[i]
    return finalBbox

   
def writeResToJson(resFilePath,coco,catIds,img,iou_threshold):
    result = []
    imgIds = set() #set to avoid repetition
    for image in img:
        image_Id = image["id"]
        bbox = getBbox(coco,image_Id,catIds)
        bboxAfterNms = pseudoNMS(bbox,iou_threshold)
        for i in range(len(bboxAfterNms)):
            #ex : {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
            properties = {}
            #json format doesnt support int64
            properties["category_id"] = int(catIds)
            properties["image_id"] = int(image_Id)
            imgIds.add(image_Id)
            

            #we want [ymin,xmin,ymax,xmax] -> [xmin,ymin,width,height]
            properties["bbox"] = bboxAfterNms[i]
            properties["score"]= float(1.)

            result.append(properties)
    with open(resFilePath, 'w+') as fs:
        json.dump(result, fs, indent=1)
    return list(imgIds) 

def getAP05(img,resFilePath,cocoApi,catIds,number_IoU_thresh = 50):
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
    
    iou_thresholdXaxis = np.linspace(0.2,0.9,number_IoU_thresh)
    AP = []
    res_iou = list()
    for iou_threshold in tqdm(iou_thresholdXaxis,desc = "progressbar IoU Threshold"):
        #Create the Json result file and read it.
        imgIds = writeResToJson(resFilePath,cocoApi,catIds,img,iou_threshold)
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
        # for iou_arrays in cocoEval.ious.values():
          
        #     for iou in iou_arrays[0]:
        #         if iou > 0.1 and iou <0.98: 
        #             res_iou.append(iou)
        # print(res_iou)
        # return res_iou
        cocoEval.accumulate()
        cocoEval.summarize()
        #readDoc and find self.evals
        AP.append(cocoEval.stats[1])
    
    return AP



# def getIoU(coco,catIds):
    
#     iou = list()
#     imgIds = coco.getImgIds(catIds=catIds)
#     cocoDt=cocoApi.loadRes(resFilePath)
#     cocoEval = COCOeval(cocoApi,cocoDt,'bbox')
#     cocoEval.params.imgIds  = imgIds
#     cocoEval.params.catIds  = catIds
#     #Here we increase the maxDet to 1000 (same as in model config file)
#     #Because we want to optimize the nms that is normally in charge of dealing with
#     #bbox that detects the same object twice or detection that are not very precise
#     #compared to the best one.
#     cocoEval.params.maxDets = [1,10,1000]
#     cocoEval.evaluate()

def plotAP(AP,catStudied,number_IoU_thresh):
    
    plt.figure(figsize=(18,10))
    iou_thresholdXaxis = np.linspace(0.2,0.9,number_IoU_thresh)
    # Plot the data
    plt.plot(iou_thresholdXaxis, AP, label='AP[IoU=0.5]')
    # Add a legend
    plt.legend(loc = "lower left")
    plt.title('Class = {}'.format(catStudied))
    plt.xlabel('iou threshold')
    plt.ylabel('AP[IoU=0.5]')
    plt.savefig('graph_result_train/{}.png'.format(catStudied), bbox_inches='tight')
    plt.clf()
   
def plotHistIou(ious,catStudied):
    plt.figure(figsize=(18,10))
    nb_bins = 20
    plt.hist(ious,bins=nb_bins)
    plt.savefig('graph_result_train/hist_{}.png'.format(catStudied), bbox_inches='tight')
    plt.clf()

coco = loadCocoApi(dataType="train2017")
resFilePath = "cocoapi/results/train_res.json"
img,catIds = getImgClass("car",coco,float("inf"))

AP = getAP05(img,resFilePath,coco,catIds[0])  
# plotHistIou(ious,"car")

plotAP(AP,"car",50)

        
          