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
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from nmsAnalysis import nmsAnalysis
import random
import os

###############################################################################



class GroundTruthFN(nmsAnalysis):
    
    """
    annotationPath: Path to annotations
    dataType: Either "train" or "validation"
    catFocus: If None will consider all categories in annotations else consider the list of categories one input. For example ["person","car"]
    number_IoU_tresh: Number of different threshold to consider in between 0.2 and 0.9
    """ 
    
    def __init__(self,annotationPath,dataType = "train" ,catFocus=None, number_IoU_thresh=50):
        
        print("Initialization....")
        self.DIRECTORY = "FN_with_nms/"
        self.resultPath = "trainFN/" if "train" in dataType else "validationFN/"
        self.annotationPath = annotationPath
        self.resFilePath = self._createResFilePath()
        self.dataType = dataType
        self.number_IoU_thresh = number_IoU_thresh
        self.iou_thresholdXaxis = np.linspace(0.2, 0.9, number_IoU_thresh)

        self.coco = self.loadCocoApi()
        self.categories = self.getCategories() if catFocus is None else catFocus

        # All the variable that will change throughout the study and will be needed in many functions
        self._study = {
            "img": dict(),
            "catId": int(),
            "catStudied": str(),
            "all_output_dict": dict(),
            "iouThreshold": float(),
        }
        
        if not os.path.isdir(self.DIRECTORY):
            os.mkdir(self.DIRECTORY)

        if not os.path.isdir(self.DIRECTORY + self.resultPath):
            os.mkdir(self.DIRECTORY + self.resultPath)

    def getBbox(self,image_Id):
        """
        Input:
            [left,top,width,height]
        output:
            List of the form [xmin,ymin,xmax,ymax]  describing each bbox of the given image_Id
        """
        annIds = self.coco.getAnnIds(imgIds=image_Id, catIds=self._study["catId"], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        bbox = list()
        for annotation in anns:
            box_annotation = annotation["bbox"]
            bbox.append(box_annotation)
        return bbox

    def IoU(self,box1,box2):
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
        

    def pseudoNMS(self,bbox,seed = 30):
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
                iou = self.IoU(bboxToCompare,bbox_to_study[i])
                if iou > self._study["iouThreshold"]:
                    del bbox_to_study[i]
        return finalBbox

    
    def writeResToJson(self,newFile = True):
        result = []
        imgIds = set() #set to avoid repetition
        for image in self._study["img"]:
            image_Id = image["id"]
            imgIds.add(image_Id)
            bbox = self.getBbox(image_Id)
            bboxAfterNms = self.pseudoNMS(bbox)
            for i in range(len(bboxAfterNms)):
                #ex : {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
                properties = {}
                #json format doesnt support int64
                properties["category_id"] = int(self._study["catId"])
                properties["image_id"] = int(image_Id)
                
                

                #we want [ymin,xmin,ymax,xmax] -> [xmin,ymin,width,height]
                properties["bbox"] = bboxAfterNms[i]
                properties["score"]= float(1.)

                result.append(properties)
                
        if newFile:
            with open(self.resFilePath, 'w') as fs:
                json.dump(result, fs, indent=1)
        else:
            with open(self.resFilePath, 'r') as fs:
                data = json.load(fs)
                result += data
            with open(self.resFilePath, 'w') as fs:
                json.dump(result, fs, indent=1) 
        return list(imgIds) 

    def getClassAP(self):
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
        
        
        AP = list()
        FN = list()
        for iouThreshold in tqdm(self.iou_thresholdXaxis,desc = "progressbar IoU Threshold"):
            self._study["iouThreshold"] = iouThreshold
            #Create the Json result file and read it.
            imgIds = self.writeResToJson()
            cocoDt= self.coco.loadRes(self.resFilePath)
            cocoEval = COCOeval(self.coco,cocoDt,'bbox')
            cocoEval.params.imgIds  = imgIds
            cocoEval.params.catIds  = self._study["catId"]
            #Here we increase the maxDet to 1000 (same as in model config file)
            #Because we want to optimize the nms that is normally in charge of dealing with
            #bbox that detects the same object twice or detection that are not very precise
            #compared to the best one.
            cocoEval.params.maxDets = [1,10,1000]
            cocoEval.evaluate()
            number_FN = 0
            instances_non_ignored = 0
            for evalImg in cocoEval.evalImgs:
                number_FN += sum(evalImg["FN"])
                
                instances_non_ignored += sum(np.logical_not(evalImg['gtIgnore']))     
            FN.append(int(number_FN))
            #Need it only once
            cocoEval.accumulate(iouThreshold,withTrain=False)
            cocoEval.summarize()
            #readDoc and find self.evals
            #modified version of pycocotools to have 3rd argument to be AP[IoU = 0.95]
            AP.append(cocoEval.stats[2])
        with open(self.DIRECTORY + self.resultPath+ "{}.json".format(self._study["catStudied"]), 'w') as fs:
            json.dump({"iou threshold": list(self.iou_thresholdXaxis),"AP[IoU:0.95]":AP,"False Negatives":FN,"number of instances":int(instances_non_ignored)}, fs, indent=1)
        return AP

    

    def getIoU(self):
        
        res_iou = list()
        self._study["iouThreshold"] = 1
        imgIds = self.writeResToJson()
        cocoDt=self.coco.loadRes(self.resFilePath)
        cocoEval = COCOeval(self.coco,cocoDt,'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = self._study["catId"]
        #Here we increase the maxDet to 1000 (same as in model config file)
        #Because we want to optimize the nms that is normally in charge of dealing with
        #bbox that detects the same object twice or detection that are not very precise
        #compared to the best one.
        cocoEval.params.maxDets = [1,10,1000]
        cocoEval.evaluate()
        for iou_arrays in cocoEval.ious.values():
            for iou in iou_arrays[0]:
                if iou > 0.1 and iou <0.98: 
                    res_iou.append(iou)
            
        return res_iou
    def plotAP(self,AP,catStudied,number_IoU_thresh):
        
        plt.figure(figsize=(18,10))
        # Plot the data
        plt.plot(self.iou_thresholdXaxis, AP, label='AP[IoU=0.95]')
        # Add a legend
        plt.legend(loc = "lower left")
        plt.title('Class = {}'.format(catStudied))
        plt.xlabel('iou threshold')
        plt.ylabel('AP[IoU=0.95]')
        plt.savefig(self.DIRECTORY + self.resultPath+ 'graph/graph_{}.png'.format(catStudied), bbox_inches='tight')
        plt.clf()
    
    def plotHistIou(self,ious):
        plt.figure(figsize=(18,10))
        nb_bins = 20
        plt.hist(ious,bins=nb_bins)
        plt.ylabel('Number of detections')
        plt.xlabel("IoU")
        plt.title('Class = {}'.format(self._study["catStudied"]))
        plt.savefig(self.DIRECTORY + self.resultPath+ 'graph/hist_{}.png'.format(self._study["catStudied"]), bbox_inches='tight')
        plt.clf()
        
    def runAnalysis(self):
        
        print("Analysing {} ...".format(self.dataType))
        for catStudied in tqdm(self.categories,desc="Categories Processed",leave=False):
            self._study["catStudied"] = catStudied
            self.getImgClass(catStudied)
            AP = self.getClassAP()  
            ious = self.getIoU()
            if not os.path.isdir(self.DIRECTORY + self.resultPath + "graph/"):
                os.mkdir(self.DIRECTORY + self.resultPath + "graph/")
            self.plotHistIou(ious)
            self.plotAP(AP,catStudied,50)