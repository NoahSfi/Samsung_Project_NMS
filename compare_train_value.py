import numpy as np
import json
import matplotlib.pyplot as plt
from analysis_validation import loadCocoApi,getCategories
modelPath = "ssd_mobilenet_v1/"
dataDirVal = "validation_AP/"
dataDirValTrain = "validation_train/"
DataDirTrain = 'cocoapi/results/train/'
test = 'bus.json'
cocoDir="cocoapi"
valType="val2017"

def openJsonData(file):
    
    with open(file,"r") as fs:
        data = json.load(fs)
        iou = np.array(data['iou threshold'])
        try:
            AP = np.array(data['AP[IoU:0.5]'])
        except :
            AP = np.array(data['AP[IoU:0.95]'])
            
        numberInstances = data['instances_not_ignored']
        fn = np.array(data['False Negatives'])
    return iou,AP,fn,numberInstances


def fn_weight(AP,fn,numberInstances):
    
    return AP * (1 - fn/numberInstances)

def minMaxScaler(data):
    M = max(data)
    m = min(data)
    if m == M:
        return data
    else:
        return (data-m)/(M - m)
   

def main():
    coco = loadCocoApi()
    categories = getCategories(coco)
    for category in categories:
        try:
            iou, AP_value, _, _ = openJsonData(modelPath + dataDirVal + category + '.json')
            _ , _, fn_train, numberInstances_train = openJsonData(DataDirTrain + category + '.json')

            _ , AP_value_train, _, _ = openJsonData(modelPath + dataDirValTrain +category+'.json')
        except :
            print(category)
            continue


        plt.plot(iou,minMaxScaler(AP_value),label = "Validation : AP")
        plt.plot(iou,minMaxScaler(1 - fn_train/numberInstances_train),label = "Recall train")
        plt.plot(iou,minMaxScaler(minMaxScaler(AP_value) + minMaxScaler(1 - fn_train/numberInstances_train)),label = "Blue + Orange")
        plt.plot(iou,minMaxScaler(AP_value_train),label = "AP with MR = MRerr + MR_nms_train ")
        plt.xlabel("Iou Threshold")
        plt.ylabel("Score")
        plt.title(category)
        plt.legend()
        plt.savefig("{}/graph_optimised/{}.png".format(modelPath, category),bbox_inches='tight')

        plt.cla()

main()