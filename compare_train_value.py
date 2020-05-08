import numpy as np
import json
import matplotlib.pyplot as plt

modelPath = "ssd_mobilenet_v1_fpn/"
dataDir = "validation_AP/"
trainDataDir = 'cocoapi/results/train/'
test = 'bus.json'
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
    return (data-m)/(M - m)

iou, AP_value, fn_value, numberInstances_value = openJsonData(modelPath + dataDir + test)
_ , AP_train, fn_train, numberInstances_train = openJsonData(trainDataDir + test)

_ , AP_value_train, fn_value_train, numberInstances_value_train = openJsonData(modelPath + "validation_train/" + test)


# plt.plot(iou,minMaxScaler(fn_weight(AP_value,fn_value,numberInstances_value)),label = "AP *(1 - #fn/#instances)")
# plt.plot(iou,minMaxScaler(fn_weight(AP_train,fn_train,numberInstances_train)))

plt.plot(iou,minMaxScaler(AP_value),label = "Validation : AP")
# plt.plot(iou,minMaxScaler(AP_train),label = "AP_train")
plt.plot(iou,minMaxScaler(1 - fn_train/numberInstances_train),label = "Recall train")
plt.plot(iou,minMaxScaler(minMaxScaler(AP_value) + minMaxScaler(1 - fn_train/numberInstances_train)),label = "Blue + Orange")
plt.plot(iou,minMaxScaler(AP_value_train),label = "AP with MR = MRerr + MR_nms_train ")
plt.xlabel("Iou Threshold")
plt.ylabel("Score")
plt.title("Bicycle")
plt.legend()
plt.savefig("optimised_bicycle.png",bbox_inches='tight')

plt.show()