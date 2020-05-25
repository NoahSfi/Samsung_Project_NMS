import numpy as np
import json
import matplotlib.pyplot as plt
from nmsAnalysis import nmsAnalysis
from tqdm import tqdm
import os





class optimisedNMS(nmsAnalysis):
    
    #Inside any model directory after running analysis on it
    DIR_GENERAL = "nms_analysis/"
    DIR_ANALYSIS = DIR_GENERAL +  "AP[IoU=0.5]/"
    DIR_VALIDATION = DIR_ANALYSIS +  "validation/"
    DIR_VALIDATION_TRAIN = DIR_ANALYSIS + "validation_train/"
    DIR_MODEL_COMPARISON = "model_comparisons/"
        
    def openJsonData(self,file):
        
        with open(file,"r") as fs:
            data = json.load(fs)
            iou = np.array(data['iou threshold'])
            try:
                AP = np.array(data['AP[IoU:0.5]'])
            except :
                AP = np.array(data['AP[IoU:0.95]'])
                
            numberInstances = data["number of instances"]
            fn = np.array(data['False Negatives'])
        return iou,AP,fn,numberInstances


    def _minMaxScaler(self,array):
        M = max(array)
        m = min(array)
        if m == M:
            return array
        else:
            return (array-m)/(M - m)

    def _plotModels(self,ax,iou,AP,model):
        ax.plot(iou,AP,label = model)
        idx = np.argmax(np.flip(AP))
        xmax = np.flip(iou)[idx]
        ymax = np.flip(AP)[idx]
        ax.annotate('IoU = {}, var = {}'.format(np.format_float_scientific(xmax, unique=False, precision=2),np.format_float_scientific(np.var(AP), unique=False, precision=2)), xy=(xmax, ymax),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )

    def compare_model(self):
            
        if not os.path.isdir(self.DIR_MODEL_COMPARISON):
            os.mkdir(self.DIR_MODEL_COMPARISON)
        
        for category in tqdm(self.categories,desc="model comparisons"):
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18, 10))
            no_detection = 0
            for model in self.models:
                path = model + "/"
                file = category + '.json'
                if not os.path.isfile(path + self.DIR_VALIDATION + file):
                    print("No detection by the model {} for the category {}".format(model,category))
                    no_detection += 1
                    continue
                
                iou,AP,fn,numberInstances = self.openJsonData(path + self.DIR_VALIDATION+file)
                self._plotModels(ax1,iou,AP,model)
                
                iou,AP,fn,numberInstances = self.openJsonData(path + self.DIR_VALIDATION_TRAIN+file)
                self._plotModels(ax2,iou,AP,model)
                
                
            if no_detection == len(self.models):
                plt.close('all')
                continue  
            fig.suptitle("Comparison per model of AP to IoU Threshold for {}".format(category))
            
            ax1.title.set_text("AP[IoU=0.5] validation")
            ax1.set_xlabel("IoU threshold")
            ax1.set_ylabel("AP[IoU=0.5]")
            
            ax2.title.set_text("AP[IoU=0.5] validation with train MR_nms")
            ax2.set_xlabel("IoU threshold")
            ax2.set_ylabel("AP[IoU=0.5]")
            
            ax1.legend(loc = 'lower left')
            ax2.legend(loc = 'lower left')
            fig.savefig(self.DIR_MODEL_COMPARISON + '{}.png'.format(category), bbox_inches='tight')
            plt.close('all')

    # best_overall = argmax ( sum APclass[iou]*var(class))
    def overallArgmax(self,model,computationDir = DIR_VALIDATION_TRAIN,weight = dict()):
        
        data = dict()
        
        final_weight = {category : 1 for category in self.categories}
        for category in weight.keys():
            final_weight[category] = weight[category]
            
        for category in self.categories:
            path = model + "/" 
            file = category + '.json'
            if not os.path.isfile(path + computationDir + file):
                print("No detection by the model {} for the category {}".format(model,category))
                continue
            
            iou,AP,_,_ = self.openJsonData(path + computationDir+file)
            data[category] = AP
            
        overallSum = {
            "AP" : np.zeros(len(iou)),
            "AP*var" : np.zeros(len(iou)),
            "AP*weight" : np.zeros(len(iou)),
        } #All the corresponding sum for each ious i.e sum_{category}formula
        
        for i,iouThresh in enumerate(iou):
            s = 0
            for category,AP in data.items():
                overallSum["AP"][i] += AP[i]
                overallSum["AP*var"][i] += AP[i] * np.var(AP)
                overallSum["AP*weight"][i] += AP[i] * final_weight[category]
                
            
        plt.figure(figsize=(18,10))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(iou,self._minMaxScaler(overallSum["AP*var"]),label = r"$\displaystyle\sum_{cat}AP(cat,iou)*var(AP(cat))$")
        plt.plot(iou,self._minMaxScaler(overallSum["AP"]),label = r"$\displaystyle\sum_{cat}AP(cat,iou)$")
        plt.plot(iou,self._minMaxScaler(overallSum["AP*weight"]),label = r"$\displaystyle\sum_{cat}AP(cat,iou)*w(cat)$")
        # plt.annotate('IoU = {}'.format(np.format_float_scientific(finalIoU, unique=False, precision=2)), xy=(finalIoU, bestScore),arrowprops=dict(facecolor='black', shrink=0.05))
        plt.ylabel("AP[IoU] = 0.5")
        plt.xlabel("IoU Threshold")
        plt.legend()
        
        # plt.rc('text', usetex=False)
        plt.title("Function min max scaled")
        path = model + "/" + self.DIR_GENERAL
        if not os.path.isdir(path + "optimal_overall/"):
            os.mkdir(path + "optimal_overall/")
        plt.savefig(path + "optimal_overall/overallSum_{}.png".format("validation_train" if computationDir == self.DIR_VALIDATION_TRAIN else self.DIR_VALIDATION),bbox_inches='tight')
        
        result = {
            "Normal Distribution": iou[np.argmax(overallSum["AP"])],
            "Weight with variance":iou[np.argmax(overallSum["AP*var"])],
            "Weighted by user": iou[np.argmax(overallSum["AP*weight"])],
        }
        with open(path + "optimal_overall/mean_{}.json".format("validation_train" if computationDir == self.DIR_VALIDATION_TRAIN else self.DIR_VALIDATION),"w") as fs:
            json.dump(result,fs,indent=1)
        return 0


    def plotOverall(self):
        plt.figure(figsize=(18, 10))
        for model in self.models:
            path = model + "/"
            file = 'all.json'
            
            
            iou,AP,fn,numberInstances = self.openJsonData(path + self.DIR_VALIDATION+file)
        
            plt.plot(iou,AP,label = model)
            idx = np.argmax(np.flip(AP))
            xmax = np.flip(iou)[idx]
            ymax = np.flip(AP)[idx]
            plt.annotate('IoU = {}, var = {}'.format(np.format_float_scientific(xmax, unique=False, precision=2),np.format_float_scientific(np.var(AP), unique=False, precision=2)), xy=(xmax, ymax),
            arrowprops=dict(facecolor='black', shrink=0.05))
            
        plt.title("AP[IoU=0.5] validation")
        plt.xlabel("IoU threshold")
        plt.ylabel("AP[IoU=0.5]")
        plt.legend()
        plt.savefig(self.DIR_MODEL_COMPARISON + 'all.png', bbox_inches='tight')
        plt.close('all')

    def writeMapIoU(self,with_train=False):
        end = '\n'
        s = ' '
        for model in self.models:
            path = model + "/"
            open(path + self.DIR_GENERAL + "iouThreshmap.pbtxt","w").close() 
            for category in self.categories:
                file = category + '.json'
                if not os.path.isfile(path + self.DIR_VALIDATION + file):
                    print("No detection by the model {} for the category {}".format(model,category))
                    continue
                catId = self.getCatId(category)
                if not with_train:
                    iou,AP,_,_ = self.openJsonData(path + self.DIR_VALIDATION+file)
                else:
                    iou,AP,fn,numberInstances = self.openJsonData(path + self.DIR_VALIDATION_TRAIN+file)
                    
                idx = np.argmax(np.flip(AP))
                bestIoU = np.flip(iou)[idx]
                out = ''
                out += 'item' + s + '{' + end
                out += s*2 + 'id:' + s + str(catId) + end
                out += s*2 + 'display_name:' + s + str(category)  + end
                out += s*2 + 'iou_threshold:' + s  + str(bestIoU)  + end
                out += '}' + end*2
                with open(path + self.DIR_GENERAL + "iouThreshmap.pbtxt","a") as fs:
                    fs.write(out) 
