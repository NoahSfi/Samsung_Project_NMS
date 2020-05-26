import numpy as np
import json
import matplotlib.pyplot as plt
from nmsAnalysis import nmsAnalysis
from tqdm import tqdm
import os





class optimisedNMS(nmsAnalysis):
    
    """
    The goal of this class is to use the results of `nmsAnalysis` in order to compute the optimal 
    IoU Threshold to use in the normal suppression algorithm.
    
    The initialisation is the same than `nmsAnalysis`, `imagePath` can be set to None.
    
    In the continuity of `nmsAnalysis` all the results will be written inside your modelPath in the 
    folder `nmsAnalysis`.
    
    - Here is a complete example of use if one only wants to deal with the validation data set.
    
        - analyser = nmsAnalysis(models,imagesPath,annotationValidation,catFocus,number_IoU_thresh,overall)
        - analyser.with_train = False
        - analyser.runAnalysis()
        - optimiser = optimisedNMS(models,None,annotationValidation,catFocus=catFocus)
        -optimiser.with_train = False
        -optimiser.compare_model()
        -for model in models:
            - optimiser.overallArgmax(model)
        -optimiser.plotOverall()
        -optimiser.writeMapIoU()
    
    - Here is a complete example of use if one wants to replace the ratio of false negatives generated
    by the nms algoeithm in the validation dataset by the one in the training dataset. The run will be
    longer but the results will be more precise. In any case one will be able to visualise the difference
    with optimiser.compare_model().

        - fn_validation = GroundTruthFN(annotationValidation,dataType="validation",catFocus=catFocus)
        - fn_validation.runAnalysis()
        - fn_train = GroundTruthFN(annotationTrain,dataType="train",catFocus=catFocus)
        - fn_train.runAnalysis()
        
        - analyser = nmsAnalysis(models,imagesPath,annotationValidation,catFocus,number_IoU_thresh,overall)
        - analyser.with_train = True
        - analyser.runAnalysis()
        - optimiser = optimisedNMS(models,None,annotationValidation,catFocus=catFocus)
        - optimiser.with_train = True
        - optimiser.compare_model()
        - for model in models:
            - optimiser.overallArgmax(model)
        - optimiser.plotOverall()
        - optimiser.writeMapIoU()
    
    """
    
    #Inside any model directory after running analysis on it
    DIR_GENERAL = "nms_analysis/"
    DIR_ANALYSIS = DIR_GENERAL +  "AP[IoU=0.5]/"
    DIR_VALIDATION = DIR_ANALYSIS +  "validation/"
    DIR_VALIDATION_TRAIN = DIR_ANALYSIS + "validation_train/"
    DIR_MODEL_COMPARISON = "model_comparisons/"
        
    def openJsonData(self,file):
        """
        Load a json file in the shape of the one written in `nmsAnalysis`.
        :return:
        
        - iou: List of different IoU treshold studied
        - AP:  AP corresponding to each treshold
        - fn:  false negatives corresponding to each treshold
        - numberInstances: The number of instances that were evaluated.
        """
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
        """
        :param array: array of digits
        
        :return: (array-min)/(max - min) if max != min else array.
        """
        M = max(array)
        m = min(array)
        if m == M:
            return array
        else:
            return (array-m)/(M - m)

    def _plotModels(self,ax,iou,AP,model):
        """
        Plot the AP to IoU for a given model onto a given axis of the graph
        :param ax: a matplotlib.pyplot axis
        :param iou: list of IoU Threshold i.e the X-axis.
        :param AP: list of corresponding AP i.e the Y-axis
        :param model: Under which model were the results found
        
        :return: None
        """
        ax.plot(iou,AP,label = model)
        idx = np.argmax(np.flip(AP))
        xmax = np.flip(iou)[idx]
        ymax = np.flip(AP)[idx]
        ax.annotate('IoU = {}, var = {}'.format(np.format_float_scientific(xmax, unique=False, precision=2),np.format_float_scientific(np.var(AP), unique=False, precision=2)), xy=(xmax, ymax),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )

    def compare_model(self):
        """
        Plot the AP to IoU treshold for each `self.models` and each category onto the same graph.
        The results will be written in the folder `model_comparisons` in the relative path.
        If `self.withTrain` is set to True, 2 axis will be plot one using only the validation set and one
        using the fn/npig ratio of the training dataset.
        """
        if not os.path.isdir(self.DIR_MODEL_COMPARISON):
            os.mkdir(self.DIR_MODEL_COMPARISON)
        
        for category in tqdm(self.categories,desc="model comparisons"):
            if self.with_train:
                fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18, 10))
            else:
                fig,ax1 = plt.subplots(1, 1,figsize=(18, 10))
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
                if self.with_train:
                    iou,AP,fn,numberInstances = self.openJsonData(path + self.DIR_VALIDATION_TRAIN+file)
                    self._plotModels(ax2,iou,AP,model)
                
                
            if no_detection == len(self.models):
                plt.close('all')
                continue  
            fig.suptitle("Comparison per model of AP to IoU Threshold for {}".format(category))
            
            ax1.title.set_text("AP[IoU=0.5] validation")
            ax1.set_xlabel("IoU threshold")
            ax1.set_ylabel("AP[IoU=0.5]")
            ax1.legend(loc = 'lower left')
            if self.with_train:
                ax2.title.set_text("AP[IoU=0.5] validation with train MR_nms")
                ax2.set_xlabel("IoU threshold")
                ax2.set_ylabel("AP[IoU=0.5]")
                ax2.legend(loc = 'lower left')
            
            
            
            fig.savefig(self.DIR_MODEL_COMPARISON + '{}.png'.format(category), bbox_inches='tight')
            plt.close('all')


    def overallArgmax(self,model,weight = dict()):
        """
        Compute the overall best IoU treshold for the entire `self.categories`. 
        
        3 results will be given:
        
        - argmax_{iou}(sum_{cat}AP(cat,iou))
        - argmax_{iou}(sum_{cat}AP(cat,iou)*var(AP(cat)))
        - argmax_{iou}(sum_{cat}AP(cat,iou)*weight(cat))
        
        Each formula inside the argmax will be ploted. The result of it will be written
        inside `argmax_{computationDir}.json`
        
        Results can be found in the model folder inside `nms_analysis/optimal_overall`
        
        :param model: path to the model to study
        :weight: dictionary of the form {category: weight} if not precise each category has a weight of 1.
        
        :return: None
        """
        data = dict()
        computationDir = self.DIR_VALIDATION if not self.with_train else self.DIR_VALIDATION_TRAIN
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
        with open(path + "optimal_overall/argmax_{}.json".format("validation_train" if computationDir == self.DIR_VALIDATION_TRAIN else self.DIR_VALIDATION),"w") as fs:
            json.dump(result,fs,indent=1)
        


    def plotOverall(self):
        """
        Plot the AP to iouThreshold for the overall categories.
        Inside the plot there will be a curve for each model in `self.models`.
        
        Result will be found in the `model_comparisons` folder in the relative path.
        """
        
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
        """
        For each model in self.models write a pbtxt of the form:
        
        - item {
            id: 2
            display_name: category
            iou_threshold: optimal iou
        }
        
        The file will be in `modelPath/nms_analysis/iouThreshmap.pbtxt`
        """
        
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
