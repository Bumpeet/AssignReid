import torchreid
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from torch.nn import functional as F
import utils
import shutil
import os
import re
from tqdm import tqdm
from time import perf_counter
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

class RadiusRequirement():
    def __init__(self, model, weights, imgs, bs, n_batches, thresh):
        self.device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.extractor = torchreid.utils.FeatureExtractor(model,weights, device=str(self.device))

        self.bs = bs
        self.n_batches = n_batches
        self.thresh = thresh
        
        self.imgs_list = imgs
        self.ams = None
        self.features = None
        self.cluster_idx = None

    def Task1(self):

        imgs_batches = utils.generate_batches(self.imgs_list, self.bs)
        self.features, self.ams = self.run_inference_batches(imgs_batches)
        self.cluster_idx = self.cluster()

        print(f'The total number of unique persons among {len(self.imgs_list)} images are {len(self.cluster_idx)} for threshold = {self.thresh}')     

    def Task2(self):
        dummy_input = torch.randn(1, 3, 256, 128, device="cuda")
        torch.onnx.export(self.extractor.model, dummy_input, "./models/osnet.onnx", verbose=False)

    def Task3(self):
        outer_label = self.generate_visuals(0.6)
        self.visualise_cluster(outer_label, n_clusters=5, n_beads=5)  

    def cluster(self) -> list[list[int]]:
        '''
        This method identifies the number of unique persons based on the cosine similarity of the features

        INPUT:

        outputs: these contains the feature maps  of all the images provided from the DL model
        bs: batch size
        n_batches: number of batches
        thresh: this defines how strong the matching should be (as you increase it the number of unique person counts changes)

        OUTPUT:

        returns the list containing the similar persons
        '''
        print('\t\t\t--------Running the clustering algo, please wait----------')
        outputs = self.features.reshape(self.bs*self.n_batches, 512)

        outer = []
        master = list(range(self.bs*self.n_batches))

        t1 = perf_counter()
        while(len(master)>0):
            key = master.pop(0)
            inner = [key]

            for i in master:
                if utils.cosine(outputs[key, :], outputs[i, :]) > self.thresh:
                    inner.append(i)
                    master.remove(i)
            outer.append(inner)

        t2 = perf_counter()

        print(f"Took {t2-t1} secs to run the algo")

        return outer

    def run_inference_batches(self, imgs_batches) -> tuple[np.ndarray, torch.Tensor]:
        '''
        This method helps in running the inference by batches to save time
        '''

        # Registering the Hook to save activation maps
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        self.extractor.model.conv5.register_forward_hook(get_activation('conv5'))
        n_batches, bs = imgs_batches.shape
        feature_maps = np.ones((n_batches, bs,512), dtype=np.float32)
        activations = torch.ones((n_batches, bs, 128, 16, 8), device=torch.device('cuda'))

        for i in tqdm(range(n_batches), "Running Inference"):
            output = self.extractor(imgs_batches[i, :].tolist())
            feature_maps[i, :, :] = output.detach().cpu().numpy()
            activations[i, ...] = activation['conv5']

        return (feature_maps, activations)
    
    def post_process_am(self, ams) -> np.ndarray:
        outputs = (ams**2).sum(1)
        b, h, w = outputs.shape
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        ams = np.ones((b, 128, 64, 3))

        for batch in range(b):

            am = outputs[batch, ...].detach().cpu().numpy()
            am = cv2.resize(am, (64, 128))
            am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            ams[batch, ...] = cv2.applyColorMap(am, cv2.COLORMAP_JET)

        return ams
   
    def generate_visuals(self, heatmap_weight: float):
        '''
        This Function helps in creating the activation maps upon the image for visualization.
        This also creates the cluster folders
        '''

        #Generate the folder for saving visuals
        shutil.rmtree('./Inference/results/',  ignore_errors=True)
        os.makedirs('./Inference/results/')

        #post processing of the am obtained from inference
        nb, bs, w, h, f = self.ams.shape
        ams = self.ams.reshape(nb*bs, w, h, f)
        ams = self.post_process_am(ams)

        regex = r"\d+"
        outer_list_names = utils.map_outer(self.cluster_idx, self.imgs_list)
        outer_label = []
        
        for i, (idx_batch, img_batch) in tqdm(enumerate(zip(self.cluster_idx, outer_list_names)), "genearting clusters and activations maps"):
            os.makedirs(f'./Inference/results/{i}')

            for idx, img_name in zip(idx_batch, img_batch):
                im = cv2.imread(img_name)
                reg = re.search(regex, img_name)
                name = img_name[reg.start(): reg.end()]
                output = (1-heatmap_weight)*im + heatmap_weight*ams[idx]
                cv2.imwrite(f'./Inference/results/{i}/{name}.jpg', output)
                outer_label.append(i)
        
        print("\t\t\t ---------------Check the output folder in the root directory for visuals------------")

        return outer_label
    
    def visualise_cluster(self, outer_label, n_clusters, n_beads):
        print("----------Creating the Clusters visualization----------")
        data = self.features.reshape(self.bs*self.n_batches, 512)
        outer_label_arr = np.array(outer_label, dtype=np.int16)
        list_collect = []

        #Trying to collect the inner list with entries greater than n_beads
        for i, inner in enumerate(self.cluster_idx):
            if (len(inner)>=n_beads) and len(list_collect)<n_clusters:
                list_collect.append(inner)
        
        assert len(list_collect) == n_clusters, "Reduce the number of beads to generate the required clusters" 


        data_sub = np.ones((2, n_clusters*n_beads))
        outer_label_arr_sub = np.ones((n_clusters*n_beads,))

        model = PCA(n_components=2)
        model.fit(data.T)

        idx = 0
        for i, inner in enumerate(list_collect):
            for j, val in enumerate(inner[:n_beads]):
                data_sub[:, idx] = model.components_[:, val]
                outer_label_arr_sub[idx] = i
                idx += 1
                
        plot_data = np.vstack((data_sub, outer_label_arr_sub.reshape(1,-1))).T
        df = pd.DataFrame(data = plot_data,
        columns =("Dim_1", "Dim_2", "label"))

        sn.FacetGrid(df, hue ="label").map(
        plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    
        plt.savefig('./Inference/cluster.png')
        print("saved the cluster figure in the Inference folder")