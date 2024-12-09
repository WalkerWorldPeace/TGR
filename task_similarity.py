import os
import numpy as np
import pickle
from scipy.spatial import distance
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import torchvision.models as models
from torchvision.models import ResNet34_Weights

def fisher(model, data_loader, device):
    loss_fn = nn.CrossEntropyLoss()
    for p in model.parameters():
        p.grad2_acc = torch.zeros_like(p, device=device)
        p.grad_counter = 0

    for data, target in tqdm(data_loader, leave=False, desc="Computing Fisher"):
        data = data.to(device)
        output = model(data)
        target = target.to(device)
        loss = loss_fn(output, target)
        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad2_acc += p.grad.data ** 2
                p.grad_counter += 1

    FIM = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            FIM[name] = parameter.grad2_acc / parameter.grad_counter
            del parameter.grad2_acc, parameter.grad_counter
    return FIM

class ModelSimilarityCalculator:
    def __init__(self, num_premodels, fim_directory):
        self.num_premodels = num_premodels
        self.fim_directory = fim_directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(self.device)

    def compute_fim_and_save(self, data_loader, node_id):
        FIM = fisher(self.model, data_loader, self.device)
        fim_file_path = os.path.join(self.fim_directory, f'FIM_teacher_{node_id}.pkl')
        with open(fim_file_path, 'wb') as fp:
            pickle.dump(FIM, fp)

    def cosine_similarity(self, e0, e1):
        e0 = e0.reshape(-1).cpu()
        e1 = e1.reshape(-1).cpu()
        e0, e1 = e0 / (e0 + e1 + 1e-6), e1 / (e0 + e1 + 1e-6)
        return 1 - distance.cosine(e0, e1)

    def load_fims_and_compute_similarity_matrix(self,fim_directory_1=None, fim_directory_2=None, fim_directory_3=None):
        FIMs = []
        if fim_directory_1 is None:
            for node_id in range(self.num_premodels):
                with open(os.path.join(self.fim_directory, f'FIM_teacher_{node_id}.pkl'), 'rb') as fp:
                    FIMs.append(pickle.load(fp))
        else:
            for node_id in range(self.num_premodels):
                with open(os.path.join(fim_directory_1, f'FIM_teacher_{node_id}.pkl'), 'rb') as fp:
                    FIM = pickle.load(fp)
                    if isinstance(FIM, dict):
                        for key, value in FIM.items():
                            if isinstance(value, torch.Tensor):
                                FIM[key] = value.cpu()
                    elif isinstance(FIM, torch.Tensor):
                        FIM = FIM.cpu()
                    FIMs.append(FIM)
            for node_id in range(self.num_premodels):
                with open(os.path.join(fim_directory_2, f'FIM_teacher_{node_id}.pkl'), 'rb') as fp:
                    FIM = pickle.load(fp)
                    if isinstance(FIM, dict):
                        for key, value in FIM.items():
                            if isinstance(value, torch.Tensor):
                                FIM[key] = value.cpu()
                    elif isinstance(FIM, torch.Tensor):
                        FIM = FIM.cpu()
                    FIMs.append(FIM)
            for node_id in range(self.num_premodels):
                with open(os.path.join(fim_directory_3, f'FIM_teacher_{node_id}.pkl'), 'rb') as fp:
                    FIM = pickle.load(fp)
                    if isinstance(FIM, dict):
                        for key, value in FIM.items():
                            if isinstance(value, torch.Tensor):
                                FIM[key] = value.cpu()
                    elif isinstance(FIM, torch.Tensor):
                        FIM = FIM.cpu()
                    FIMs.append(FIM)
        sim_matrix = np.zeros((len(FIMs), len(FIMs)))
        for i, target_task in enumerate(FIMs):
            rank = []
            for k in target_task:
                if np.inf in target_task:
                    continue
                cosine_sim = []
                flag = True
                for emb in FIMs[i + 1:]:
                    cosine_sim.append(self.cosine_similarity(target_task[k], emb[k]))
                if flag:
                    component_rank = sorted(enumerate(cosine_sim), key=lambda x: -x[1])
                    rank.append(component_rank)
            task_sum = np.zeros(len(FIMs) - i - 1)
            for r in rank:
                for tmp in r:
                    task_sum[tmp[0]] += tmp[1]
            task_sum = task_sum / len(rank)
            sim_matrix[i, i + 1:] = task_sum

        for i in range(len(sim_matrix)):
            for j in range(i):
                sim_matrix[i, j] = sim_matrix[j, i]
            sim_matrix[i, i] = 1
        return np.around(sim_matrix, decimals=4)

    def get_similar_models(self, fim_directory_1=None, fim_directory_2=None, fim_directory_3=None):
        if fim_directory_1 is None:
            sim_matrix = self.load_fims_and_compute_similarity_matrix()
        else:
            sim_matrix = self.load_fims_and_compute_similarity_matrix(fim_directory_1, fim_directory_2, fim_directory_3)
        df_sim_matrix = pd.DataFrame(sim_matrix)
        df_sim_matrix.to_csv(os.path.join(self.fim_directory, 'similarity_matrix.csv'), index=False)

        return sim_matrix
