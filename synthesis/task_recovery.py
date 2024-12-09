import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSynthesis
from ._utils import ImagePool2


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )

    def remove(self):
        self.hook.remove()

class Synthesizer(BaseSynthesis):
    def __init__(self, args,teacher, student, generator, nz, num_classes, img_size,
                 iterations=200, lr_g=0.1,
                 synthesis_batch_size=128,
                  oh=1,adv=1, bn=1, num_teacher=4,
                 save_dir='./datapoolkd/',transform=None,transform_no_toTensor=None,
                 device='cpu', c_abs_list=None,max_batch_per_class=20):
        super(Synthesizer, self).__init__(teacher, student)
        self.args=args
        self.save_dir = save_dir
        self.loss_fn = nn.CrossEntropyLoss()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.oh = oh
        self.adv = adv
        self.bn = bn
        self.num_teacher = num_teacher
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.max_batch_per_class = max_batch_per_class
        self.c_abs_list=c_abs_list
        self.transform = transform
        self.transforms = [1,1,1,1]
        self.generator = generator.to(device).train()
        self.device = device
        self.ep = 0
        self.transform_no_toTensor=transform_no_toTensor
        self.transform_no_toTensors = [1,1,1,1]
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.data_pool = ImagePool2(args=self.args, root=self.save_dir, num_classes=self.num_classes,
                                    transform=self.transform, max_batch_per_class=max_batch_per_class)

    def synthesize_pre(self, targets=None,c_num=5,node_id=None):
        self.synthesis_batch_size_per_class =len(targets)//c_num
        ########################
        start = time.time()

        targets = torch.LongTensor(targets).to(self.device)
        reset_model(self.generator)
        self.teacher.eval()
        hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m, 0))

        for batch in range(self.max_batch_per_class):
            z = torch.randn(size=(len(targets), self.nz), device=self.device).requires_grad_()
            best_cost = 1e6
            best_inputs = self.generator(z).data
            optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                         betas=[0.5, 0.999])

            for it in range(self.iterations):
                inputs = self.generator(z)
                inputs_change = self.transform_no_toTensor(inputs)
                #############################################
                #Loss
                #############################################
                t_out = self.teacher(inputs_change)
                loss_oh = F.cross_entropy( t_out, targets )
                loss_bn = sum([h.r_feature for h in hooks])
                loss = self.oh * loss_oh + self.bn * loss_bn

                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.detach()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.data_pool.add_pre(imgs=best_inputs, c_abs_list=self.c_abs_list,node_id=node_id,
                                   targets=targets)
        for hook in hooks:
            hook.remove()
        end = time.time()
        return end - start

    def synthesize(self, targets=None,c_num=5):
        self.synthesis_batch_size =len(targets)//c_num
        ########################
        start = time.time()
        targets = torch.LongTensor(targets).to(self.device)

        best_inputs = []
        for id in range(self.num_teacher):
            self.teacher[id].eval()
            hooks = []
            for m in self.teacher[id].modules():
                if isinstance(m, nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(m, 0))
            reset_model(self.generator)
            z = torch.randn(size=(len(targets), self.nz), device=self.device).requires_grad_()
            best_cost = 1e6
            best_input = self.generator(z).data
            optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                         betas=[0.5, 0.999])
            for it in range(self.iterations):
                inputs = self.generator(z)
                if self.args.dataset == 'mix':
                    inputs_change = self.transform_no_toTensors[id](inputs)
                else:
                    inputs_change = self.transform_no_toTensor(inputs)
                #############################################
                #Loss
                #############################################
                t_out = self.teacher[id](inputs_change)
                loss_oh = F.cross_entropy( t_out, targets )
                loss_bn = sum([h.r_feature for h in hooks])
                loss = self.oh * loss_oh + self.bn * loss_bn

                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_input = inputs.detach()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            best_inputs.append(best_input)
            self.data_pool.add(imgs=best_input, c_abs_list=self.c_abs_list[id],
                               synthesis_batch_size_per_class=self.synthesis_batch_size)
        for hook in hooks:
            hook.remove()
        end = time.time()

        return best_inputs, end - start

    def get_random_task(self, num_w=5, num_s=5, num_q=15):
        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q)

    def synthesize_abl(self, targets=None,c_num=5,node_id=None):
        self.synthesis_batch_size_per_class =len(targets)//c_num
        ########################
        start = time.time()

        targets = torch.LongTensor(targets).to(self.device)
        reset_model(self.generator)
        self.teacher.eval()
        hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m, 0))

        for batch in range(self.max_batch_per_class):
            z = torch.randn(size=(len(targets), self.nz), device=self.device).requires_grad_()
            best_cost = 1e6
            best_inputs = self.generator(z).data
            optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                         betas=[0.5, 0.999])

            for it in range(self.iterations):
                inputs = self.generator(z)
                inputs_change = self.transform_no_toTensor(inputs)
                #############################################
                #Loss
                #############################################
                t_out = self.teacher(inputs_change)
                loss_oh = F.cross_entropy( t_out, targets )
                loss_bn = sum([h.r_feature for h in hooks])
                loss = self.oh * loss_oh + self.bn * loss_bn

                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.detach()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.data_pool.add_abl(imgs=best_inputs, c_abs_list=self.c_abs_list,node_id=node_id,
                                   targets=targets)
        for hook in hooks:
            hook.remove()
        end = time.time()
        return end - start
