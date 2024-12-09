import torch
import torch.nn.functional as F
import torch.nn as nn

class Maml():
    def __init__(self,args):
        self.args=args
    def run(self,model_maml,support,support_label,query,query_label,criteria,device,mode='train'):
        if mode=='train':
            model_maml.train()
            inner_update_num=self.args.inner_update_num
        else:
            model_maml.train()
            inner_update_num = self.args.test_inner_update_num

        # inner
        fast_parameters = list(model_maml.parameters())
        for weight in model_maml.parameters():
            weight.fast = None
        model_maml.zero_grad()
        correct, total = 0, 0
        for inner_step in range(inner_update_num):
            pred = model_maml(support)
            loss_inner = criteria(pred, support_label)
            if mode=='train':
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
            else:
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=False)
            if self.args.approx == True:
                grad = [g.detach() for g in grad]
            fast_parameters = []
            for k, weight in enumerate(model_maml.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.args.inner_lr * grad[k]
                else:
                    weight.fast = weight.fast - self.args.inner_lr * grad[k]
                fast_parameters.append(weight.fast)
        # outer
        score = model_maml(query)
        prediction = torch.max(score, 1)[1]
        # if mode=='test':
        #     print('test inference time per task',inf_time.measure())
        correct = correct + (prediction.cpu() == query_label.cpu()).sum()
        total = total + len(query_label)
        loss_outer = criteria(score, query_label)
        acc=1.0*correct/total*100.0
        for weight in model_maml.parameters():
            weight.fast = None
        return loss_outer,acc


class MamlKD():
    def __init__(self, args):
        self.args = args
        self.celoss = nn.CrossEntropyLoss()

    def run_inner(self, model_maml, device, displacement, mode='train'):
        if mode == 'train':
            model_maml.train()
        else:
            model_maml.train()
        # inner
        for weight in model_maml.parameters():
            weight.fast = None
        model_maml.zero_grad()

        for k, weight in enumerate(model_maml.parameters()):
            if weight.fast is None:
                weight.fast = weight - self.args.displace_lr * displacement[k]
            else:
                weight.fast = weight.fast - self.args.displace_lr * displacement[k]
        return

    def run_outer(self, model_maml, query,query_label, criteria, device, teacher, mode='train'):
        if mode == 'train':
            model_maml.train()
        else:
            model_maml.train()
        teacher.eval()
        # outer
        correct,total=0,0
        s_logits = model_maml(query)
        with torch.no_grad():
            t_logits = teacher(query)
        loss_outer = criteria(F.softmax(s_logits, dim=-1), F.softmax(t_logits.detach(), dim=-1))
        prediction = torch.max(s_logits, 1)[1]
        correct = correct + (prediction.cpu() == query_label.cpu()).sum()
        total = total + len(query_label)
        acc = 1.0*correct/total*100.0
        loss_gt = self.args.gt_loss_weight * self.celoss(s_logits, query_label)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_outer,
        }
        return losses_dict, acc

    def run(self,model_maml,support,support_label,query,query_label,criteria,mode='train'):
        if mode=='train':
            model_maml.train()
            inner_update_num=self.args.inner_update_num
        else:
            model_maml.train()
            inner_update_num = self.args.test_inner_update_num

        # inner
        fast_parameters = list(model_maml.parameters())
        for weight in model_maml.parameters():
            weight.fast = None
        model_maml.zero_grad()
        correct, total = 0, 0
        for inner_step in range(inner_update_num):
            pred = model_maml(support)
            loss_inner = criteria(pred, support_label)
            if mode=='train':
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
            else:
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=False)
            if self.args.approx == True:
                grad = [g.detach() for g in grad]
            fast_parameters = []
            for k, weight in enumerate(model_maml.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.args.inner_lr * grad[k]
                else:
                    weight.fast = weight.fast - self.args.inner_lr * grad[k]
                fast_parameters.append(weight.fast)
        # outer
        score = model_maml(query)
        prediction = torch.max(score, 1)[1]
        correct = correct + (prediction.cpu() == query_label.cpu()).sum()
        total = total + len(query_label)
        loss_outer = criteria(score, query_label)
        acc=1.0*correct/total*100.0
        for weight in model_maml.parameters():
            weight.fast = None
        return loss_outer,acc

    def split_support_query(self, query, query_label, num_support=5):
        classes = torch.unique(query_label)
        support_indices = []
        query_indices = []

        for c in classes:
            indices = torch.where(query_label == c)[0]
            support_indices.append(indices[:num_support])
            query_indices.append(indices[num_support:])

        support_indices = torch.cat(support_indices)
        query_indices = torch.cat(query_indices)

        support = query[support_indices]
        support_label = query_label[support_indices]
        new_query = query[query_indices]
        new_query_label = query_label[query_indices]

        return support, support_label, new_query, new_query_label

    def run_maml(self, model_maml, query, query_label, device, mode='train'):
        model_maml.train()
        # MAML
        support, support_label, new_query, new_query_label = self.split_support_query(query, query_label)
        support, support_label = support.to(device), support_label.to(device)
        new_query, new_query_label = new_query.to(device), new_query_label.to(device)
        loss_maml, acc = self.run(model_maml, support, support_label, new_query, new_query_label, nn.CrossEntropyLoss(),
                                  mode)
        return loss_maml, acc

