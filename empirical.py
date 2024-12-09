import argparse
import os
import random
import torch
from torch import nn
from tools import get_model, get_premodel, get_transform, \
    get_transform_no_toTensor, \
    label_abs2relative, get_dataloader, data2supportquery, Timer, setup_seed, compute_confidence_interval, Generator, \
    pretrains, pretrain
from torch.utils.data import DataLoader
from methods.maml import Maml, MamlKD
from synthesis.task_recovery import Synthesizer
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

parser = argparse.ArgumentParser(description='DFML')
#basic
parser.add_argument('--multigpu', type=str, default='1', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='cifar100', help='test dataset')
parser.add_argument('--pretrained_path_prefix', type=str, default='./pretrained_model', help='user-defined')
#meta-learning
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4',help='architecture of the meta model')
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
parser.add_argument('--episode_batch',type=int, default=16)
parser.add_argument('--epoch',type=int, default=50)
#data free
parser.add_argument('--num_recover', type=int, default=30)
parser.add_argument('--oh', type=float, default=1.0)
parser.add_argument('--adv', type=float, default=0.0)
parser.add_argument('--bn', type=float, default=1.0)
parser.add_argument('--way_pretrain', type=int, default=5, help='way')
parser.add_argument('--pre_model_num', type=int, default=100)
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet10/resnet18')
parser.add_argument('--pretrain', action='store_true',default=False)
parser.add_argument('--generate_interval', type=int, default=4)
parser.add_argument('--generate_iterations', type=int, default=200)
parser.add_argument('--Glr', type=float, default=0.001)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
setup_seed(42)

########################################################################################################################
if args.dataset=='cifar100':
    img_size = 32
    args.img_size =32
    channel = 3
    args.channel=3
    class_num = 64
    args.class_num=64
model_maml = get_model(args,'train')
device=torch.device('cuda:{}'.format(args.gpu))
_, _, test_loader = get_dataloader(args)
criteria = nn.CrossEntropyLoss()
maml = Maml(args)
mamlkd = MamlKD(args)
########################################################################################################################
nz = 256
generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=channel).cuda()
transform=get_transform(args)
transform_no_toTensor=get_transform_no_toTensor(args)
max_batch_per_class = 20
synthesizer = Synthesizer(args, None, None, generator,
                          nz=nz, num_classes=class_num,
                          img_size=(channel, img_size, img_size),
                          iterations=args.generate_iterations, lr_g=args.Glr,
                          synthesis_batch_size=30,
                          oh=args.oh, adv=args.adv, bn=args.bn,
                          save_dir='./datapoolkd/',
                          transform=transform, transform_no_toTensor=transform_no_toTensor,
                          device=args.gpu, c_abs_list=None, max_batch_per_class=max_batch_per_class)
########################################################################################################################
## pretrain models and synthesize training data
def generate_specific_lists(specific, total_classes, num_lists_per_overlap=10):
    all_classes = set(range(total_classes))
    specific_set = set(specific)
    flattened_lists = []
    flattened_lists.append(specific)

    for overlap in range(len(specific) + 1):
        for _ in range(num_lists_per_overlap):
            overlapping_classes = set(random.sample(specific_set, overlap))
            non_overlapping_classes = set(random.sample(all_classes - specific_set, len(specific) - overlap))
            specific_ = list(overlapping_classes | non_overlapping_classes)
            flattened_lists.append(specific_)
    return flattened_lists

def calculate_overlap(classes1, classes2):
    return len(set(classes1) & set(classes2)) / float(len(set(classes1) | set(classes2)))

pretrained_path = './pretrained_model/cifar100/conv4/abl'
os.makedirs(pretrained_path, exist_ok=True)
specific = [0, 1, 2, 3, 4]
total_classes = 10
flattened_lists = generate_specific_lists(specific, total_classes)
# backbone = ['conv4'] + ['conv4','resnet10','resnet18','resnet50'] * 10
# num_node = len(backbone)
num_node = len(flattened_lists)

for i in range(num_node):
    setup_seed(222 + i)
    specific=flattened_lists[i]
    print(specific)
    # args.pre_backbone = backbone[i]
    teacher_param_specific,acc=pretrain(args, specific, device)
    print('teacher{}_acc:'.format(i),acc)
    torch.save({'teacher':teacher_param_specific,'specific':specific},
               os.path.join(pretrained_path,'model_specific_{}.pth'.format(i)))
    teacher = get_premodel(args).cuda(device)
    teacher.load_state_dict(teacher_param_specific)
    synthesizer.teacher = teacher
    synthesizer.c_abs_list = specific
    cost = synthesizer.synthesize_abl(
        targets=torch.LongTensor((list(range(len(specific)))) * args.num_recover), c_num=len(specific), node_id=i)
print('pretrain end!')
########################################################################################################################
model_classes = []
for node_id in range(num_node):
    teacher_param_specific=torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
    specific=teacher_param_specific['specific']
    model_classes.append(specific)
# overlapping classes
overlap_matrix = np.zeros((num_node, num_node))
for i in range(num_node):
    for j in range(num_node):
        overlap_matrix[i, j] = calculate_overlap(model_classes[i], model_classes[j])
########################################################################################################################
class CustomDataset(Dataset):
    def __init__(self, root_dir, num_samples_per_class=20, transform=None):
        self.root_dir = root_dir
        self.num_samples_per_class = num_samples_per_class
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for class_id, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                all_images = os.listdir(class_dir)
                if len(all_images) >= num_samples_per_class:
                    sampled_images = random.sample(all_images, num_samples_per_class)
                else:
                    sampled_images = all_images
                for img_name in sampled_images:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def custom_collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    return torch.stack(batch_images), torch.tensor(batch_labels)
########################################################################################################################
## basic model
node_id = 0
# args.pre_backbone = backbone[node_id]
teacher = get_premodel(args).cuda(device)
teacher_param_specific = torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
teacher.load_state_dict(teacher_param_specific['teacher'])
specific = teacher_param_specific['specific']
dataset_root = os.path.join(pretrained_path, str(node_id))
dataset = CustomDataset(root_dir=dataset_root, transform=transform)
data_loader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
#######################################################################################################################
def compute_accuracy(model):
    acc_val = []
    for test_batch in test_loader:
        data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
        support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
        _, acc = maml.run(model_maml=model, support=support, support_label=support_label_relative,
                                      query=query, query_label=query_label_relative, criteria=criteria, device=device,
                                      mode='test')
        acc_val.append(acc)
    acc, pm = compute_confidence_interval(acc_val)
    return acc

accuracy_list = []
for node_id in range(num_node):
    setup_seed(42)
    model = get_model(args, 'train')
    model.cuda(device)
    optimizer = optim.Adam(model.parameters(), lr=args.outer_lr)
    dataset_new = CustomDataset(root_dir=os.path.join(pretrained_path, str(node_id)), transform=transform)
    data_loader_new = DataLoader(dataset_new, batch_size=100, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
    for epoch in range(args.epoch):
        for (data1, target1), (data2, target2) in zip(data_loader,data_loader_new):
            loss1, train_acc = mamlkd.run_maml(model_maml=model,query=data1,query_label=target1,device=device,mode='train')
            loss2, train_acc = mamlkd.run_maml(model_maml=model,query=data2,query_label=target2,device=device,mode='train')
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    accuracy = compute_accuracy(model)
    accuracy_list.append(accuracy)
    print(f"Model {node_id} Accuracy: {accuracy}%")
accuracy_diffs = [accuracy - accuracy_list[0] for accuracy in accuracy_list[1:]]
########################################################################################################################