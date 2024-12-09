import argparse
import os
import random
import shutil
import torch.nn.functional as F
import torch
from torch import nn
from tools import get_model, get_premodel, get_transform, \
    get_transform_no_toTensor, \
    label_abs2relative, get_dataloader, data2supportquery, Timer, setup_seed, compute_confidence_interval, Generator, \
    pretrains, average_cosine_similarity
from methods.maml import Maml, MamlKD
from synthesis.task_recovery import Synthesizer
import time
from sklearn.cluster import SpectralClustering
import logging
import pickle
import numpy as np
from task_similarity import ModelSimilarityCalculator
from torch.autograd import Variable
import wandb
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd

parser = argparse.ArgumentParser(description='DFML')
#basic
parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='miniimagenet', help='cifar100/miniimagenet/cub/mix')
parser.add_argument('--pretrained_path_prefix', type=str, default='./pretrained_model', help='user-defined')
#meta-learning
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4',help='architecture of the meta model')
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
parser.add_argument('--episode_batch',type=int, default=16)
parser.add_argument('--val_interval',type=int, default=20)
#task grouping
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--way_pretrain', type=int, default=5, help='way')
parser.add_argument('--pre_model_num', type=int, default=100)
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet10/resnet18/mix')
parser.add_argument('--pretrain', action='store_true',default=False)
#task regularization
parser.add_argument('--gt_loss_weight', type=float, default=1.0)
parser.add_argument('--kd_loss_weight', type=float, default=1.0)
parser.add_argument('--displace_lr', type=float, default=0.001)
parser.add_argument('--num_teacher', type=int, default=4)
#data free
parser.add_argument('--num_recover', type=int, default=30)
parser.add_argument('--oh', type=float, default=1.0)
parser.add_argument('--adv', type=float, default=0.0)
parser.add_argument('--bn', type=float, default=1.0)
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
elif args.dataset == 'miniimagenet':
    img_size = 84
    args.img_size = 84
    channel = 3
    args.channel = 3
    class_num = 64
    args.class_num = 64
elif args.dataset=='cub':
    img_size = 84
    args.img_size=84
    channel = 3
    args.channel=3
    class_num = 100
    args.class_num = 100
elif args.dataset == 'mix':
    img_size = 84
    args.img_size = 84
    channel = 3
    args.channel = 3
    class_num = 228
    args.class_num = None
########################################################################################################################
if args.dataset == 'mix':
    model_maml=get_model(args=args,set_maml_value=True,arbitrary_input=True)
else:
    model_maml=get_model(args,'train')
device = torch.device('cuda:{}'.format(args.gpu))
model_maml.cuda(device)
if args.dataset!='mix':
    _, _, test_loader = get_dataloader(args)
elif args.dataset=='mix':
    test_loader_cifar, test_loader_mini, test_loader_cub=get_dataloader(args)
if args.dataset != 'mix':
    pretrained_path=os.path.join(args.pretrained_path_prefix,'{}/{}/{}/{}way/model'
                                 .format(args.dataset,args.pre_backbone,'meta_train', args.way_pretrain))
    os.makedirs(pretrained_path, exist_ok=True)
criteria = nn.CrossEntropyLoss()
maml = Maml(args)
mamlkd = MamlKD(args)
optimizer = torch.optim.Adam(params=[{'params': model_maml.parameters(), 'lr': args.outer_lr}])
########################################################################################################################
timer = Timer()
feature='{}_{}clusters_{}displace_{}teacher_{}shot_{}numkd_{}prebackbone_{}gpu'.\
    format(args.dataset,args.n_clusters,args.displace_lr,args.pre_model_num,args.num_sup_test,args.num_recover,args.pre_backbone,args.multigpu)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_path = os.path.join('log', '{}.log'.format(feature))
handler = logging.FileHandler(log_file_path)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
########################################################################################################################
if args.pretrain:
    pretrains(args,args.pre_model_num,device,pretrained_path)
    print('pretrain end!')
    raise NotImplementedError
########################################################################################################################
nz = 256
generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=channel).cuda()
transform = get_transform(args)
transform_no_toTensor = get_transform_no_toTensor(args)
if os.path.exists('./datapoolkd/' + feature):
    shutil.rmtree('./datapoolkd/' + feature)
    print('remove')
os.makedirs('./datapoolkd/' + feature,exist_ok=True)
max_batch_per_class = 20
synthesizer = Synthesizer(args, None, None, generator,
                          nz=nz, num_classes=class_num,
                          img_size=(channel, img_size, img_size),
                          iterations=args.generate_iterations, lr_g=args.Glr,
                          synthesis_batch_size=30,
                          oh=args.oh, adv=args.adv, bn=args.bn,
                          save_dir='./datapoolkd/' + feature,
                          transform=transform, transform_no_toTensor=transform_no_toTensor,
                          device=args.gpu, c_abs_list=None, max_batch_per_class=max_batch_per_class)
########################################################################################################################
def compute_similar_models(model_root, num_models):
    if args.dataset != 'mix':
        if args.pre_backbone != 'mix':
            fim_directory = os.path.join(model_root, 'FIMs')
            if not os.path.exists(fim_directory):
                os.makedirs(fim_directory)

            filename = os.path.join(fim_directory, 'similarity_matrix.csv')
            if os.path.exists(filename):
                print(f"File {filename} exists, loading the data.")
                df = pd.read_csv(filename)
                sim_matrix = df.values.tolist()
            else:
                print(f"File {filename} does not exist, generating new data.")
                similarity_calculator = ModelSimilarityCalculator(num_models, fim_directory)

                for node_id in range(num_models):
                    fim_file_path = os.path.join(fim_directory, f'FIM_teacher_{node_id}.pkl')
                    if not os.path.exists(fim_file_path):
                        teacher = get_premodel(args).cuda(device)
                        teacher_param_specific = torch.load(
                            os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
                        teacher.load_state_dict(teacher_param_specific['teacher'])
                        specific = teacher_param_specific['specific']
                        synthesizer.teacher = teacher
                        synthesizer.c_abs_list = specific
                        if not os.path.exists(os.path.join(model_root, str(node_id))):
                            cost = synthesizer.synthesize_pre(
                                targets=torch.LongTensor((list(range(len(specific)))) * args.num_recover), c_num=len(specific),
                                node_id=node_id)
                        dataset = ImageFolder(root=os.path.join(model_root, str(node_id)), transform=transform)
                        data_loader = DataLoader(dataset, batch_size=600, shuffle=True)
                        similarity_calculator.compute_fim_and_save(data_loader, node_id)
                sim_matrix = similarity_calculator.get_similar_models()

        elif args.pre_backbone == 'mix':
            model_root_conv4 = os.path.join('./pretrained_model/{}/{}'.format(args.dataset, 'conv4'), 'inversion')
            model_root_resnet10 = os.path.join('./pretrained_model/{}/{}'.format(args.dataset, 'resnet10'), 'inversion')
            model_root_resnet18 = os.path.join('./pretrained_model/{}/{}'.format(args.dataset, 'resnet18'), 'inversion')
            fim_directory_conv4 = os.path.join(model_root_conv4, 'FIMs')
            fim_directory_resnet10 = os.path.join(model_root_resnet10, 'FIMs')
            fim_directory_resnet18 = os.path.join(model_root_resnet18, 'FIMs')

            filename = os.path.join(model_root, 'similarity_matrix.csv')
            if os.path.exists(filename):
                print(f"File {filename} exists, loading the data.")
                df = pd.read_csv(filename)
                sim_matrix = df.values.tolist()
            else:
                if not os.path.exists(model_root):
                    os.makedirs(model_root)
                print(f"File {filename} does not exist, generating new data.")
                similarity_calculator = ModelSimilarityCalculator(num_models, model_root)
                sim_matrix = similarity_calculator.get_similar_models(fim_directory_conv4, fim_directory_resnet10,
                                                                      fim_directory_resnet18)
    elif args.dataset == 'mix':
        model_root_cifar = os.path.join('./pretrained_model/{}/{}'.format('cifar100', args.pre_backbone), 'inversion')
        model_root_mini = os.path.join('./pretrained_model/{}/{}'.format('miniimagenet', args.pre_backbone),'inversion')
        model_root_cub = os.path.join('./pretrained_model/{}/{}'.format('cub', args.pre_backbone), 'inversion')
        fim_directory_cifar = os.path.join(model_root_cifar, 'FIMs')
        fim_directory_mini = os.path.join(model_root_mini, 'FIMs')
        fim_directory_cub = os.path.join(model_root_cub, 'FIMs')

        filename = os.path.join(model_root, 'similarity_matrix.csv')
        if os.path.exists(filename):
            print(f"File {filename} exists, loading the data.")
            df = pd.read_csv(filename)
            sim_matrix = df.values.tolist()
        else:
            if not os.path.exists(model_root):
                os.makedirs(model_root)
            print(f"File {filename} does not exist, generating new data.")
            similarity_calculator = ModelSimilarityCalculator(num_models, model_root)
            sim_matrix = similarity_calculator.get_similar_models(fim_directory_cifar, fim_directory_mini, fim_directory_cub)

    return np.array(sim_matrix)[:num_models,:num_models]

model_root = os.path.join('./pretrained_model/{}/{}'.format(args.dataset,args.pre_backbone),'inversion')
sim_matrix = compute_similar_models(model_root, args.pre_model_num)
ones_matrix = np.ones(sim_matrix.shape)
difference_matrix = ones_matrix - sim_matrix
clustering = SpectralClustering(n_clusters=args.n_clusters, affinity='precomputed', n_init=30, random_state=42)
labels = clustering.fit_predict(difference_matrix)
print(labels)
pretrained_model_groups = [[] for _ in range(args.n_clusters)]
for model_id, cluster_id in enumerate(labels):
    pretrained_model_groups[cluster_id].append(model_id)
print(pretrained_model_groups)
########################################################################################################################
maxAcc=None
max_acc_val=-1
max_acc_val_all=[-1,-1,-1]
max_it_all=[-1,-1,-1]
max_pm_all=[-1,-1,-1]
loss_batch, acc_batch = [], []
generate_num = 0
time_cost = 0

for epoch in range(args.start_id, args.epochs + 1):
    teachers = []
    specifics = []
    transform_no_toTensors = []
    # sample pre-trained models
    if args.dataset !='mix':
        if args.pre_backbone == 'mix':
            selected_group = random.choice(pretrained_model_groups)
            for id in range(args.num_teacher):
                node_id = random.choice(selected_group)
                if node_id < args.pre_model_num:
                    random_pretrain = 'conv4'
                elif node_id < args.pre_model_num * 2:
                    random_pretrain = 'resnet10'
                    node_id -= args.pre_model_num
                elif node_id < args.pre_model_num * 3:
                    random_pretrain = 'resnet18'
                    node_id -= args.pre_model_num * 2
                teacher = get_premodel(args, random_pretrain).cuda(device)
                pretrained_path = os.path.join(args.pretrained_path_prefix,'{}/{}/{}/{}way/model'.format(args.dataset,
                                                                    random_pretrain,'meta_train', args.way_pretrain))
                teacher_param_specific = torch.load(
                    os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
                teacher.load_state_dict(teacher_param_specific['teacher'])
                specific = teacher_param_specific['specific']
                teachers.append(teacher)
                specifics.append(specific)

        elif args.pre_backbone != 'mix':
            selected_group = random.choice(pretrained_model_groups)
            for id in range(args.num_teacher):
                teacher = get_premodel(args).cuda(device)
                node_id = random.choice(selected_group)
                teacher_param_specific=torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
                teacher.load_state_dict(teacher_param_specific['teacher'])
                specific=teacher_param_specific['specific']
                teachers.append(teacher)
                specifics.append(specific)
    elif args.dataset =='mix':
        selected_group = random.choice(pretrained_model_groups)
        for id in range(args.num_teacher):
            node_id = random.choice(selected_group)
            if node_id < args.pre_model_num:
                random_dataset = 'cifar100'
            elif node_id < args.pre_model_num * 2:
                random_dataset = 'miniimagenet'
                node_id -= args.pre_model_num
            elif node_id < args.pre_model_num * 3:
                random_dataset = 'cub'
                node_id -= args.pre_model_num * 2
            if random_dataset == 'cifar100':
                args.img_size = 32
            else:
                args.img_size = 84
            teacher = get_premodel(args).cuda(device)
            pretrained_path = os.path.join(args.pretrained_path_prefix,
                                           '{}/{}/{}/{}way/model'.format(random_dataset, args.pre_backbone, 'meta_train',
                                                                         args.way_pretrain))
            teacher_param_specific=torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
            teacher.load_state_dict(teacher_param_specific['teacher'])
            specific=teacher_param_specific['specific']
            if random_dataset == 'cifar100':
                pass
            elif random_dataset == 'miniimagenet':
                specific = [i + 64 for i in specific]
            elif random_dataset == 'cub':
                specific = [i + 128 for i in specific]
            synthesizer.transform = get_transform(args, dataset=random_dataset)
            synthesizer.transform_no_toTensors[id] = get_transform_no_toTensor(args, dataset=random_dataset)
            transform_no_toTensors.append(get_transform_no_toTensor(args, dataset=random_dataset))
            teachers.append(teacher)
            specifics.append(specific)

    # generate for training data
    synthesizer.teacher = teachers
    synthesizer.c_abs_list = specifics
    kd_tensors, cost = synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * args.num_recover),
                                              c_num=len(specific))

    time_cost += cost
    loss_kd = F.kl_div
    generate_num += args.num_recover * len(specific)

    if args.dataset == 'mix':
        kd_datas = torch.stack(kd_tensors, dim=0)
    elif args.dataset != 'mix':
        kd_datas = transform_no_toTensor(torch.stack(kd_tensors, dim=0))
    label_relative = torch.LongTensor((list(range(len(specific)))) * args.num_recover).cuda(device)


    full_gradient = [torch.zeros_like(p) for p in (model_maml.parameters())]
    grads = []
    for id in range(args.num_teacher):
        if args.dataset == 'mix':
            loss_outer, train_acc = mamlkd.run_outer(model_maml=model_maml,query=transform_no_toTensors[id](kd_datas[id]),
                                                     query_label=label_relative, criteria=loss_kd, device=device,
                                                     teacher=teachers[id], mode='train')
        elif args.dataset != 'mix':
            loss_outer, train_acc = mamlkd.run_outer(model_maml=model_maml, query=kd_datas[id],
                                                     query_label=label_relative, criteria=loss_kd, device=device,
                                                     teacher=teachers[id], mode='train')
        loss = sum(loss_outer.values())
        grad = torch.autograd.grad(loss, model_maml.parameters(), create_graph=True,retain_graph=True, only_inputs=True)
        grads.append(grad)
        for i, g in enumerate(grad):
            full_gradient[i] += g


    full_gradient = [g / args.num_teacher for g in full_gradient]
    gradient_adjustments = []

    # # gradient norm
    # reg_loss = 0.0
    # for grad in grads:
    #     for g, avg_g in zip(grad, full_gradient):
    #         reg_loss += (g - avg_g).norm(2).pow(2)
    # reg_loss /= (2 * args.num_teacher)
    # cosine_similarity = average_cosine_similarity(grads)
    # wandb.log({"reg_loss": reg_loss, "similarity": cosine_similarity})

    # Process each mini-batch
    for id in range(args.num_teacher):
        displacement = [(f - g) for g, f in zip(grads[id], full_gradient)]
        mamlkd.run_inner(model_maml=model_maml, device=device, displacement=displacement, mode='train')
        if args.dataset == 'mix':
            loss_outer, train_acc = mamlkd.run_outer(model_maml=model_maml,query=transform_no_toTensors[id](kd_datas[id]),
                                                     query_label=label_relative, criteria=loss_kd, device=device,
                                                     teacher=teachers[id], mode='train')
        elif args.dataset != 'mix':
            loss_outer, train_acc = mamlkd.run_outer(model_maml=model_maml, query=kd_datas[id],
                                                     query_label=label_relative, criteria=loss_kd, device=device,
                                                     teacher=teachers[id], mode='train')
        loss = sum(loss_outer.values())
        # Compute the displacement for the current mini-batch
        mini_batch_gradient = torch.autograd.grad(loss, model_maml.parameters(),
                                                  create_graph=False, retain_graph=True)
        gradient_adjustments.append(mini_batch_gradient)

    avg_mini_batch_gradient = []
    for grad in gradient_adjustments[0]:
        avg_mini_batch_gradient.append(torch.zeros_like(grad))
    for mini_batch_gradient in gradient_adjustments:
        for i, grad in enumerate(mini_batch_gradient):
            avg_mini_batch_gradient[i] += grad
    num_mini_batches = len(gradient_adjustments)
    for i in range(len(avg_mini_batch_gradient)):
        avg_mini_batch_gradient[i] /= num_mini_batches

    for param, grad in zip(model_maml.parameters(), avg_mini_batch_gradient):
        if param.grad is None:
            param.grad = Variable(torch.zeros(param.size())).cuda()
        param.grad.data = grad.data

    optimizer.step()
    optimizer.zero_grad()

    #replay
    e_count = 0
    maxAcc = None
    while e_count < args.generate_interval:

        support_data, support_label_abs, query_data, query_label_abs, specific = synthesizer.get_random_task(
            num_w=args.way_train, num_s=args.num_sup_train, num_q=args.num_qur_train)
        support_label = label_abs2relative(specific, support_label_abs).cuda()
        query_label = label_abs2relative(specific, query_label_abs).cuda()
        support, support_label, query, query_label = support_data.cuda(device), support_label.cuda(
            device), query_data.cuda(device), query_label.cuda(device)
        loss_outer, train_acc = maml.run(model_maml=model_maml, support=support, support_label=support_label,
                                         query=query, query_label=query_label, criteria=criteria, device=device,
                                         mode='train')
        loss_batch.append(loss_outer)
        acc_batch.append(train_acc)

        if len(loss_batch) and len(loss_batch) % args.episode_batch == 0:
            loss = torch.stack(loss_batch).sum(0)
            acc = torch.stack(acc_batch).mean()
            loss_batch, acc_batch = [], []
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if maxAcc == None or acc > maxAcc:
                maxAcc = acc
                e_count = 0
            else:
                e_count = e_count + 1

    #val
    if epoch % args.val_interval == 0:
        acc_val = []
        if args.dataset != 'mix':
            for test_batch in test_loader:
                data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
                _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                  query=query, query_label=query_label_relative, criteria=criteria, device=device,
                                  mode='test')
                acc_val.append(acc)
            del _

            acc_val, pm = compute_confidence_interval(acc_val)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = epoch
                max_pm = pm
        if args.dataset == 'mix':
            test_loader_all = [test_loader_cifar, test_loader_mini, test_loader_cub]
            acc_val_all = [[], [], []]
            for i, test_loader in enumerate(test_loader_all):
                for test_batch in test_loader:
                    data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                    support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
                    _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                      query=query, query_label=query_label_relative, criteria=criteria,
                                      device=device,
                                      mode='test')
                    acc_val_all[i].append(acc)
                acc_val, pm = compute_confidence_interval(acc_val_all[i])
                acc_val_all[i] = acc_val
            acc_val = sum(acc_val_all) / len(acc_val_all)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = epoch
                max_pm = pm
            for i in range(3):
                if acc_val_all[i] > max_acc_val_all[i]:
                    max_acc_val_all[i] = acc_val_all[i]
                    max_it_all[i] = epoch
                    max_pm_all[i] = pm

        logger.info('task_id:' + str(epoch) + ' test acc: ' + str(acc_val) + '+-'+ str(pm))
        logger.info(str(max_it) + ' best test acc: ' + str(max_acc_val) + '+-' + str(max_pm))
        logger.info('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((epoch) / (args.epochs))))
        logger.info("Generation Cost: %1.3f" % (time_cost / 3600.))
        print('generate:', generate_num, 'images')
        print(epoch, 'test acc:', acc_val, '+-', pm)
        print(max_it, 'best test acc:', max_acc_val, '+-', max_pm)
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((epoch) / (args.epochs)))
        )
        print("Generation Cost: %1.3f" % (time_cost / 3600.))