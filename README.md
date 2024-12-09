# [ICML 2024] Task Groupings Regularization: Data-Free Meta-Learning with Heterogeneous Pre-trained Models

## Abstract
Data-Free Meta-Learning (DFML) aims to derive knowledge from a collection of pre-trained models without accessing their original data, enabling the rapid adaptation to new unseen tasks. Current methods often overlook the heterogeneity among pre-trained models, which leads to performance degradation due to task conflicts. In this paper, we empirically and theoretically identify and analyze the model heterogeneity in DFML. We find that model heterogeneity introduces a heterogeneity-homogeneity trade-off, where homogeneous models reduce task conflicts but also increase the overfitting risk. Balancing this trade-off is crucial for learning shared representations across tasks. Based on our findings, we propose Task Groupings Regularization, a novel approach that benefits from model heterogeneity by grouping and aligning conflicting tasks. Specifically, we embed pre-trained models into a task space to compute dissimilarity, and group heterogeneous models together based on this measure. Then, we introduce implicit gradient regularization within each group to mitigate potential conflicts. By encouraging a gradient direction suitable for all tasks, the meta-model captures shared representations that generalize across tasks. Comprehensive experiments showcase the superiority of our approach in multiple benchmarks, effectively tackling the model heterogeneity in challenging multi-domain and multi-architecture scenarios.

## Requirements

```
pip install -r requirements.txt
```

## Datasets & Pre-trained Modes:

**Datasets:**

* **CIFAR-FS:** 

  * Please manually download the CIFAR-FS dataset. The directory structure is presented as follows:

    ```css
    cifar100
    ├─mete_train
    	├─apple (label_directory)
    		└─ ***.png (image_file)
    	...
    ├─mete_val
    	├─ ...
    		├─ ...
    └─mete_test
    	├─ ...
    		├─ ...
    ```

  * Place it in "./DFL2Ldata/cifar100".

* **miniImagenet:** Please manually download it, and then place it in "./DFL2Ldata/Miniimagenet".

* **CUB:** Please manually download it, and then place it in "./DFL2Ldata/CUB_200_2011".

**Pre-trained models:**


- You can prepare pre-trained models following the instructions below (Step 3).

## Quick Start:

1. Make sure that the root directory is "./TGR".

2. Prepare the dataset files.

   - For CIFAR-FS:

     ```shell
     python write_file/write_cifar100_filelist.py
     ```

     After running, you will obtain "meta_train.csv", "meta_val.csv", and "meta_test.csv" files under "./DFL2Ldata/cifar100/split/".

   - For miniImageNet:
     ```shell
     python write_file/write_miniimagenet_filelist.py
     ```
     
   - For CUB:
     ```shell
     python write_file/write_CUB_filelist.py
     ```
    
3. Prepare the pre-trained models.

    ```shell
    bash ./scripts/pretrain.sh
    ```
	
    Some options you may change:

    |     Option     |           Help            |
    | :------------: | :-----------------------: |
    |   --dataset    | cifar100/miniimagenet/cub |
    |   --pre_backbone    | conv4/resnet10/resnet18/resnet50 |

4. Data-free meta-learning
   - For main results:
     ```shell
      bash ./scripts/mr.sh
     ```
   - For multi-domain scenario:
     ```shell
      bash ./scripts/md.sh
     ```
   - For multi-architecture scenario:
     ```shell
      bash ./scripts/ma.sh
     ```
   Some options you may change:
   
   |     Option     |           Help            |
   | :------------: | :-----------------------: |
   |   --dataset    | cifar100/miniimagenet/cub/mix |
   | --num_sup_train |  1 for 1-shot, 5 for 5-shot  |
   |   --pre_backbone    | conv4/resnet10/resnet18/mix |

## Citation
If you find TGR useful for your research and applications, please cite using this BibTeX:
```bash
@inproceedings{weitask,
  title={Task Groupings Regularization: Data-Free Meta-Learning with Heterogeneous Pre-trained Models},
  author={Wei, Yongxian and Hu, Zixuan and Shen, Li and Wang, Zhenyi and Li, Yu and Yuan, Chun and Tao, Dacheng},
  booktitle={Forty-first International Conference on Machine Learning}
  year={2024}
}
```