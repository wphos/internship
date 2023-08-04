# 生产实习

---

### 代码说明

- hash_center : 用于生成哈希中心的代码
- SEMICON_optimize_global : 对 SEMICON 的 global feature learning branch 进行优化的代码
- SEMICON_optimize_all : 对整个 SEMICON 模型进行优化的代码

---

### 环境

- Python  3.8.5
- pytorch  1.11.0
- torchvision  0.12.0
- numpy  1.24.4
- pandas  1.5.3
- scipy  1.9.3
- loguru  0.5.3
- tqdm  4.54.1

---

### 生成哈希中心

修改 hash_center/generate.py 的第 3 行和第 4 行，指明数据集的类别数量和需要生成的哈希编码长度，然后直接运行 generate.py

---

### 训练模型

以下是在不同数据集上训练模型的命令，训练前需要将 root 的参数改为数据集的路径。

CUB 200-2011 数据集:

```
python run.py --dataset cub-2011 --root dataset/CUB2011/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'CUB-SEMICON' --momen=0.91
```

Aircraft 数据集:

```
python run.py --dataset aircraft --root dataset/aircraft/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'Aircraft-SEMICON' --momen=0.91
```

VegFru 数据集:

```
python run.py --dataset vegfru --root dataset/vegfru/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info 'VegFru-SEMICON' --momen=0.91
```

Food101 数据集:

```
python run.py --dataset food101 --root dataset/food101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info 'Food101-SEMICON' --momen 0.91
```

NABirds 数据集:

```
python run.py --dataset nabirds --root dataset/nabirds/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info 'NAbirds-SEMICON' --momen=0.91
```

---

### 测试模型

CUB 200-2011 数据集:

```
python run.py --dataset cub-2011 --root dataset/CUB2011/CUB_200_2011 --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'CUB-SEMICON'
```

Aircraft 数据集:

```
python run.py --dataset aircraft --root dataset/aircraft/ --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'Aircraft-SEMICON'
```

VegFru 数据集:

```
python run.py --dataset vegfru --root dataset/vegfru/ --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'VegFru-SEMICON'
```

Food101 数据集:

```
python run.py --dataset food101 --root dataset/food101/ --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'Food101-SEMICON'
```

NABirds 数据集:

```
python run.py --dataset nabirds --root dataset/nabirds/ --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'NAbirds-SEMICON'
```

