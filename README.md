## SIGIR24_AutoDCS

Experiments codes for the paper:

Dugang Liu, Shenxian Xian, Yuhao Wu, Chaohua Yang, Xing Tang, Xiuqiang He, Zhong Ming. AutoDCS: Automated Decision Chain Selection in Deep Recommender Systems. In Proceedings of SIGIR '24.

**Please cite our SIGIR '24 paper if you use our codes. Thanks!**

## Requirement

See the contents of requirements.txt

## Usage

Note that our implementation is based on MBCGCN ([Link](https://github.com/SS-00-SS/MBCGCN)). Please first go to this link ([Link](https://drive.google.com/file/d/1siYEqgCFdVjvIYA6gQZq1jV9dP13-8ea/view?usp=sharing)) to obtain the dataset and pre-trained model files we provide.

The examples of running AutoDCS:

For Tmall,

```bash
python AutoDCS-MBCGCN_concate.py --data_path Data/Tmall/ --data_path Data/Tmall/ --weights_path ./output/Tmall/ --dataset 'buy' --dataset2 'cart' --dataset3 'view' --pretrain 1  --epoch 1000  --embed_size 64  --layer_size [64,64,64]  --layer_size2 [64,64,64] --layer_size3 [64,64,64]  --layer_size4 [64,64]   --batch_size 2048  --regs [1e-4]   --lr 0.001  --model_type 'mbcgcn'  --alg_type 'mbcgcn'  --adj_type 'pre'  --save_flag 0
```

```bash
python AutoDCS-MBCGCN_add.py --data_path Data/Tmall/ --data_path Data/Tmall/ --weights_path ./output/Tmall/ --dataset 'buy' --dataset2 'cart' --dataset3 'view' --pretrain 1  --epoch 1000  --embed_size 64  --layer_size [64,64,64]  --layer_size2 [64,64,64] --layer_size3 [64,64,64]  --layer_size4 [64,64]   --batch_size 2048  --regs [1e-4]   --lr 0.001  --model_type 'mbcgcn'  --alg_type 'mbcgcn'  --adj_type 'pre'  --save_flag 0 --gumbel 1
```

For taobao,

```bash
python AutoDCS-MBCGCN_concate.py --data_path Data/taobao/ --weights_path ./output/taobao/ --dataset 'buy' --dataset2 'cart' --dataset3 'view' --pretrain 1  --epoch 1000  --embed_size 64  --layer_size [64,64,64]  --layer_size2 [64,64,64] --layer_size3 [64,64,64]  --layer_size4 [64,64]   --batch_size 2048  --regs [1e-4]   --lr 0.001  --model_type 'mbcgcn'  --alg_type 'mbcgcn'  --adj_type 'pre'  --save_flag 0
```

```bash
python AutoDCS-MBCGCN_add.py --data_path Data/taobao/ --weights_path ./output/taobao/ --dataset 'buy' --dataset2 'cart' --dataset3 'view' --pretrain 1  --epoch 1000  --embed_size 64  --layer_size [64,64,64]  --layer_size2 [64,64,64] --layer_size3 [64,64,64]  --layer_size4 [64,64]   --batch_size 2048  --regs [1e-4]   --lr 0.001  --model_type 'mbcgcn'  --alg_type 'mbcgcn'  --adj_type 'pre'  --save_flag 0
```

For Jdata,

```bash
python AutoDCS-MBCGCN_concate.py --data_path Data/jdata/ --weights_path ./output/jdata/ --dataset 'buy' --pretrain 1  --epoch 1000  --embed_size 64  --layer_size [64,64,64]  --layer_size2 [64,64,64] --layer_size3 [64,64,64,64]  --layer_size4 [64,64,64] --batch_size 2048 --save_flag 0
```

```bash
python AutoDCS-MBCGCN_add.py --data_path Data/jdata/ --weights_path ./output/jdata/ --dataset 'buy' --pretrain 1  --epoch 1000  --embed_size 64  --layer_size [64,64,64]  --layer_size2 [64,64,64] --layer_size3 [64,64,64,64]  --layer_size4 [64,64,64] --batch_size 2048 --save_flag 0 --gumbel 1
```

## 

If you have any issues or ideas, feel free to contact us ([dugang.ldg@gmail.com](mailto:dugang.ldg@gmail.com)).
