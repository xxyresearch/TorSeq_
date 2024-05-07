# TorSeq: Torsion Sequential Modeling for Molecular 3D Conformation Generation
Implementation of TorSeq by Xiangyang Xu, Meng Liu, and Hongyang Gao 

If you have any question, please open issue or send us email at xyxu@iastate.edu

## Enviornment
Create a new conda enviornment and install pytorch and pyg

    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install torch_geometric
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
    pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html

## Dataset and checkpoints
We use the same dataset and split as Torsional Diffusion (Jing et al), you can download dataset from [here](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=sharing).

Also, we provide checkpoints and sampled molecular conformers pickle [here](https://drive.google.com/drive/folders/14z7b4-TzAinVt2rjDlA2flX70Qmt7pbi?usp=drive_link).

## Training
    python train_main.py --mpnn_conv --no_random_start --no_local_feature

## Sampling
    python sampling_torseq.py --no_local_feature --use_motif_gcn

## Testing
    python test.py --save TorSeq

