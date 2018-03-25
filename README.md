# RPDehazingNet
solution for NTIRE2018 Image Dehazing Challenge (19.3db for Indoor and 21.3db for Outdoor)

## Preparation
Using data_argument to enchance the datasets, it will produce below datasets
```bash
$ python dara_argument.py --fold_A=IndoorTrainHzay --fold_B=IndoorTrainGT --fold_AB=IndoorTrain 

IndoorTrain
    \data   hazy image
    \label  clear image
```

## Train
Using default parameter to train
```bash
python train.py --cuda --gpus=4 --train=/path/to/train --test=/path/to/test --lr=0.0001 --step=1000
```

## Test
```bash
python test.py --cuda --checkpoints=/path/to/checkpoint --test=/path/to/testimages
```