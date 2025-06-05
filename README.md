# SoundRain

## Quick setup instructions

### Generate dataset splits

```
python split_data.py path/to/dataset ./splits 
```

### Change your config file for training (mainly change paths to be yours)

```
vi config/train/train.json 
```

### Run experiment

```
python train.py -C config/train/train.json
```
