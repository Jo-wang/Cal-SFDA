# Cal-SFDA
The official repo of the paper "Cal-SFDA: Source-Free Domain-adaptive Semantic Segmentation with Differentiable Expected Calibration Error", published in ACM Multimedia 2023.

## Training & Testing
### Train the source-only model
* Train with a pretrained ResNet model: 
```shell script
python so_run.py
```
where all the configs are in the  `./config/so_config.yml` file.

### Train the value net model
* Train with selected source checkpoint: 
```shell script
python rl_run.py
```
where all the configs are in the  `./config/rl_config.yml` file.

### Target adaptation
* Train with seg model + value net checkpoint: 
```shell script
python adaptive_target_run.py
```
where all the configs are in the  `./config/adaptive_target_config.yml` file.

#### To use multiple GPU:
Modify your GPU id in  `xx_run.py` file. (replace xx with the corresponding training process)

e.g.,
 `os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'`
 
 #### some important config in `xx_config.yml`:
 - `init_weight`: initial weight from ResNet101 pretrained model
 - `restore_from`: restore from a specific epoch address, e.g., `epoch1.pth`
 -  `snapshot`: the address you want to save your checkpoint.
 - `rl_restore_from`: restore the seg model + value net checkpoint.
 - `plabel`: your pseudo label path.
  
 **Note**: run this experiment requires Weights & Biases to log the performance. Please install it in your own environment: `pip install wandb`

## Acknowledgement
- https://github.com/fumyou13/LDBE

## Citation
If you find our repository is helpful, please consider citing our paper

      @article{DBLP:journals/corr/abs-2308-03003,
  author       = {Zixin Wang and
                  Yadan Luo and
                  Zhi Chen and
                  Sen Wang and
                  Zi Huang},
  title        = {Cal-SFDA: Source-Free Domain-adaptive Semantic Segmentation with Differentiable
                  Expected Calibration Error},
  journal      = {CoRR},
  volume       = {abs/2308.03003},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2308.03003},
  doi          = {10.48550/ARXIV.2308.03003},
  eprinttype    = {arXiv},
  eprint       = {2308.03003},
  timestamp    = {Mon, 21 Aug 2023 17:38:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2308-03003.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
