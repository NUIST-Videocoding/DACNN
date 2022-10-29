# DACNN

This is the source code for the paper "DACNN: Blind Image Quality Assessment via A Distortion-Aware Convolutional Neural Network".

## Dependencies

- python=3.6+
- pytorch=1.1+
- scipy=1.5+
- efficientnet-pytorch=0.7.0

## Usages

### Pretrained model

The pretrained efficientnet for SDANet is in the `./pretrain_model/model` folder. If you want to re-train this model, please remember to revise the code in the `./pretrain_model/EfficientNet.py` and run the `./pretrain_model/train.py`.

### Training & Testing on IQA databases

Please put the databases in the `data` folder, and revise the `--dataset` options in the `train_test.py`, then run:

```python
python train_test.py
```

### Citation

<u>**If you find this work useful for your research, please cite our paper:**</u>

@ARTICLE{9817377,
  author={Pan, Zhaoqing and Zhang, Hao and Lei, Jianjun and Fang, Yuming and Shao, Xiao and Ling, Nam and Kwong, Sam},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DACNN: Blind Image Quality Assessment via A Distortion-Aware Convolutional Neural Network}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3188991}}
