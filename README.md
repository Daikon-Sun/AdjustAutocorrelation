# Adjusting for Autocorrelated Errors in Neural Networks for Time Series

This repository is the official implementation of the paper "Adjusting for Autocorrelated Errors in Neural Networks for Time Series" ([arXiv link](https://arxiv.org/abs/2101.12578)).

For simplicity, we only provide the code for time series forecasting.
However, it is straightforward to implement our method with other models on other time series tasks as described in the appendix of the paper.

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Datasets

Available datasets are located in the directory `data/forecasting`.
Traffic is not included because it exceeds the 100MB size limit set by Github.
However, you can download it [here](https://github.com/laiguokun/multivariate-time-series-data) and format the it into a `.npy` file.
ADI-related datasets are not released because they are proprietary.

To use your own dataset, format it into a numpy array with size TxN and saved it into the data directory as a `.npy` file.

## Training and Evaluation

Example commands can be found in `run.sh`.

## Citation

```
@inproceedings{sun2021adjusting,
    title={Adjusting for Autocorrelated Errors in Neural Networks for Time Series Regression and Forecasting}, 
    author={Fan-Keng Sun and Christopher I. Lang and Duane S. Boning},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2021},
}
```
