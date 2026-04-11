# Model Summary

## Architecture

```
NeuralNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=4, out_features=244, bias=True)
    (1): LeakyReLU(negative_slope=0.01)
    (2): Dropout(p=0.009364659319868551, inplace=False)
    (3): Linear(in_features=244, out_features=244, bias=True)
    (4): LeakyReLU(negative_slope=0.01)
    (5): Dropout(p=0.009364659319868551, inplace=False)
    (6): Linear(in_features=244, out_features=1, bias=True)
  )
  (dropout): Dropout(p=0.009364659319868551, inplace=False)
)
```

## Torchinfo Summary

```
Layer (type:depth-idx)                   Output Shape              Param #
NeuralNetwork                            [64, 1]                   --
  ModuleList: 1-5                        --                        (recursive)
    Linear: 2-1                          [64, 244]                 1,220
    LeakyReLU: 2-2                       [64, 244]                 --
  Dropout: 1-2                           [64, 244]                 --
  ModuleList: 1-5                        --                        (recursive)
    Linear: 2-3                          [64, 244]                 59,780
    LeakyReLU: 2-4                       [64, 244]                 --
  Dropout: 1-4                           [64, 244]                 --
  ModuleList: 1-5                        --                        (recursive)
    Linear: 2-5                          [64, 1]                   245

Total params: 61,245
Trainable params: 61,245
Non-trainable params: 0
Total mult-adds (MB): 3.92
Input size (MB): 0.00
Forward/backward pass size (MB): 0.25
Params size (MB): 0.24
Estimated Total Size (MB): 0.50
```
