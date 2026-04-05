NeuralNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=9, out_features=256, bias=True)
    (1): GELU(approximate='none')
    (2): Dropout(p=1.855105006454693e-05, inplace=False)
    (3): Linear(in_features=256, out_features=256, bias=True)
    (4): GELU(approximate='none')
    (5): Dropout(p=1.855105006454693e-05, inplace=False)
    (6): Linear(in_features=256, out_features=256, bias=True)
    (7): GELU(approximate='none')
    (8): Dropout(p=1.855105006454693e-05, inplace=False)
    (9): Linear(in_features=256, out_features=256, bias=True)
    (10): GELU(approximate='none')
    (11): Dropout(p=1.855105006454693e-05, inplace=False)
    (12): Linear(in_features=256, out_features=256, bias=True)
    (13): GELU(approximate='none')
    (14): Dropout(p=1.855105006454693e-05, inplace=False)
    (15): Linear(in_features=256, out_features=3, bias=True)
  )
  (dropout): Dropout(p=1.855105006454693e-05, inplace=False)
)


--- Torchinfo Summary ---
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
NeuralNetwork                            [128, 3]                  --
├─ModuleList: 1-11                       --                        (recursive)
│    └─Linear: 2-1                       [128, 256]                2,560
│    └─GELU: 2-2                         [128, 256]                --
├─Dropout: 1-2                           [128, 256]                --
├─ModuleList: 1-11                       --                        (recursive)
│    └─Linear: 2-3                       [128, 256]                65,792
│    └─GELU: 2-4                         [128, 256]                --
├─Dropout: 1-4                           [128, 256]                --
├─ModuleList: 1-11                       --                        (recursive)
│    └─Linear: 2-5                       [128, 256]                65,792
│    └─GELU: 2-6                         [128, 256]                --
├─Dropout: 1-6                           [128, 256]                --
├─ModuleList: 1-11                       --                        (recursive)
│    └─Linear: 2-7                       [128, 256]                65,792
│    └─GELU: 2-8                         [128, 256]                --
├─Dropout: 1-8                           [128, 256]                --
├─ModuleList: 1-11                       --                        (recursive)
│    └─Linear: 2-9                       [128, 256]                65,792
│    └─GELU: 2-10                        [128, 256]                --
├─Dropout: 1-10                          [128, 256]                --
├─ModuleList: 1-11                       --                        (recursive)
│    └─Linear: 2-11                      [128, 3]                  771
==========================================================================================
Total params: 266,499
Trainable params: 266,499
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 34.11
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 1.31
Params size (MB): 1.07
Estimated Total Size (MB): 2.38
==========================================================================================