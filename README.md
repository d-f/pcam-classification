# pcam-classification
EfficientNet-B0 was used to classify tumor cells in images of H&E stained lymph node tissue. 

The first round of training froze all of the parameters in the base of the model except for the last two blocks each containing a convolutional layer, batch norm and SiLU and the classifier. Parameters were initialized from an EfficientNet-B0  model that was previously trained on ImageNet. The last two blocks were unfrozen instead of just the classifier since the ImageNet features should be significantly different than PCAM, given the different subject material and image resolution. The classifier parameters were replaced since they require a different number of neurons.

Hyperparameters
| Learning rate  | Batch size | Loss          | Optimizer | Weight decay | 
| -------------- | ---------- | ------------- | --------- | ------------ |
| 1e-4           | 32         | Cross Entropy | AdamW     | 1e-4         |

Test Results
| Accuracy  | ROC-AUC | Test Loss | 
| --------- | ------- | --------- |
| 76%       | 0.83    | 0.5124    |

In the second round of training, all parameters were allowed to update in the model. 

Hyperparameters
| Learning rate  | Batch size | Loss          | Optimizer | Weight decay | 
| -------------- | ---------- | ------------- | --------- | ------------ |
| 1e-5           | 32         | Cross Entropy | AdamW     | 1e-4         |

Test Results
| Accuracy  | ROC-AUC | Test Loss | 
| --------- | ------- | --------- |
| 78%       | 0.87    | 0.4640    |
