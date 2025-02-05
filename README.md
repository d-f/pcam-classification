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


![image](https://github.com/user-attachments/assets/a61ee0f6-8962-41f6-8a68-7b606841ca7d)
Original image (no tumor cells)

![image](https://github.com/user-attachments/assets/940b2bf4-4962-458a-b461-3556e64adffa)
GradCAM

![image](https://github.com/user-attachments/assets/9a058a17-08fb-41f0-9a8e-bc6703626eae)
Original Image (tumor cells)

![image](https://github.com/user-attachments/assets/ad902bcc-436e-4f0a-a628-4ddd29340c3c)
GradCAM
