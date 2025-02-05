# pcam-classification
EfficientNet-B0 was used to classify tumor cells in images of H&E stained lymph node tissue. 

The first round of training froze all of the parameters in the base of the model except for the last two blocks each containing a convolutional layer, batch norm and SiLU and the classifier. Parameters were initialized from an EfficientNet-B0  model that was previously trained on ImageNet. The last two blocks were unfrozen instead of just the classifier since the ImageNet features should be significantly different than PCAM, given the different subject material and image resolution. The classifier parameters were replaced since they require a different number of neurons.

| Learning rate  | Batch size | Loss          | Optimizer | Weight decay | 
| -------------- | ---------- | ------------- | --------- | ------------ |
| 1e-4           | 32         | Cross Entropy | AdamW     | 1e-4         |

Table 1: Hyperparameters (round 1)

| Accuracy  | ROC-AUC | Test Loss | 
| --------- | ------- | --------- |
| 76%       | 0.83    | 0.5124    |

Table 2: Test Results (round 1)

In the second round of training, all parameters were allowed to update in the model. 

| Learning rate  | Batch size | Loss          | Optimizer | Weight decay | 
| -------------- | ---------- | ------------- | --------- | ------------ |
| 1e-5           | 32         | Cross Entropy | AdamW     | 1e-4         |

Table 3: Hyperparameters (round 2)

| Accuracy  | ROC-AUC | Test Loss | 
| --------- | ------- | --------- |
| 78%       | 0.87    | 0.4640    |

Table 4: Test Results (round 2)

![image](https://github.com/user-attachments/assets/a61ee0f6-8962-41f6-8a68-7b606841ca7d)  

Figure 1: PCAM image (no tumor)


![image](https://github.com/user-attachments/assets/940b2bf4-4962-458a-b461-3556e64adffa)

Figure 2: GradCAM for fig. 1

![image](https://github.com/user-attachments/assets/9a058a17-08fb-41f0-9a8e-bc6703626eae)

Figure 3: PCAM image (tumor)

![image](https://github.com/user-attachments/assets/ad902bcc-436e-4f0a-a628-4ddd29340c3c)

Figure 4: GradCAM for fig. 3
