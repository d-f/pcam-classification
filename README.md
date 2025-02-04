# pcam-classification
EfficientNet-b0 was used to classify malignancies in images of H&E stained lymph node tissue. 

Hyperparameters
| Learning rate  | Batch size | Loss          | Optimizer | Weight decay | 
| -------------- | ---------- | ------------- | --------- | ------------ |
| 1e-4           | 32         | Cross Entropy | AdamW     | 1e-4         |

Test Results
| Accuracy  | ROC-AUC | Test Loss | 
| --------- | ------- | --------- |
| 76%       | 0.83    | 0.5124    |
