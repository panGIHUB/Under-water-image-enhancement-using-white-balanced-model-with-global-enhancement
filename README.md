# Handwritten Digit Recognition using Random Forest Classifier

## 📌 Project Overview
This project implements a machine learning model to recognize handwritten digits (0-9) from 8×8 pixel images using a Random Forest classifier. The model achieves **99% accuracy** on the test set.

![Sample Digits](https://via.placeholder.com/400x200?text=Handwritten+Digits+Example)

## 🛠️ Technical Implementation

### Dataset
- Uses scikit-learn's built-in `digits` dataset
- Contains **1,797 samples** of 8×8 grayscale images
- Each image represents a digit from 0 to 9

### Data Preprocessing
1. **Flattening**: Converts 8×8 images to 64-pixel vectors
2. **Normalization**: Scales pixel values from 0-16 range to 0-1
```python
# Flatten and normalize example
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data = data/16  # Normalize
```

### Model Training
- **Algorithm**: Random Forest Classifier
- **Train-Test Split**: 70% training, 30% testing
- **Default Parameters**: Used scikit-learn's default Random Forest settings

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

## 📊 Performance Metrics

### Classification Report
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99        61
           1       0.96      1.00      0.98        54
           2       0.98      0.98      0.98        54
           3       1.00      1.00      1.00        45
           4       0.98      1.00      0.99        52
           5       1.00      0.96      0.98        56
           6       1.00      1.00      1.00        54
           7       0.98      0.98      0.98        53
           8       1.00      0.96      0.98        57
           9       0.98      0.98      0.98        54

    accuracy                           0.99       540
   macro avg       0.99      0.99      0.99       540
weighted avg       0.99      0.99      0.99       540
```

### Confusion Matrix
![Confusion Matrix](https://via.placeholder.com/300x300?text=Confusion+Matrix)

## 🚀 How to Run

1. **Install dependencies**:
bash
pip install numpy matplotlib scikit-learn


2. **Run the Jupyter notebook**:
```bash
jupyter notebook machine_learning.ipynb
```

3. **View results**:
- Sample digit visualizations
- Model training process
- Performance metrics

 📂 **Repository Structure**
```
handwritten-digit-recognition/
├── machine_learning.ipynb      # Main Jupyter notebook
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

 🔍 **Future Improvements**
- Implement real-time digit drawing and prediction
- Experiment with convolutional neural networks
- Add hyperparameter tuning for Random Forest
- Create web interface for predictions

 📜 **License**
This project is licensed under the MIT License.

 ✉️ **Contact**
Pankaj Kumar  
📧 pankajpk7017kr@gmail..com 
🔗 https://github.com/panGIHUB

---

⭐ **Please star this repository if you find it useful!**
