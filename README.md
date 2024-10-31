# Traffic Accident Severity Prediction 🚗

## Project Overview
This project implements machine learning models to predict the severity of traffic accidents using various features such as driver characteristics, road conditions, vehicle information, and environmental factors. The analysis employs multiple classification algorithms and compares their performance using different metrics.

## 📊 Features
- Comprehensive data preprocessing and feature engineering
- Implementation of multiple machine learning algorithms
- Handling of imbalanced data using SMOTE
- Detailed performance comparison and visualization
- Confusion matrix analysis for each model

## 🛠️ Technologies Used
- Python 3.x
- Libraries:
  - pandas
  - scikit-learn
  - imbalanced-learn
  - seaborn
  - matplotlib
  - numpy

## 💾 Dataset Description
The dataset used in this study is sourced from the Road Accident Severity in India dataset, comprising:
- **Size**: 2890 records with 32 attributes
- **Source**: Road Accident Severity in India dataset
- **Target Variable**: Accident severity (3 levels)
  - Slight Injury
  - Serious Injury
  - Fatal Injury

### Key Features Include:
1. **Temporal Information**
   - Time
   - Day of week

2. **Driver Characteristics**
   - Age band
   - Sex
   - Educational level
   - Driving experience

3. **Vehicle Information**
   - Type of vehicle
   - Owner of vehicle
   - Service year of vehicle

4. **Road Conditions**
   - Area accident occurred
   - Road alignment
   - Types of junction
   - Road surface type

5. **Environmental Factors**
   - Light conditions
   - Weather conditions

6. **Accident Details**
   - Type of collision
   - Vehicle movement
   - Casualty class
   - Cause of accident

## 🔄 Machine Learning Pipeline
1. **Data Preprocessing**
   - Time feature extraction
   - Categorical variable encoding
   - Feature scaling
   - Missing value handling

2. **Feature Engineering**
   - Conversion of categorical variables to numeric format
   - Standardization of features
   - SMOTE for handling class imbalance

3. **Models Implemented**
   - Histogram-based Gradient Boosting (HGB)
   - Random Forest (RF)
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

## 📈 Model Performance
Performance comparison of the implemented algorithms:

| Algorithm | Accuracy | Precision | Recall | F1-score |
|-----------|----------|-----------|---------|-----------|
| HGB       | 0.938    | 0.943     | 0.938   | 0.939     |
| RF        | 0.947    | 0.954     | 0.947   | 0.947     |
| SVM       | 0.937    | 0.942     | 0.937   | 0.937     |
| KNN       | 0.868    | 0.894     | 0.872   | 0.863     |

### Key Findings:
- Random Forest achieved the best overall performance with highest scores across all metrics
- HGB and SVM showed comparable performance, both achieving >93% accuracy
- KNN performed relatively lower but still maintained decent accuracy at 86.8%

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib numpy
```

### Usage
1. Clone the repository
2. Ensure your dataset is in the correct path
3. Run the main script:
```python
cd  /Accident_Severity
python accident_severity_prediction.py
```

## 📊 Visualizations
The project includes various visualizations:
- Performance metric comparisons across models
- Confusion matrices for each algorithm
- Feature importance plots
- Correlation heatmaps

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Authors
- _srirahul_

## 📮 Contact
- GitHub: [\My Profile\]](https://github.com/Sri-Rahul)

## 🙏 Acknowledgments
- Road Accident Severity in India dataset contributors - [Dataset](https://www.kaggle.com/datasets/kanuriviveknag/road-accidents-severity-dataset)
- Special thanks to all researchers and organizations working on road safety improvement
