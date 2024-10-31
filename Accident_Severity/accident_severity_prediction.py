import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Load the data

df = pd.read_csv("/RTADatasetE1.csv")

# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

# Extract the hour from the 'Time' column
df['Hour'] = df['Time'].dt.hour

# Convert the categorical columns to numeric
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6, 'NA': -1}
df['Day_of_week'] = df['Day_of_week'].map(day_map)

age_map = {'Under 18': 0, '18-30': 1, '31-50': 2, 'Above 50': 3, 'NA': -1}
df['Age_band_of_driver'] = df['Age_band_of_driver'].map(age_map)
df['Age_band_of_casualty'] = df['Age_band_of_casualty'].map(age_map)

le = LabelEncoder()
df['Sex_of_driver'] = le.fit_transform(df['Sex_of_driver'])
df['Sex_of_casualty'] = le.fit_transform(df['Sex_of_casualty'])

edu_map = {'Elementary school': 0, 'Junior high school': 1, 'High school': 2, 'College': 3, 'Illiterate': -1, 'Writing & reading': -1}
df['Educational_level'] = df['Educational_level'].map(edu_map)

vdr_map = {'Employee': 1, 'Owner': 0, 'Unknown': -1, 'Other': 3}
df['Vehicle_driver_relation'] = df['Vehicle_driver_relation'].map(vdr_map)

de_map = {'Above 10yr': 4, '1-2yr': 1, '5-10yr': 3, 'Below 1yr': -1, '2-5yr': 2, 'No Licence': -1}
df['Driving_experience'] = df['Driving_experience'].map(de_map)

tov_map = {'Lorry (41–100Q)': 1, 'Automobile': 0, 'Public (> 45 seats)': 2, 'Long lorry': -1, 'Lorry (11–40Q)': -1, 'Taxi': -1, 'Public (12 seats)': -1, 'Ridden horse': -1, 'Other': -1, 'Pick up upto 10Q': -1, 'Bajaj': -1, 'Public (13–45 seats)': -1, 'Motorcycle': 3, 'Stationwagen': -1, 'Special vehicle': -1, 'Turbo': -1, 'Bicycle': -1}
df['Type_of_vehicle'] = df['Type_of_vehicle'].map(tov_map)

ov_map = {'Owner': 0, 'Governmental': -1, 'Organization': -1, 'Other': -1}
df['Owner_of_vehicle'] = df['Owner_of_vehicle'].map(ov_map)

syv_map = {'1-2yr': 1, '2-5yrs': 2, 'Above 10yr': 4, '5-10yrs': 3, 'Below 1yr': -1}
df['Service_year_of_vehicle'] = df['Service_year_of_vehicle'].map(syv_map)

dov_map = {'No defect': 0, '7': 1, '5': 1}
df['Defect_of_vehicle'] = df['Defect_of_vehicle'].map(dov_map)

aao_map = {'Industrial areas': 2, 'Residential areas': 0, 'Office areas': 1, 'Church areas': -1, 'Market areas': -1, 'Rural village areas': -1, 'Other': -1, 'RecreatioNAl areas': -1, 'School areas': -1, 'Outside rural areas': -1, 'Hospital areas': -1, 'Rural village areasOffice areas': -1}
df['Area_accident_occured'] = df['Area_accident_occured'].map(aao_map)

lom_map = {'other': 2, 'Undivided Two way': 0, 'Double carriageway (median)': 1, 'One way': -1, 'Two-way (divided with broken lines road marking)': -1, 'Two-way (divided with solid lines road marking)': -1}
df['Lanes_or_Medians'] = df['Lanes_or_Medians'].map(lom_map)

ra_map = {'Tangent road with flat terrain': 0, 'Escarpments': 1, 'Tangent road with rolling terrain': -1, 'Gentle horizontal curve': -1, 'Tangent road with mountainous terrain and': -1, 'Tangent road with mild grade and flat terrain': -1, 'Steep grade downward with mountainous terrain': -1, 'Sharp reverse curve': -1}
df['Road_allignment'] = df['Road_allignment'].map(ra_map)

toj_map = {'Y Shape': 1, 'No junction': 0, 'Crossing': -1, 'O Shape': -1, 'Other': -1, 'T Shape': 2, 'X Shape': -1}
df['Types_of_Junction'] = df['Types_of_Junction'].map(toj_map)

rst_map = {'Earth roads': 1, 'Asphalt roads': 0, 'Gravel roads': -1, 'Other': -1, 'Asphalt roads with some distress': -1}
df['Road_surface_type'] = df['Road_surface_type'].map(rst_map)

rsc_map = {'Dry': 0, 'Wet or damp': 1, 'Snow': -1, 'Flood over 3cm. deep': -1}
df['Road_surface_conditions'] = df['Road_surface_conditions'].map(rsc_map)

lc_map = {'Daylight': 0, 'Darkness - lights lit': -1, 'Darkness - lights unlit': -1, 'Darkness - no lighting': -1}
df['Light_conditions'] = df['Light_conditions'].map(lc_map)

wc_map = {'Normal': 0, 'Raining': -1, 'Raining and Windy': -1, 'Cloudy': -1, 'Windy': -1, 'Other': -1, 'Unknown': -1, 'Snow': -1}
df['Weather_conditions'] = df['Weather_conditions'].map(wc_map)

toc_map = {'Collision with roadside-parked vehicles': 0, 'Collision with animals': 1, 'Rollover': -1, 'Fall from vehicles': -1, 'Vehicle with vehicle collision': -1, 'Collision with pedestrians': -1, 'Collision with roadside objects': -1, 'Other': -1, 'With Train': -1}
df['Type_of_collision'] = df['Type_of_collision'].map(toc_map)

vm_map = {'Going straight': 0, 'U-Turn': 1, 'Waiting to go': 2, 'Moving Backward': -1, 'Reversing': -1, 'Turnover': -1, 'Other': -1, 'Stopping': -1, 'Overtaking': -1, 'Parked': -1, 'Getting off': -1, 'Entering a junction': -1}
df['Vehicle_movement'] = df['Vehicle_movement'].map(vm_map)

cc_map = {'Pedestrian': 0, 'Passenger': 1, 'Driver or rider': 2}
df['Casualty_class'] = df['Casualty_class'].map(cc_map)

df['Sex_of_casualty'] = le.fit_transform(df['Sex_of_casualty'])

cs_map = {'Slight Injury': 1, 'Serious Injury': 2, 'Fatal injury': 3}
df['Casualty_severity'] = df['Casualty_severity'].map(cs_map)

woc_map = {'Driver': 0, 'Passenger': 1, 'Pedestrian': 2, 'Other': 3, 'Self-employed': -1, 'Employee': -1, 'Unemployed': -1, 'Student': -1}
df['Work_of_casuality'] = df['Work_of_casuality'].map(woc_map)

foc_map = {'Normal': 0, 'Blind': -1, 'Deaf': -1}
df['Fitness_of_casuality'] = df['Fitness_of_casuality'].map(foc_map)

pm_map = {'Crossing from driver\'s nearside': 0, 'Not a Pedestrian': 2, 'Unknown or other': -1, 'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle': -1, 'Walking along in carriageway, back to traffic': -1, 'Crossing from offside - masked by parked or statioNot a Pedestrianry vehicle': -1, 'In carriageway, statioNot a Pedestrianry - not crossing (standing or playing)': -1, 'In carriageway, statioNot a Pedestrianry - not crossing (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle': -1}
df['Pedestrian_movement'] = df['Pedestrian_movement'].map(pm_map)

coa_map = {'Changing lane to the right': 0, 'Moving Backward': -1, 'No distancing': 3, 'No priority to vehicle': 2, 'Overtaking': -1, 'Other': -1, 'No priority to pedestrian': -1, 'Changing lane to the left': 1, 'Driving carelessly': -1, 'Turnover': -1, 'Driving to the left': -1, 'Driving at high speed': -1, 'Driving under the influence of drugs': -1, 'Getting off the vehicle improperly': -1, 'Overturning': -1, 'Overspeed': -1, 'Overloading': -1, 'Improper parking': -1, 'Drunk driving': -1, 'Unknown': -1}
df['Cause_of_accident'] = df['Cause_of_accident'].map(coa_map)


# Drop the 'Time' column

# Drop the 'Time' column
X = df.drop(['Time', 'Accident_severity'], axis=1)
y = df['Accident_severity']

# Remove columns with missing data
X = X.dropna(axis=1, how='any')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalanced data using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning
# 1. Histogram-based Gradient Boosting
hgb_params = {'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 5, 7], 'max_iter': [100, 200, 300]}
hgb_model = GridSearchCV(HistGradientBoostingClassifier(), hgb_params, cv=5)
hgb_model.fit(X_train, y_train)
hgb_pred = hgb_model.predict(X_test)
hgb_accuracy = accuracy_score(y_test, hgb_pred)
hgb_precision = precision_score(y_test, hgb_pred, average='macro', zero_division='warn')
hgb_recall = recall_score(y_test, hgb_pred, average='macro', zero_division='warn')
hgb_f1 = f1_score(y_test, hgb_pred, average='macro')
print("Histogram-based Gradient Boosting Accuracy:", hgb_accuracy)
print("Histogram-based Gradient Boosting Precision:", hgb_precision)
print("Histogram-based Gradient Boosting Recall:", hgb_recall)
print("Histogram-based Gradient Boosting F1-score:", hgb_f1)

# 2. Random Forest
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
rf_model = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average='macro', zero_division='warn')
rf_recall = recall_score(y_test, rf_pred, average='macro', zero_division='warn')
rf_f1 = f1_score(y_test, rf_pred, average='macro')
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-score:", rf_f1)


# 5. SVM
svm_params = {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}
svm_model = GridSearchCV(SVC(), svm_params, cv=5)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average='macro', zero_division='warn')
svm_recall = recall_score(y_test, svm_pred, average='macro', zero_division='warn')
svm_f1 = f1_score(y_test, svm_pred, average='macro')
print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1-score:", svm_f1)

# 6. KNN
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, average='macro', zero_division='warn')
knn_recall = recall_score(y_test, knn_pred, average='macro', zero_division='warn')
knn_f1 = f1_score(y_test, knn_pred, average='macro')
print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN F1-score:", knn_f1)




# Define the metrics and corresponding values for each algorithm
metrics = ['Precision', 'Accuracy', 'F1-score', 'Recall']
precision_values = [hgb_precision, rf_precision, svm_precision, knn_precision]
accuracy_values = [hgb_accuracy, rf_accuracy, svm_accuracy, knn_accuracy]
f1_score_values = [hgb_f1, rf_f1, svm_f1, knn_f1]
recall_values = [hgb_recall, rf_recall, svm_recall, knn_recall]

# Define the labels for the algorithms
algorithms = ['HGB', 'RF', 'SVM', 'KNN']

# Function to plot bar graph with highlighting the highest value
def plot_bar_graph(metric, values, algorithms):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, values, color='lightblue', edgecolor='black')
    max_index = values.index(max(values))
    bars[max_index].set_edgecolor('red')
    plt.xlabel('Algorithms')
    plt.ylabel(metric)
    plt.title(f'{metric} of Different Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot precision bar graph
plot_bar_graph('Precision', precision_values, algorithms)

# Plot accuracy bar graph
plot_bar_graph('Accuracy', accuracy_values, algorithms)

# Plot F1-score bar graph
plot_bar_graph('F1-score', f1_score_values, algorithms)

# Plot recall bar graph
plot_bar_graph('Recall', recall_values, algorithms)


# Confusion matrix for Histogram-based Gradient Boosting
hgb_conf_matrix = confusion_matrix(y_test, hgb_pred)

# Confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(y_test, rf_pred)





# Confusion matrix for SVM
svm_conf_matrix = confusion_matrix(y_test, svm_pred)

# Confusion matrix for KNN
knn_conf_matrix = confusion_matrix(y_test, knn_pred)

def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Plot each confusion matrix separately
plot_confusion_matrix(hgb_conf_matrix, 'HGB Confusion Matrix')
plot_confusion_matrix(rf_conf_matrix, 'RF Confusion Matrix')
plot_confusion_matrix(svm_conf_matrix, 'SVM Confusion Matrix')
plot_confusion_matrix(knn_conf_matrix, 'KNN Confusion Matrix')

