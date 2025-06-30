import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTENC
import joblib

# --- Load & Preprocess Data ---
def clean_data(data):
    data = data.dropna().drop_duplicates()
    data = data[(data['Age'] > 0) & (data['Height'] > 0)]
    return data

def load_data(path):
    df = pd.read_excel(path, sheet_name='Obesity_Dataset ')
    df = clean_data(df)
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col])
    return df

data = load_data("Obesity_Dataset.xlsx")
X = data.drop(columns=["Class"])
y = data["Class"]

# --- Tentukan kolom kategorikal ---
# Ubah ke list index, misalnya kolom string awalnya: Sex, Overweight_Obese_Family, dll
categorical_cols = [
    'Sex',
    'Overweight_Obese_Family',
    'Consumption_of_Fast_Food',
    'Frequency_of_Consuming_Vegetables',
    'Number_of_Main_Meals_Daily',
    'Food_Intake_Between_Meals',
    'Smoking',
    'Liquid_Intake_Daily',
    'Calculation_of_Calorie_Intake',
    'Physical_Excercise',
    'Schedule_Dedicated_to_Technology',
    'Type_of_Transportation_Used'
]
cat_indices = [X.columns.get_loc(col) for col in categorical_cols]

# --- Split sebelum scaling ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# --- Apply SMOTENC on unscaled data ---
smote = SMOTENC(categorical_features=cat_indices, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Distribusi kelas setelah SMOTENC (y_train):")
print(pd.Series(y_train_resampled).value_counts())

# --- Scaling setelah SMOTENC ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# --- Save model & scaler ---
joblib.dump(model, "model/rf_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# --- Evaluate ---
y_pred = model.predict(X_test_scaled)

print("\n=== Evaluation Report ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nJumlah data per kelas (y):")
print(y.value_counts())
print("\nDistribusi kelas di test set:")
print(pd.Series(y_test).value_counts())

print("\nModel dan scaler berhasil disimpan ke folder 'model/'")
