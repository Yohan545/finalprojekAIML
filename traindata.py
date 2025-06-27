import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
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

data = load_data("Obesity_Dataset.xlsx")  # Ganti dengan path file kamu
X = data.drop(columns=["Class"])
y = data["Class"]

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split with Stratification ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.3, random_state=42
)

# --- Apply SMOTE only to training data ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Distribusi kelas setelah SMOTE (y_train):")
print(pd.Series(y_train_resampled).value_counts())

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# --- Save model & scaler ---
joblib.dump(model, "model/rf_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# --- Evaluate ---
y_pred = model.predict(X_test)

print("\n=== Evaluation Report ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nJumlah data per kelas (y):")
print(y.value_counts())
print("\nDistribusi kelas di test set:")
print(pd.Series(y_test).value_counts())

print("\nModel dan scaler berhasil disimpan ke folder 'model/'")
