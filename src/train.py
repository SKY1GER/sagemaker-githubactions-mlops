import os
import joblib
import tarfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("/opt/ml/processing/train/train.csv")
test = pd.read_csv("/opt/ml/processing/test/test.csv")
X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("test_accuracy:", accuracy_score(y_test, model.predict(X_test)))

output_dir = "/opt/ml/processing/model"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.joblib")
joblib.dump(model, model_path)

tar_path = os.path.join(output_dir, "model.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(model_path, arcname="model.joblib")
os.remove(model_path)
