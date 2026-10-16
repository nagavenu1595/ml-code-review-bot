import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

enc = OneHotEncoder()
X_enc = enc.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enc)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))