import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

sheets = ["2021", "2022", "2023", "2024"]
df_list = []

for sheet in sheets:
    df_sheet = pd.read_excel("stunting-nutritional-data.xlsx", sheet_name=sheet)
    
    df_sheet = df_sheet.rename(columns={
        "Age (Month)": "age",
        "Weight": "weight",
        "Height": "height",
        "Height for Age": "label"
    })
    
    df_sheet = df_sheet[["age", "weight", "height", "label"]]
    
    df_sheet["label"] = df_sheet["label"].str.strip().str.title()
    df_sheet["label"] = df_sheet["label"].replace({"Not Stunted": "Normal"})
    
    df_list.append(df_sheet)

df = pd.concat(df_list, ignore_index=True)

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["height"] = pd.to_numeric(df["height"], errors="coerce")

df = df.dropna(subset=["age", "weight", "height", "label"])

print("Label unik:", df["label"].unique())
print("Jumlah per label:")
print(df["label"].value_counts())
print("Jumlah data setelah drop:", len(df))

X = df[["age", "weight", "height"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

with open("stunting_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai stunting_model.pkl")
