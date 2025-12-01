import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

sheets = ["2021", "2022", "2023", "2024"]
df_list = []

for sheet in sheets:
    df_sheet = pd.read_excel("stunting-nutritional-data.xlsx", sheet_name=sheet)
    df_sheet = df_sheet.rename(columns={
        "Age (Month)": "age",
        "Weight": "weight",
        "Height": "height",
        "Height for Age": "label",
        "Gender": "sex" 
    })
    df_sheet = df_sheet[["age", "weight", "height", "label", "sex"]]
    df_sheet["label"] = df_sheet["label"].str.strip().str.title()
    df_sheet["label"] = df_sheet["label"].replace({"Not Stunted": "Normal"})
    df_list.append(df_sheet)

df = pd.concat(df_list, ignore_index=True)

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["height"] = pd.to_numeric(df["height"], errors="coerce")
df = df.dropna(subset=["age", "weight", "height", "label", "sex"])

df["age"] = (df["age"] / 12).apply(lambda x: int(round(x)))

bb_table = pd.DataFrame([
    [1, "M", (9.17+10.83)/2, (10.83-9.17)/4],
    [1, "F", (8.62+10.43)/2, (10.43-8.62)/4],
    [2, "M", (11.22+13.43)/2, (13.43-11.22)/4],
    [2, "F", (10.86+13.19)/2, (13.19-10.86)/4],
    [3, "M", (13.38+16.40)/2, (16.40-13.38)/4],
    [3, "F", (12.86+15.94)/2, (15.94-12.86)/4],
    [4, "M", (15.15+19.19)/2, (19.19-15.15)/4],
    [4, "F", (14.68+18.70)/2, (18.70-14.68)/4],
    [5, "M", (16.92+21.65)/2, (21.65-16.92)/4],
    [5, "F", (16.43+21.77)/2, (21.77-16.43)/4]
], columns=["age", "sex", "median", "sd"])

tb_table = pd.DataFrame([
    [1, "M", (73.39+78.71)/2, (78.71-73.39)/4],
    [1, "F", (72.09+77.26)/2, (77.26-72.09)/4],
    [2, "M", (83.97+90.74)/2, (90.74-83.97)/4],
    [2, "F", (82.12+88.76)/2, (88.76-82.12)/4],
    [3, "M", (92.03+99.75)/2, (99.75-92.03)/4],
    [3, "F", (90.41+97.94)/2, (97.94-90.41)/4],
    [4, "M", (98.47+106.93)/2, (106.93-98.47)/4],
    [4, "F", (97.35+105.76)/2, (105.76-97.35)/4],
    [5, "M", (104.72+113.64)/2, (113.64-104.72)/4],
    [5, "F", (103.46+112.43)/2, (112.43-103.46)/4]
], columns=["age", "sex", "median", "sd"])

def z_score(value, median, sd):
    return (value - median) / sd

df = df.merge(bb_table, on=["age", "sex"], how="left")
df["z_bb"] = df.apply(lambda row: z_score(row["weight"], row["median"], row["sd"]), axis=1)
df = df.drop(columns=["median", "sd"])

df = df.merge(tb_table, on=["age", "sex"], how="left")
df["z_tb"] = df.apply(lambda row: z_score(row["height"], row["median"], row["sd"]), axis=1)
df = df.drop(columns=["median", "sd"])

def kategori_gizi(z):
    if z < -3:
        return "Stunted"
    elif -3 <= z < -2:
        return "Berisiko"
    else:
        return "Normal"

df["status_gizi"] = df[["z_bb", "z_tb"]].min(axis=1).apply(kategori_gizi)

print(df["status_gizi"].value_counts())
print(df[["age", "sex", "weight", "height", "z_bb", "z_tb", "status_gizi"]].head())

label_map = {"Normal": 0, "Berisiko": 1, "Stunted": 2}
df["status_gizi_encoded"] = df["status_gizi"].map(label_map)

X = df[["age", "weight", "height"]]
y = df["status_gizi_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy model:", accuracy)

joblib.dump(model, "nutrisense_model.pkl")
print("Model berhasil disimpan!")
