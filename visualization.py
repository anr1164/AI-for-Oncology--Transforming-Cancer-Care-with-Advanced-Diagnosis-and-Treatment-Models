import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = "D:\\cancer_project\\cancer_dataset_updated.csv"
df = pd.read_csv(file_path)

# Create folder to save visualizations
save_dir = "D:/cancer_project/Cancer_Visualizations"
os.makedirs(save_dir, exist_ok=True)

# Set seaborn style
sns.set(style="whitegrid")

### ---- 1. Cancer Type Distribution (Pie Chart) ----
plt.figure(figsize=(10, 6))
df["CancerType"].value_counts().plot.pie(autopct="%1.1f%%", cmap="coolwarm", startangle=90)
plt.title("Distribution of Different Cancer Types")
plt.ylabel("")
plt.savefig(f"{save_dir}/Cancer_Distribution_Pie.png")

### ---- 2. Gender Distribution ----
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", data=df, palette="coolwarm")
plt.title("Gender Distribution of Cancer Patients")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig(f"{save_dir}/Gender_Distribution.png")

### ---- 3. Age Distribution ----
plt.figure(figsize=(10, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution of Cancer Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig(f"{save_dir}/Age_Distribution.png")

### ---- 4. Cancer Stage Distribution ----
plt.figure(figsize=(8, 5))
sns.countplot(x="Stage", data=df, palette="magma")
plt.title("Distribution of Cancer Stages")
plt.xlabel("Stage")
plt.ylabel("Count")
plt.savefig(f"{save_dir}/Cancer_Stage_Distribution.png")

### ---- 5. Survival Months by Cancer Stage ----
plt.figure(figsize=(10, 6))
sns.boxplot(x="Stage", y="SurvivalMonths", data=df, palette="coolwarm")
plt.title("Survival Months Based on Cancer Stage")
plt.xlabel("Cancer Stage")
plt.ylabel("Survival Months")
plt.savefig(f"{save_dir}/Survival_Stage_Boxplot.png")

### ---- 6. Impact of Treatment Type on Survival ----
plt.figure(figsize=(12, 6))
sns.boxplot(x="TreatmentType", y="SurvivalMonths", data=df, palette="viridis")
plt.xticks(rotation=45)
plt.title("Survival Months for Different Treatment Types")
plt.xlabel("Treatment Type")
plt.ylabel("Survival Months")
plt.savefig(f"{save_dir}/Treatment_Survival_Boxplot.png")

### ---- 7. Metastasis Impact on Survival ----
plt.figure(figsize=(8, 5))
sns.boxplot(x="Metastasis", y="SurvivalMonths", data=df, palette="coolwarm")
plt.title("Impact of Metastasis on Survival")
plt.xlabel("Metastasis")
plt.ylabel("Survival Months")
plt.savefig(f"{save_dir}/Metastasis_Survival.png")

### ---- 8. Smoking and Cancer Correlation ----
plt.figure(figsize=(8, 5))
sns.countplot(x="SmokingHistory", hue="CancerType", data=df, palette="tab10")
plt.title("Smoking History and Cancer Types")
plt.xlabel("Smoking History")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/Smoking_Cancer_Correlation.png")

### ---- 9. Family History and Cancer ----
plt.figure(figsize=(8, 5))
sns.countplot(x="FamilyHistory", hue="CancerType", data=df, palette="tab20")
plt.title("Family History and Cancer Types")
plt.xlabel("Family History")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/FamilyHistory_Cancer.png")

### ---- 10. Alive vs. Died Based on Cancer Type ----
plt.figure(figsize=(12, 6))
sns.countplot(x="CancerType", hue="AliveOrDied", data=df, palette="coolwarm")
plt.title("Survival Status Based on Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/Survival_CancerType.png")

### ---- 11. Obesity and Cancer ----
plt.figure(figsize=(8, 5))
sns.countplot(x="Obesity", hue="CancerType", data=df, palette="mako")
plt.title("Obesity and Cancer Types")
plt.xlabel("Obesity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/Obesity_Cancer.png")

### ---- 12. Cancer Type vs. Age Distribution ----
plt.figure(figsize=(12, 6))
sns.boxplot(x="CancerType", y="Age", data=df, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Age Distribution by Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("Age")
plt.savefig(f"{save_dir}/Cancer_Age_Distribution.png")

### ---- 13. Survival Rate Based on Age Group ----
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
plt.figure(figsize=(10, 6))
sns.countplot(x="AgeGroup", hue="AliveOrDied", data=df, palette="coolwarm")
plt.title("Survival Rate Based on Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.savefig(f"{save_dir}/Survival_AgeGroup.png")

### ---- 14. Cancer Type vs. Number of Treatments Taken ----
plt.figure(figsize=(12, 6))
sns.boxplot(x="CancerType", y="NumberOfTreatments", data=df, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Number of Treatments Taken for Each Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("Number of Treatments")
plt.savefig(f"{save_dir}/Cancer_Treatment_Count.png")

### ---- 15. Stage-wise Distribution of Cancer Types ----
plt.figure(figsize=(12, 6))
sns.countplot(x="Stage", hue="CancerType", data=df, palette="coolwarm")
plt.title("Stage-wise Distribution of Different Cancer Types")
plt.xlabel("Cancer Stage")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/Stagewise_Cancer_Distribution.png")

### ---- 16. Heatmap of Correlations ----
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig(f"{save_dir}/Correlation_Heatmap.png")

# Show message after saving all plots
print(f"All visualizations have been saved in: {save_dir}")
