# --------------------------------------------
# Big Startup EDA Project | Crunchbase Dataset
# By: David Oladipupo
# --------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv(r"C:\Users\dell\Downloads\big_startup_secsees_dataset.csv\big_startup_secsees_dataset.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# --------------------------------------------
# 2. Clean Funding Column (convert to numeric)
# --------------------------------------------
df['funding_total_usd'] = pd.to_numeric(df["funding_total_usd"], errors='coerce')

# --------------------------------------------
# 3. Convert Date Columns & Feature Engineering
# --------------------------------------------
df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['last_funding_at'] = pd.to_datetime(df['last_funding_at'], errors='coerce')

# Extract founding year
df['founded_year'] = df['founded_at'].dt.year

# Calculate lifespan in years
df['Lifespan_Years'] = (df['last_funding_at'] - df['founded_at']).dt.days / 365.25

# Create binary indicators
df['Is_Operating'] = df['status'].apply(lambda x: 1 if x == 'operating' else 0)
df['Is_IPO'] = df['status'].apply(lambda x: 1 if x == 'ipo' else 0)

# --------------------------------------------
# 4. Explore Missing Data
# --------------------------------------------
missing_data = df.isnull().sum().sort_values(ascending=False)
print("\nMissing Data:\n", missing_data[missing_data > 0])

# --------------------------------------------
# 5. Trend: Startups Founded Per Year
# --------------------------------------------
yearly_count = df['founded_year'].value_counts().sort_index()
plt.figure(figsize=(12,6))
sns.lineplot(x=yearly_count.index, y=yearly_count.values)
plt.title("Number of Startups Founded by Year")
plt.xlabel('Year')
plt.ylabel('Number of Startups')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 6. Startup Outcome Over Time
# --------------------------------------------
year_status = df.groupby(['founded_year', 'status']).size().unstack().fillna(0)
year_status.plot(kind='bar', stacked=True, figsize=(14,6), colormap='tab20')
plt.title("Startup Outcomes by Founded Year")
plt.xlabel("Founded Year")
plt.ylabel("Number of Startups")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 7. Hypothesis 1: Funding vs Status
# --------------------------------------------
operating_funding = df[df['status'] == 'operating']['funding_total_usd'].dropna()
closed_funding = df[df['status'] == 'closed']['funding_total_usd'].dropna()

# t-test
t_stat, p_value = ttest_ind(operating_funding, closed_funding, equal_var=False)
print(f"\n[Funding vs Status] T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")

# --------------------------------------------
# 8. Hypothesis 2: Lifespan vs Status
# --------------------------------------------
operating_life = df[df['status'] == 'operating']['Lifespan_Years'].dropna()
closed_life = df[df['status'] == 'closed']['Lifespan_Years'].dropna()

# t-test
t_stat2, p_value2 = ttest_ind(operating_life, closed_life, equal_var=False)
print(f"[Lifespan vs Status] T-statistic: {t_stat2:.2f}, P-value: {p_value2:.4f}")

# --------------------------------------------
# 9. Hypothesis 3: Industry vs Status
# --------------------------------------------
industry_status_ct = pd.crosstab(df['category_list'], df['status'])
top_industries = df['category_list'].value_counts().head(10).index
filtered_ct = industry_status_ct.loc[top_industries]

# Chi-square test
chi2, p_val, dof, expected = chi2_contingency(filtered_ct)
print(f"[Industry vs Status] Chi2: {chi2:.2f}, P-value: {p_val:.4f}")
