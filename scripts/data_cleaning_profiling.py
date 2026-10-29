"""
Data Cleaning and Profiling
Employee Turnover Dataset

Purpose:
    Profile the raw employee turnover dataset, detect data quality issues
    (duplicates, missing values, inconsistent entries, formatting errors,
    outliers), clean them, and export a cleaned CSV for future analysis.

How to run (from the project root folder in a VS Code terminal):
    python scripts/data_cleaning_profiling.py

Folder layout this script expects:
    project_root/
        data/employee_turnover.csv             <- raw input (unchanged)
        data/employee_turnover_cleaned.csv     <- cleaned CSV is written here
        images/                                <- boxplots are written here
        scripts/data_cleaning_profiling.py                   <- this script

References used for the code (in-text citations appear in comments below):
    (pandas development team, 2024) pandas documentation. https://pandas.pydata.org/docs/
    (Harris et al., 2020) NumPy. https://numpy.org/doc/stable/
    (Hunter, 2007) Matplotlib. https://matplotlib.org/stable/
    (Waskom, 2021) seaborn. https://seaborn.pydata.org/
    (Python Software Foundation, 2024) pathlib. https://docs.python.org/3/library/pathlib.html
"""

# ----------------------------------------------------------------------------
# 0. Imports and setup
# ----------------------------------------------------------------------------
import sys
import platform
from pathlib import Path  

import numpy as np                
import pandas as pd                
import matplotlib
matplotlib.use("Agg")              
import matplotlib.pyplot as plt    
import seaborn as sns              

# Show wide tables in the terminal so every column prints on one line
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# Build paths relative to this file 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "employee_turnover.csv"
CLEAN_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "images"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def section(title):
    """Print a clear heading so each step is easy to follow in the terminal/video."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# Identify the programming environment version 
section("PROGRAMMING ENVIRONMENT")
print(f"Python version : {platform.python_version()}")
print(f"pandas version : {pd.__version__}")
print(f"NumPy version  : {np.__version__}")
print(f"Matplotlib     : {matplotlib.__version__}")
print(f"seaborn        : {sns.__version__}")

# ----------------------------------------------------------------------------
# 1. Load the raw data  (records and variables)
# ----------------------------------------------------------------------------
# read_csv loads the file into a DataFrame 
df_raw = pd.read_csv(RAW_PATH)
df = df_raw.copy()  

section("PROFILE: RECORDS AND VARIABLES")
print(f"Number of records (rows):    {df.shape[0]}")
print(f"Number of variables (cols):  {df.shape[1]}")

# ----------------------------------------------------------------------------
# 2. Profile each variable  (data types and sample values)
# ----------------------------------------------------------------------------
section("PROFILE: DATA TYPES AND SAMPLE OBSERVABLE VALUES")
print(df.dtypes)  # pandas storage type of each column
print("\nFirst five records:")
print(df.head())

print("\nSample of observable (unique) values for each variable:")
for col in df.columns:
    # .unique() lists distinct values; show the first 6 as a sample
    sample = df[col].dropna().unique()[:6]
    print(f"  {col!r:32} -> {list(sample)}")

# ============================================================================
# DETECT DATA QUALITY ISSUES
# ============================================================================

# ----------------------------------------------------------------------------
# Detect 1. Formatting errors
# ----------------------------------------------------------------------------
section("DETECT: FORMATTING ERRORS")
# Column names: repr() exposes hidden leading/trailing spaces
print("Column names with hidden whitespace:",
      [repr(c) for c in df.columns if c != c.strip()])

# HourlyRate is stored as text with a '$' sign and trailing spaces,
# so it cannot be used in calculations until it is converted to a number.
print("HourlyRate dtype:", df_raw["HourlyRate "].dtype)
print("HourlyRate sample:", df_raw["HourlyRate "].head(5).tolist())

# Columns stored as float only because NaN is present (should be whole numbers)
for col in ["NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"]:
    print(f"{col} dtype: {df[col].dtype}  (whole-number counts stored as float)")

# ----------------------------------------------------------------------------
# Detect 2. Duplicate entries
# ----------------------------------------------------------------------------
section("DETECT: DUPLICATE ENTRIES")
# duplicated() flags rows that exactly match an earlier row
full_dupes = df.duplicated().sum()
id_dupes = df["EmployeeNumber"].duplicated().sum()
print(f"Exact duplicate rows:              {full_dupes}")
print(f"Duplicate EmployeeNumber values:   {id_dupes}")
print("Example duplicated records (keep=False shows both copies):")
print(df[df.duplicated(keep=False)].sort_values("EmployeeNumber").head(4))

# ----------------------------------------------------------------------------
# Detect 3. Missing values
# ----------------------------------------------------------------------------
section("DETECT: MISSING VALUES")
# isna().sum() counts missing values per column 
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"missing": missing, "percent": missing_pct})
      .query("missing > 0"))

# ----------------------------------------------------------------------------
# Detect 4. Inconsistent entries
# ----------------------------------------------------------------------------
section("DETECT: INCONSISTENT ENTRIES")
# value_counts() reveals spelling variants of the same category
for col in df.select_dtypes(include=["object", "string"]).columns:
    if col.strip() == "HourlyRate":
        continue  # numeric column stored as text; handled under formatting
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

# Constant columns (only one value) carry no information but are kept
print("\nConstant columns:",
      [c for c in df.columns if df[c].nunique(dropna=True) == 1])

# Logical consistency check: AnnualSalary should equal HourlyRate x HoursWeekly x 52
hourly_check = pd.to_numeric(
    df["HourlyRate "].str.replace(r"[\$,\s]", "", regex=True))
expected_salary = (hourly_check * df["HoursWeekly"] * 52).round(2)
salary_mismatch = (df["AnnualSalary"] - expected_salary).abs() > 1
print(f"\nAnnualSalary not equal to HourlyRate x HoursWeekly x 52: "
      f"{salary_mismatch.sum()} rows")
print(f"Negative AnnualSalary values: {(df['AnnualSalary'] < 0).sum()}")
print(f"Negative DrivingCommuterDistance values: "
      f"{(df['DrivingCommuterDistance'] < 0).sum()}")

# ----------------------------------------------------------------------------
# Detect 5. Outliers
# ----------------------------------------------------------------------------
section("DETECT: OUTLIERS (IQR METHOD)")
numeric_cols = ["Age", "Tenure", "AnnualSalary", "DrivingCommuterDistance",
                "NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"]


def iqr_bounds(series):
    """Return lower/upper fences: Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    q1, q3 = series.quantile([0.25, 0.75])  
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


print(df[numeric_cols].describe().T)  # summary statistics
for col in numeric_cols:
    low, high = iqr_bounds(df[col].dropna())
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col:30} fences [{low:,.2f}, {high:,.2f}]  outliers: {n_out}")

# Boxplots of the raw numeric variables 
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(x=df[col], ax=ax, color="#7aa6c2")
    ax.set_title(f"{col} (raw)")
plt.tight_layout()
plt.savefig(FIG_DIR / "boxplots_raw.png", dpi=120)
plt.close()
print(f"\nSaved raw boxplots to {FIG_DIR / 'boxplots_raw.png'}")

# ============================================================================
# CLEAN THE DATA
# ============================================================================

# ----------------------------------------------------------------------------
# Clean 1. Fix formatting errors
# ----------------------------------------------------------------------------
section("CLEAN: FIXING FORMATTING ERRORS")
# Strip whitespace from column names ('HourlyRate ' -> 'HourlyRate')
df.columns = df.columns.str.strip()

# Remove '$', commas, and spaces, then convert HourlyRate to a float
# str.replace with regex + to_numeric 
df["HourlyRate"] = pd.to_numeric(
    df["HourlyRate"].str.replace(r"[\$,\s]", "", regex=True), errors="coerce")
print("HourlyRate dtype after cleaning:", df["HourlyRate"].dtype)

# Strip stray whitespace from every text column
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip()

# ----------------------------------------------------------------------------
# Clean 2. Remove duplicate entries
# ----------------------------------------------------------------------------
section("CLEAN: REMOVING DUPLICATES")
before = len(df)
# keep="first" keeps the original record and drops the repeated copy
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"Rows removed: {before - len(df)}   Rows remaining: {len(df)}")
print("Duplicate EmployeeNumbers remaining:",
      df["EmployeeNumber"].duplicated().sum())

# ----------------------------------------------------------------------------
# Clean 3. Standardize inconsistent categorical entries
# ----------------------------------------------------------------------------
section("CLEAN: STANDARDIZING INCONSISTENT ENTRIES")
# map() replaces each spelling variant with one standard label
job_map = {
    "InformationTechnology": "Information Technology",
    "Information_Technology": "Information Technology",
    "HumanResources": "Human Resources",
    "Human_Resources": "Human Resources",
}
df["JobRoleArea"] = df["JobRoleArea"].replace(job_map)

paycheck_map = {
    "Mail Check": "Mail Check",
    "Mailed Check": "Mail Check",
    "Mail_Check": "Mail Check",
    "MailedCheck": "Mail Check",
    "Direct Deposit": "Direct Deposit",
    "DirectDeposit": "Direct Deposit",
    "Direct_Deposit": "Direct Deposit",
}
df["PaycheckMethod"] = df["PaycheckMethod"].map(paycheck_map)

print(df["JobRoleArea"].value_counts().to_string())
print()
print(df["PaycheckMethod"].value_counts().to_string())

# Recalculate AnnualSalary from HourlyRate so the two columns agree.
# HoursWeekly is 40 for every employee and pay is annualized over 52 weeks.
# This corrects negative salaries and data-entry errors 
df["AnnualSalary"] = (df["HourlyRate"] * df["HoursWeekly"] * 52).round(2)
print(f"\nAnnualSalary recalculated; negatives remaining: "
      f"{(df['AnnualSalary'] < 0).sum()}")

# ----------------------------------------------------------------------------
# Clean 4. Handle outliers / impossible values in DrivingCommuterDistance
# ----------------------------------------------------------------------------
section("CLEAN: TREATING OUTLIERS")
# Step 1: A distance cannot be negative. Small negatives (-1 to -15) look like
# sign-entry errors, so take the absolute value 
df["DrivingCommuterDistance"] = df["DrivingCommuterDistance"].abs()

# Step 2: Values far outside the IQR fences (125, 250, 275, 322, 910, 950)
# are not realistic daily commutes. Replace them with NaN, then impute the
# median so no employee records are lost.
low, high = iqr_bounds(df["DrivingCommuterDistance"])
outlier_mask = (df["DrivingCommuterDistance"] < low) | \
               (df["DrivingCommuterDistance"] > high)
print(f"Commute IQR fences after abs(): [{low:.2f}, {high:.2f}]")
print(f"Commute outliers replaced: {outlier_mask.sum()}")
print("Outlier values found:",
      sorted(df.loc[outlier_mask, "DrivingCommuterDistance"].unique().tolist()))
df.loc[outlier_mask, "DrivingCommuterDistance"] = np.nan
commute_median = df["DrivingCommuterDistance"].median()
df["DrivingCommuterDistance"] = df["DrivingCommuterDistance"].fillna(commute_median)
print(f"Imputed commute median: {commute_median}")

# AnnualSalary outliers flagged by IQR are now valid high earners (consistent
# with HourlyRate), so they are retained rather than removed.
low_s, high_s = iqr_bounds(df["AnnualSalary"])
print(f"AnnualSalary outliers after recalculation: "
      f"{((df['AnnualSalary'] < low_s) | (df['AnnualSalary'] > high_s)).sum()}")

# ----------------------------------------------------------------------------
# Clean 5. Handle missing values
# ----------------------------------------------------------------------------
section("CLEAN: IMPUTING MISSING VALUES")
# Discrete numeric counts -> median (robust to skew), then store as integers
for col in ["NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"]:
    med = df[col].median()
    df[col] = df[col].fillna(med).astype(int)  
    print(f"{col}: filled with median {med}")

# Categorical opt-in -> explicit 'Unknown' so we do not invent a Yes/No answer
df["TextMessageOptIn"] = df["TextMessageOptIn"].fillna("Unknown")
print("TextMessageOptIn: filled with 'Unknown'")
print(df["TextMessageOptIn"].value_counts().to_string())

# Make the commute column an integer again (it became float when NaN was used)
df["DrivingCommuterDistance"] = df["DrivingCommuterDistance"].round().astype(int)

# ============================================================================
# VERIFY THE CLEANED DATA
# ============================================================================
section("VERIFICATION OF CLEANED DATA")
print(f"Shape: {df.shape}  (all {df_raw.shape[1]} variables retained)")
print(f"Missing values remaining:   {df.isna().sum().sum()}")
print(f"Duplicate rows remaining:   {df.duplicated().sum()}")
print(f"Negative commutes/salaries: "
      f"{(df['DrivingCommuterDistance'] < 0).sum()} / "
      f"{(df['AnnualSalary'] < 0).sum()}")
print(df.dtypes)
print(df.describe().T)

# Boxplots after cleaning for a before/after comparison
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(x=df[col], ax=ax, color="#8cc28c")
    ax.set_title(f"{col} (cleaned)")
plt.tight_layout()
plt.savefig(FIG_DIR / "boxplots_cleaned.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# Export the cleaned dataset
# ----------------------------------------------------------------------------
out_path = CLEAN_DIR / "employee_turnover_cleaned.csv"
df.to_csv(out_path, index=False)  # index=False avoids an extra index column
section("DONE")
print(f"Cleaned dataset saved to: {out_path}")
print(f"Figures saved to:         {FIG_DIR}")
