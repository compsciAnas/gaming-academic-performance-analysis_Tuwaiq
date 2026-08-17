# EDA Project Methodology

## 1. Goal

The objective is to produce an individual Exploratory Data Analysis project that:

- Covers every requirement in the project brief.
- Evaluates and cleans the data using justified decisions.
- Explores meaningful relationships between gaming behavior, lifestyle, and academic performance.
- Communicates findings through a well-organized notebook and a 5–7 minute presentation.
- Avoids unsupported causal conclusions.

## 2. Dataset suitability

The attached `Gaming_Academic_Performance.csv` dataset is suitable for the complete project.

### Dataset structure

- **Rows:** 8,000
- **Columns:** 14
- **Stored numerical columns:** 11, including `student_id`
- **Analytical numerical measures:** 10 after excluding the identifier
- **Categorical columns:** 3
- **Natural target:** `grades`, a continuous numerical variable
- **Row meaning:** Each row appears to represent one student and their gaming behavior, study habits, lifestyle characteristics, and academic performance.

### Variables

| Variable | Analytical role |
|---|---|
| `student_id` | Unique identifier; exclude from numerical analysis |
| `age` | Numerical feature |
| `gender` | Categorical feature |
| `gaming_hours` | Numerical feature |
| `study_hours` | Numerical feature |
| `sleep_hours` | Numerical feature |
| `attendance` | Numerical feature |
| `gaming_genre` | Categorical feature |
| `social_activity` | Numerical feature |
| `device_usage` | Numerical feature |
| `reaction_time_ms` | Numerical feature |
| `addiction_score` | Numerical feature requiring scale validation |
| `stress_level` | Categorical feature |
| `grades` | Continuous target variable |

### Administrative requirement

This dataset is not one of the datasets listed in the project brief. Therefore:

1. Obtain or confirm instructor approval.
2. Locate and cite the original Kaggle or UCI page.
3. Save the data dictionary or column descriptions.
4. Confirm the valid scales for `grades` and `addiction_score`.

## 3. Preliminary data-quality assessment

| Requirement | Result | Recommended decision |
|---|---:|---|
| Missing values | 0 in every column | No imputation |
| Exact duplicate rows | 0 | Keep all records |
| Duplicates excluding `student_id` | 0 | No concealed duplicate profiles |
| Duplicate IDs | 0 | IDs are unique from 1 to 8,000 |
| Parsing/type errors | None found | Apply only semantic type improvements |
| Inconsistent categories | None found | Report the check; do not manufacture cleaning |
| Grades above 100 | 134 rows (1.675%) | Confirm scale, flag, and treat as missing if scale is 0–100 |
| Negative addiction scores | 107 rows (1.337%) | Confirm scale, flag, and treat as missing if minimum is 0 |
| Records affected by either validity issue | 239 | Use analysis-specific filtering |
| IQR outliers | 0 across numerical measures | Keep statistically valid observations |
| Invalid dates | Not applicable | Dataset contains no date columns |
| Class imbalance | Not applicable to continuous `grades` | Explain why rather than forcing classification |

An invalid value is not necessarily a statistical outlier. The grades above 100 and negative addiction scores violate likely domain rules, but they are not detected as outliers by the IQR method.

## 4. Recommended project structure

Use one notebook whose main headings match the eight parts of the project brief:

1. Dataset Understanding
2. Initial Exploration
3. Data Quality Assessment
4. Exploratory Data Analysis
5. Data Visualization
6. Key Insights
7. Business or Stakeholder Recommendations
8. Reflection

Suggested supporting structure:

```text
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── gaming_academic_performance_eda.ipynb
├── presentation/
└── README.md
```

Keep the raw dataset unchanged. Perform all cleaning on a copy.

## 5. Notebook implementation

### 5.1 Imports and data loading

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path(
    "/Users/anas/Desktop/AdvTuwaiq/unit 2 /project/"
    "Gaming_Academic_Performance.csv"
)

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
```

**Expected outcome:** A reproducible starting point where the original data remains unchanged.

**Gotcha:** Never overwrite the raw CSV with cleaned values.

## 6. Part 1 — Dataset Understanding

Explain the dataset in plain language before showing detailed analysis.

### Questions to answer

- What is the purpose of the dataset?
- What does each row represent?
- How many rows and columns are present?
- Which variables are numerical?
- Which variables are categorical?
- Is there a target variable?

### Recommended description

The dataset is intended to investigate associations between students' gaming behavior, study habits, lifestyle factors, and academic performance. Each row represents one student. The natural target is `grades`, which is continuous and makes this a regression-style analytical problem.

Include a data dictionary with:

- Column name
- Description
- Data type
- Unit or scale
- Analytical role

Mark uncertain definitions rather than guessing, especially the scale of `addiction_score`.

**Expected outcome:** The reader understands the data before seeing cleaning or visualizations.

## 7. Part 2 — Initial Exploration

Use:

```python
df.shape
df.columns.tolist()
df.head()
df.info()
df.describe().T
df.describe(include="object").T
df.nunique()
```

### Immediate observations to document

- The dataset has 8,000 rows and 14 columns.
- It is completely populated.
- `student_id` is unique and sequential.
- Categorical features contain only three categories each.
- Most numerical variables have bounded ranges.
- `grades` and `addiction_score` require domain validation.
- The unusually complete and structured data may be synthetic or heavily generated, so its provenance should be documented.

Do not place unexplained Pandas output in the notebook. Follow every important result with a short interpretation.

**Expected outcome:** A concise initial profile and a list of columns requiring further investigation.

## 8. Part 3 — Data Quality Assessment

Create a decision log:

| Issue | Evidence | Decision | Rows affected | Reason |
|---|---|---|---:|---|

This table can be completed as the assessment progresses.

### 8.1 Missing values

```python
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": df.isna().mean().mul(100)
})

missing
```

**Finding:** The original dataset contains no missing values.

**Decision:** Do not impute or delete anything.

**Reason:** Applying an unnecessary treatment would alter valid observations.

### 8.2 Duplicate rows

```python
exact_duplicates = df.duplicated().sum()
duplicates_without_id = df.drop(columns="student_id").duplicated().sum()
duplicate_ids = df["student_id"].duplicated().sum()

exact_duplicates, duplicates_without_id, duplicate_ids
```

**Finding:** All three duplicate checks return zero.

**Decision:** Keep all records.

**Reason:** No evidence supports deleting any row as a duplicate.

### 8.3 Data types

```python
df.dtypes
```

No numerical parsing problems were found. Apply semantic type improvements:

```python
df["student_id"] = df["student_id"].astype("string")

category_columns = ["gender", "gaming_genre", "stress_level"]
df[category_columns] = df[category_columns].astype("category")
```

`student_id` is converted to text because arithmetic operations on an identifier are meaningless. It must also be excluded from correlations and summary statistics.

### 8.4 Invalid values

Start with explicit domain rules:

```python
domain_checks = {
    "age": df["age"].between(15, 100),
    "gaming_hours": df["gaming_hours"].between(0, 24),
    "study_hours": df["study_hours"].between(0, 24),
    "sleep_hours": df["sleep_hours"].between(0, 24),
    "attendance": df["attendance"].between(0, 100),
    "social_activity": df["social_activity"].between(0, 24),
    "device_usage": df["device_usage"].between(0, 24),
    "reaction_time_ms": df["reaction_time_ms"] > 0,
    "addiction_score": df["addiction_score"] >= 0,
    "grades": df["grades"].between(0, 100),
}

for column, valid_mask in domain_checks.items():
    print(column, (~valid_mask).sum())
```

Preserve the raw values and create flags:

```python
df["invalid_grade"] = ~df["grades"].between(0, 100)
df["invalid_addiction"] = df["addiction_score"] < 0

df[["invalid_grade", "invalid_addiction"]].sum()
```

If the data dictionary confirms that grades must be 0–100 and addiction scores cannot be negative:

```python
df.loc[df["invalid_grade"], "grades"] = np.nan
df.loc[df["invalid_addiction"], "addiction_score"] = np.nan
```

This produces:

- 7,866 records available for grade-dependent analysis.
- 7,893 records available for addiction-score analysis.
- 7,761 records available when both values are required.

Use analysis-specific datasets:

```python
df_grade = df.dropna(subset=["grades"])
df_addiction = df.dropna(subset=["addiction_score"])
df_both = df.dropna(subset=["grades", "addiction_score"])
```

**Preferred decision:** Set confirmed invalid values to missing and filter them only when the affected variable is required.

**Gotcha:** Do not automatically cap grades at 100 or addiction scores at zero. Capping creates an artificial concentration at the boundary and is justified only if the source documentation supports it.

### 8.5 Inconsistent categories

Check raw and normalized forms:

```python
for column in category_columns:
    raw_values = df[column].astype("string")
    normalized = raw_values.str.strip().str.casefold()

    print(f"\nColumn: {column}")
    print(raw_values.value_counts(dropna=False))
    print("Raw unique values:", raw_values.nunique(dropna=False))
    print("Normalized unique values:", normalized.nunique(dropna=False))
```

**Finding:** No capitalization or whitespace inconsistencies were found.

**Decision:** Do not modify the categories.

**Reason:** Cleaning should be evidence-based.

### 8.6 Outliers

Exclude `student_id` from outlier analysis:

```python
numeric_columns = [
    "age",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "grades",
]

outlier_summary = []

for column in numeric_columns:
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    outlier_summary.append({
        "column": column,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_mask.sum(),
        "outlier_pct": outlier_mask.mean() * 100,
    })

outlier_summary = pd.DataFrame(outlier_summary)
outlier_summary
```

**Finding:** The IQR method finds no outliers in the numerical variables.

**Decision:** Keep all statistically valid observations.

**Interpretation:** The variables have relatively broad, bounded distributions. The absence of IQR outliers is a legitimate result.

### 8.7 Class imbalance

Use the following explanation:

> The original target, `grades`, is continuous. This makes the project a regression-style analysis, so class imbalance is not applicable. Class imbalance would become relevant only if grades were converted into derived categories such as Fail, Pass, Good, and Excellent.

Do not convert grades into categories only to force an imbalance analysis. Grade bands may still be created for descriptive visualizations if they are clearly identified as derived variables.

## 9. Validate the cleaned data

After cleaning, repeat the important checks:

```python
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicates:", df.duplicated().sum())

print("\nGrade range:")
print(df["grades"].min(), df["grades"].max())

print("\nAddiction score range:")
print(df["addiction_score"].min(), df["addiction_score"].max())

print("\nCategories:")
for column in category_columns:
    print(column, df[column].unique())
```

Document the difference between:

- Original missingness: zero.
- Missing values deliberately introduced when invalid observations were marked.

## 10. Part 4 — Exploratory Data Analysis

Follow the sequence:

1. Univariate numerical analysis.
2. Univariate categorical analysis.
3. Bivariate analysis.
4. Multivariate analysis.

### 10.1 Numerical analysis

```python
df_grade[numeric_columns].describe().T
```

For important variables, report:

- Mean
- Median
- Standard deviation
- Minimum and maximum
- Quartiles
- Distribution shape

Focus on:

- `grades`
- `gaming_hours`
- `study_hours`
- `sleep_hours`
- `attendance`
- `device_usage`
- `addiction_score`

### 10.2 Categorical analysis

Produce counts and percentages:

```python
for column in category_columns:
    counts = df[column].value_counts(dropna=False)
    percentages = df[column].value_counts(
        normalize=True,
        dropna=False
    ).mul(100)

    display(pd.DataFrame({
        "count": counts,
        "percentage": percentages
    }))
```

Then compare academic performance:

```python
for column in category_columns:
    display(
        df_grade.groupby(column, observed=True)["grades"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
```

### 10.3 Relationship analysis

Recommended relationships:

- Study hours and grades
- Gaming hours and grades
- Sleep hours and grades
- Attendance and grades
- Device usage and grades
- Addiction score and grades
- Reaction time and grades
- Grades by gender
- Grades by gaming genre
- Grades by stress level

Correlation analysis:

```python
analysis_columns = [
    "age",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "grades",
]

correlation_matrix = df[analysis_columns].corr()
correlation_matrix.round(3)
```

Preliminary raw-data correlations indicate that the dataset has enough analytical content:

| Relationship with grades | Approximate correlation |
|---|---:|
| Study hours | +0.73 |
| Gaming hours | −0.55 |
| Reaction time | +0.52 |
| Addiction score | −0.50 |
| Device usage | −0.47 |
| Sleep hours | +0.25 |
| Attendance | +0.13 |

Recalculate these after cleaning before using them as final results.

### Multicollinearity caution

Gaming hours are strongly related to device usage, addiction score, and reaction time. These variables should not automatically be interpreted as separate independent effects.

Correlation should also be described as association, not causation.

## 11. Part 5 — Data Visualization

Create six or seven polished charts so the project exceeds the minimum requirement.

### Visualization 1 — Distribution of grades

```python
plt.figure(figsize=(9, 5))
sns.histplot(df_grade["grades"], bins=30, kde=True)
plt.title("Distribution of Valid Student Grades")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.show()
```

Purpose: Understand overall academic performance, distribution shape, and concentration.

### Visualization 2 — Grades by stress level

```python
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_grade,
    x="stress_level",
    y="grades",
    order=["Low", "Medium", "High"]
)
plt.title("Grade Distribution by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Grade")
plt.show()
```

Purpose: Compare grade distributions rather than means alone.

### Visualization 3 — Study hours and grades

```python
plt.figure(figsize=(9, 6))
sns.regplot(
    data=df_grade,
    x="study_hours",
    y="grades",
    scatter_kws={"alpha": 0.25},
    line_kws={"color": "red"}
)
plt.title("Study Hours and Academic Grades")
plt.xlabel("Study Hours")
plt.ylabel("Grade")
plt.show()
```

Purpose: Examine whether students with more study time tend to have higher grades.

### Visualization 4 — Gaming hours and grades

```python
plt.figure(figsize=(9, 6))
sns.regplot(
    data=df_grade,
    x="gaming_hours",
    y="grades",
    scatter_kws={"alpha": 0.25},
    line_kws={"color": "red"}
)
plt.title("Gaming Hours and Academic Grades")
plt.xlabel("Gaming Hours")
plt.ylabel("Grade")
plt.show()
```

Purpose: Examine the association between gaming time and academic performance.

### Visualization 5 — Correlation heatmap

```python
plt.figure(figsize=(11, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Matrix of Numerical Variables")
plt.show()
```

Purpose: Identify the strongest relationships and potential multicollinearity.

### Visualization 6 — Average grades by gaming genre

```python
genre_grades = (
    df_grade.groupby("gaming_genre", observed=True)["grades"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=genre_grades,
    x="gaming_genre",
    y="grades"
)
plt.title("Average Grade by Gaming Genre")
plt.xlabel("Gaming Genre")
plt.ylabel("Average Grade")
plt.ylim(0, 100)
plt.show()
```

Purpose: Determine whether the genre groups have meaningful differences.

### Visualization 7 — Device usage and grades

```python
plt.figure(figsize=(9, 6))
sns.regplot(
    data=df_grade,
    x="device_usage",
    y="grades",
    scatter_kws={"alpha": 0.25},
    line_kws={"color": "red"}
)
plt.title("Device Usage and Academic Grades")
plt.xlabel("Device Usage")
plt.ylabel("Grade")
plt.show()
```

Purpose: Explore the relationship between overall device usage and performance.

### Interpretation template

After every chart, explain:

1. **What it shows:** The variables and visible pattern.
2. **What can be concluded:** The finding supported by the chart.
3. **Why it is useful:** Its relevance to students, educators, or decision-makers.

Use language such as:

- “is associated with”
- “students with higher values tend to”
- “the groups differ descriptively”

Avoid:

- “causes”
- “leads to”
- “proves”

## 12. Part 6 — Key Insights

Write at least 10 insights using:

> Finding → interpretation → stakeholder meaning or limitation

Possible insight themes:

1. Study time and academic performance.
2. Gaming time and academic performance.
3. Sleep and academic performance.
4. Attendance and academic performance.
5. Device usage and gaming behavior.
6. Addiction score and gaming time.
7. Reaction time relationships.
8. Gender comparisons.
9. Gaming genre comparisons.
10. Stress-level patterns.
11. The limited role of age or social activity.
12. Data-quality limitations.
13. Relationships among gaming-related variables.

Do not write insights that merely restate a mean or percentage.

Weak:

> The average grade is 66.

Stronger:

> Grades are concentrated around the middle of the scale, indicating that a substantial portion of the sample may benefit from academic support rather than only interventions focused on the very lowest-performing students.

## 13. Part 7 — Recommendations

Provide at least five recommendations. Each recommendation must link to an insight or visualization.

Potential recommendations, subject to the final results:

1. Encourage balanced study and gaming schedules.
2. Identify students combining high gaming or device use with low study time.
3. Promote consistent sleep and class attendance.
4. Offer counseling or support when gaming patterns align with problematic behavior.
5. Add data-entry validation preventing grades above 100 and negative addiction scores.
6. Improve the addiction score definition and collection method.
7. Collect longitudinal data across multiple semesters before claiming that gaming changes grades.
8. Avoid interventions based only on gender or gaming genre if differences are small.

Phrase these as stakeholder recommendations for:

- Students
- Educators
- Academic advisors
- School counselors
- Data collection teams

## 14. Part 8 — Reflection

Answer the required questions in the student's own words:

1. What was the most challenging data-quality problem?
2. Which decision required the most critical thinking?
3. What was the most interesting finding?
4. What would be done differently next time?
5. What was learned about EDA?

Strong reflection themes:

- Distinguishing invalid values from statistical outliers.
- Deciding not to invent missing-value or duplicate treatments.
- Handling invalid target values without unnecessarily deleting entire rows.
- Avoiding causal conclusions from correlations.
- Recognizing that cleaning quality is measured through justification, not the number of transformations.

## 15. Presentation methodology

Use approximately six slides:

| Slide | Content | Suggested time |
|---|---|---:|
| 1 | Research question and dataset | 40 seconds |
| 2 | Dataset structure and target | 40 seconds |
| 3 | Data-quality findings and decisions | 75 seconds |
| 4 | Three strongest visualizations | 2 minutes |
| 5 | Main insights and recommendations | 90 seconds |
| 6 | Limitations and reflection | 45 seconds |

Do not display every notebook output. Present the three most decision-relevant charts and summarize the remaining findings.

## 16. Final quality checklist

Before submission:

- [ ] Instructor approval is confirmed.
- [ ] Original dataset URL and data dictionary are cited.
- [ ] Notebook headings match Parts 1–8.
- [ ] Every data-quality check has evidence and a decision.
- [ ] “No issue found” results are documented.
- [ ] The raw dataset remains unchanged.
- [ ] `student_id` is excluded from numerical analysis.
- [ ] Invalid values are flagged before treatment.
- [ ] Cleaning decisions include explanations.
- [ ] Cleaned data is validated.
- [ ] At least five meaningful charts are included.
- [ ] Every visualization has a written interpretation.
- [ ] At least 10 meaningful insights are included.
- [ ] At least five evidence-supported recommendations are included.
- [ ] Correlation is not described as causation.
- [ ] Code cells contain appropriate comments.
- [ ] The notebook runs from beginning to end without errors.
- [ ] The presentation fits within 5–7 minutes.

## 17. Final verdict

The dataset can support all eight project parts. Its main weakness is that it is very clean, but that does not make it unsuitable. The correct professional approach is to prove that missing values, duplicates, parsing errors, inconsistent categories, and IQR outliers were checked and not found.

The two main quality issues are the 134 grades above 100 and 107 negative addiction scores. Their treatment must depend on the definitions in the original data source. Confirm instructor approval and the valid measurement scales before implementing the final cleaning decisions.
