# Credit-Risk-Analysis-for-Loan-Applicants
## 📍 Background
Financial institutions face critical decisions when processing loan applications:
- **Approving creditworthy applicants** → Drives business growth  
- **Approving risky applicants** → Leads to financial losses  

This project analyzes historical LendingClub loan data containing outcomes for approved applicants:
| Loan Status       | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| ✅ Fully Paid     | Loan repaid successfully                                                   |
| ⏳ Current        | Active loans (excluded from analysis)                                       |
| ⚠️ Charged-off   | Defaulted loans (target risk event)                                        |

> Note: Rejected applications aren't included as they have no repayment history.

## 🎯 Objective
Develop predictive models to:
1. **Reduce credit loss** by identifying high-risk applicants  
2. **Optimize approval rates** for creditworthy borrowers  
3. **Inform risk-based pricing strategies**  
4. **Discover key default drivers** in lending portfolios  

## 📂 Dataset
**Source**: [Lending Club Loan Data on Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)  
**Scope**: 2007-2018 approved loans filtered to 2017-2018 (~225K records)  
**Key Features**:
- Borrower characteristics (FICO, income, employment)  
- Loan details (amount, term, purpose, grade)  
- Financial metrics (DTI, installment payments)  
- Historical data (credit history length)

## 🔄 Pipeline Overview

1. **Ingest & Trim**  
   * Load LendingClub’s full 2007-2018 file (2.2 M rows) and keep/rename the 15 core borrower & loan fields we actually need.  
   * Flag `dti` outliers (< 0 or > 100) and replace with the median

2. **Exploratory Data Analysis (EDA)**  
   * Defaults make up **≈ 21 %** of the sample – a solid but manageable class imbalance.  
   * Typical borrower: \$14 k loan, 13.8 % APR, FICO ≈ 702, DTI ≈ 18 %.  
   * Strongest simple correlations with default are **interest rate**, **low FICO**, and **longer term** (confirmed later by feature importance).

3. **Data Cleaning & Feature Engineering**  
   * Median-impute numeric nulls; convert `emp_length`, `term`, `grade`, `purpose` etc. to one-hot dummies.  
   * Derive **`credit_age`** (years since earliest credit line) , **default** (charged-off -> 1 else 0) 

4. **Train-Test Split**  
   * 80 / 20 stratified on the target, newest vintages reserved for testing to mimic real-world deployment drift.

5. **Modeling & Evaluation**  
   * Baseline **Logistic Regression** 
   * **Random Forest** (300 trees, depth 5) for a quick nonlinear benchmark.  
   * **XGBoost** tuned on 400 trees with `scale_pos_weight` to handle imbalance.
  
## 📊 Key EDA Findings
• Highest risk Grade: G (50.1% default rate)

• Highest risk Home Ownership: RENT (26.4% default rate)

• Highest risk Verification Status: Verified (28.2% default rate)

• Highest risk Purpose: wedding (50.0% default rate)

• Fico Low Range < 700 == High Risk

• Revolving Utilization Rate > 38% == High Risk

• Debt to Income Ratio > 20 == High Risk

  
 ## 🏁 Model Results

| Model | ROC-AUC | Accuracy | Recall (Defaulters) | Notes |
|-------|--------:|---------:|--------------------:|-------|
| Logistic Regression | 0.688 | **0.79** |0.07 | High accuracy but poor at catching defaulters |
| XGBoost (400 × depth 4) | **0.72** | 0.64 | 0.69 | Strongest AUC thanks to clean linear features |
| Random Forest (300 × depth 5) | 0.699 | 0.60 | **0.71** |  Better minority-class recall, lower overall accuracy |

**Top 5 XGBoost Features**  
`int_rate` › `fico_range_low` › `term` › `grade_B` › `grade_D`  






**Built with**: Python, Pandas, Scikit-learn, XGBoost, RandomForest, Logistic Regression  
**Author**: Asmita Khode 
**Contact**: asmitakhode101@gmail.com 

