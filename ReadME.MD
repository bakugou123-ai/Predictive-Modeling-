The following project aims to predict the employee churn for a company dataset with 15,000 entries and 10 features.

Following are the feature column names :

satisfaction_level : Satisfaction level of the employees (numeric and continuous)
last_evaluation : Last evaluation scores of the employees (numeric and continuous)
number_project : The number of projects assigned to the employees (numeric)
average_montly_hours : Average monthly working hours completed by each employee (numeric)
time_spend_company : Total service years (numeric)
Work_accident : Whether the employee faced work accident (numeric and binomial)
left : If the employee left the company. (numeric and binomial)
promotion_last_5years : If the employee was promoted in the last 5 years (numeric and binomial)
Department : Department of each employee (Categorical)
salary : Type of salary (Categorical)

The project follows the PACE workflow (Plan, Analyze, Construct, Evaluate) to ensure seamless working.

Phase 1 : Plan
It starts with asking questions such as :
What are the stakeholder's requirements?
What kind of model to be used?
What should be the predictor variable ?
What are the features that will support the model?

It is necessary to collect descriptive data of the dataset we are working with. A descriptive analysis was then conducted to check for missing and duplicated data. It also helps in checking the range of each column and statistics such as mean, count, quartile ranges, etc. The phase also helped with spotting outliers in each column.

Phase 2 : Analyze
This phase involved Exploratory Data Analysis between the variables (columns) of the dataset.
The exercise helps us understand the split of data in the predictor variable (column : left).
Necessary visualizations were created to check the collinearity of variables and other insights that can help us provide recommendations to stakeholders.

Phase 3 : Construct
The phase involves construction of the model . It helps us dive into the following questions:
What should be our predictor variable?
What are the columns that will contribute to our result?
What model to choose?
Which model will give us the best result?

As the predictor variable is continuous we first construct a Logistic Regression model for our predictive analysis.

Logistic Regression assumptions :
Outcome variable is categorical
Observations are independent of each other
No severe multicollinearity among X variables
No extreme outliers
Linear relationship between each X variable and the logit of the outcome variable
Sufficiently large sample size

Outcome :
The model was evaluated on various parameters.
Precision score : 79%
Recall : 82%
f1-score : 80%
Accuracy : 82%

We then construct two tree based models : Decision Tree and Random Forest.
Decision tree is evaluated and gives its best score of 97%

Random forest is evaluated and gives a score of 98%

Data Leakage :
It is reasonable to be cautious about the unusually high accuracy results, as this may indicate the presence of data leakage. Data leakage occurs when information from the training dataset unintentionally appears in the test dataset.

In our case, satisfaction_level and average_monthly_hours may be contributing to this issue. An employee with a low evaluation score may already be on the verge of leaving, which the model could be learning directly. Similarly, average_monthly_hours may decrease when an employee is planning to quit and becomes disengaged from their manager, making this information a strong but potentially leaked predictor.

In order to avoid this, we perform feature engineering and improve model performance.
The best score that the model achieved was the ROC_AUC score of 97% pre-feature engineering and 98% post-feature engineering.

Recommendations to stakeholders :

Cap the number of projects that employees can work on.

Consider promoting employees who have a tenure of more than 4 years and have worked more than 175 hours per month on average.

Try and fix the low to medium salary range proporations as it is highly correlated with the satisfaction levels.

High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort

As per the confusion matrix, the model predicts of 15% employee churn
