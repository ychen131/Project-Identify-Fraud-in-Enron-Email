# Identify Fraud in Enron Email

## Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.


## Project Goal
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The goal fo this project is to build a person of interest identifer (POI,which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.) based on financial and email data made public as a result of Enron scandal. 

Machine learning is an excellent tool for classification and making predition from data, especially for large data set. In our case, machine learning will be able to use the pattern discovered from the labeled data to infer the new observations.

## Data Exploration
Our data contains both financial information of Enron employees and emails. There are 146 observations and 21 variables (6 email features, 14 financial features and 1 POI label). 18 employees are labeled as "POI" and 128 are non-POIs. 

### Missing Values
While scanning through the data set, we noted some "NaN" within a few observations. The output below shows number of missing values by features. 

```py
salary :  51
to_messages :  60
deferral_payments :  107
total_payments :  21
exercised_stock_options :  44
bonus :  64
restricted_stock :  36
shared_receipt_with_poi :  60
restricted_stock_deferred :  128
total_stock_value :  20
expenses :  51
loan_advances :  142
from_messages :  60
other :  53
from_this_person_to_poi :  60
poi :  0
director_fees :  129
deferred_income :  97
long_term_incentive :  80
email_address :  35
from_poi_to_this_person :  60
```
Apart from variable, "poi", all other features have "NaN" values. They are replaced with value zero before they are analysed by any classifier.


## Feature Selection
## Pick the Algorithm
## Tune the Algorithm
## Validation
## Evaluation

