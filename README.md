# Practical Application III: Apply and Compare 6 Common Classifiers to Predict Subscription/Direct Deposit Outcomes from Banking Marketing Campaigns 

## Overview/Learning Goal: 

In this practical application, the goal is to compare the performance of six classifiers: K Nearest Neighbor, Logistic Regression, Decision Trees, Support Vector Machines, and Random Forest. 
## Methodology: CRISP-DM Framework 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/26a9f4ba-b87d-4319-ba53-fb00675ec3eb) 


## Step#1: Business Understandings: Context, Goal and Benefits 

In today's competitive market, optimizing the efficiency and effectiveness of direct marketing campaigns is crucial for increasing customer engagement and maximizing returns on business investment.

Business/ML Goal: Use ML techniques to predict which clients are most likely to subscribe to a deposit after being contacted.

### 1.1 Highlighted Business Benefits:

    1) Targeted Campaigns and Resource Optimization: Machine learning predicts high-potential leads with high precision and recall, enabling us to focus marketing efforts and resources on the most promising clients. This targeted approach enhances the 
       effectiveness and efficiency of our campaigns. 
       
    2) Improved Customer Engagement: By identifying key client characteristics, we can personalize marketing messages and offers, leading to higher engagement and satisfaction. Tailored strategies, such as specific offers for retirees, improve customer 
       interactions and outcomes. 
       
     
By leveraging machine learning and the CRISP-DM methodology, we can transform our marketing campaigns from broad, generalized efforts to highly targeted and efficient operations, even highly personalized campaigns. This boosts our successful bank product subscribers and ensures optimal resource usage. 


## Step#2: Data Understandings: Backgound, Data Dictonoary, Quality Check, and EDA 


### 2.1 Background: 

    The dataset comes from the UCI Machine Learning repository, collected from a Portuguese banking institution based on a collection of marketing campaign results using their contact center. The primary marketing channel was telephone calls by human agents, 
    sometimes supplemented by online banking via the Internet. Each campaign was managed integrally, with combined results from all channels. 

    The dataset encompasses 17 campaigns from May 2008 to November 2010, totaling 41,188 contacts. These campaigns promoted a long-term deposit application with attractive interest rates. For each contact, numerous attributes were recorded, including whether the 
    campaign was successful as the target variable. Out of the entire dataset, there were 6,499 successful subscriptions, resulting in an 11% success rate. 

### 2.2 Data Dictionary: 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/e95d693c-385e-41ee-9cdf-797b3bffbf47)


### 2.3 Data Quality Quick Check: 

   1) 0 null/missing values
   2) 12 duplicates
   3) unbalanced dataset (11% of Success Rate)
   
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/2498bc86-9725-4d65-be43-3874b9fe068d) 


### 2.4 Exploratory Data Anlaysis (EDA) 

2.4.1 Catogrical Variable and Response Variables.

   Key Findings: Job, education, contact, previous outcome, and month appear to be more influential in determining the outcome. 

   Non-significant Features: Loan, housing, and day of the week do not significantly contribute to the subscription success.

   Feature Engineering: Encode all categorical variables for basic models. 

   
    
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/4d07a0f4-a351-45c5-9ed7-aa2e49cd1d9b) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b833d6bf-625f-431f-bda2-5816ce588909) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/85c30490-8156-4c8a-8289-cfa2c32f5842)  
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a4fb81eb-7235-4e11-9483-d9c7ffaeaf2e) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/ad89503c-3ebe-4a40-a6c1-29f35530ee39) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/d5596456-0f64-4319-b402-e4f7fec9f7f3) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/e915aed8-7287-4967-a830-a205f1fccf95) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b87e8b89-4f86-4a18-906e-2f726900d0c4)

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/d26270bc-94c7-4bde-a34a-9e07a585cfba)

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9434a7fd-bbea-4007-8df2-d1ee89509509)

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9260ba25-a3a6-48e6-83bf-daec649f60b1)
   



2.4.2 Numerical Variable and Response Variables.
   
2.4.2.1 Group Histograms

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/bbd8bdbc-df34-4a07-908f-78b4bd476156) 

2.4.2.2 Group Statistics

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/86cfbedc-a94f-4e68-824b-b0fa1b430dd7) 

2.4.3 Group Boxplots

2.4.4 Individual Histogram with Boxplot
   
   Highly Informative Features: Call duration, pdays, previous contacts, employment variation rate, and Euribor rate are strong indicators for predicting positive responses (see the titles of the charts below for detailed findings)
   Moderately Informative Features: Campaign contacts, age, and consumer confidence index provide additional insights. 
    
   Economic Indicators: Negative employment variation rates and lower Euribor rates are associated with "yes" responses, indicating less favorable economic conditions may increase receptiveness. 
   
   Customer Contact Patterns: Effective campaigns involve fewer, more targeted contacts and longer call durations. 
    

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/df4238af-28b7-46e3-9b77-27e344d43f44) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/3310e747-e38c-4de1-9d84-1791e75f7b02) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/ef850449-a748-41a4-8e1f-2e4a0394ace5) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/46e94d94-4842-4c3a-b961-a54527eaa63e) 

    
2.4.5 T-Tests 


It determines whether the means of numerical variables are significantly different between the groups (e.g., "yes" vs. "no"). By doing so, I can understand which features are potentially influential in predicting the target variable. Since significant features can be more informative for machine learning models.  

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/f860e8f4-fea4-48a9-9086-5d94c37750c9) 


2.4.6 Correlations 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/022c9b88-af82-4b2b-9177-d396f40bddf0) 
   

2.4.7 Findings from Numerical Variables. 

    Age: Most customers are younger. 
    
    Campaign: Highly skewed towards fewer contacts. 
    
    Previous: Most customers have not been contacted in previous campaigns. 
    
    Employment Variation Rate: Indicates relatively stable employment conditions. 
    
    
    Pdays: Large gap between contacts for many customers. 
    
    Consumer Price Index and Consumer Confidence Index: Clustered around specific periods.
    
    Euribor 3 Month Rate: Indicates periods of higher interest rates. 
    
    Number of Employees: Suggests periods with specific employment levels. 
    

## Step#3: Data Preparation

    Log Transformation: Correct skewed distributions (e.g., pdays, previous, campaign).
    Binning Numerical Variables: Represent clustered values for features like emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, and nr.employed. 
    Customer Contact Patterns: Implement feature engineering actions for categorical and numerical variables. 
    

3.1 Feature Engieering Actions:

3.1.1 log Duration ( Mainly for Logistic Regression)
   
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/c577a06e-a329-405a-a7cb-ff459c08f6fa) 


3.1.2 Segementation of Numerical Variables/Create New Cateogircal Variables.
   
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/af067870-fc14-4b9c-a93e-c3735aa4d265)

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a3beb1a5-244d-4463-b943-b04cae5214db) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/5bece7af-2f3a-4673-8923-c367d34521a5) 


3.1.3 Encode all Categorical variables


3.1.4 Run a correlation again based on a filter (when correlation coefficient >0.05) to identify relative important variables

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/ceaf662b-c444-4664-8af6-540ef6b59fdd) 


Insights from Correlation Anaysis:

pdays and previous are moderately negatively correlated.
emp.var.rate, euribor3m, and nr.employed are highly positively correlated with each other.
euribor3m_segment_3 and above is highly correlated with euribor3m.
poutcome_nonexistent and previous_segment_0 previous are highly correlated.
previous_segment_1 previous and poutcome_success are highly correlated.
previous_segment_2 or more previous is highly correlated with previous.


3.2 Feature Selection Strategy

Instead of dropping any variables, use a Random Forest Classifier to select top significant features. 
This is efficient given the robonest of Random Forst and My limited computational resources.

3.3. Multiliearnty detection
Random Forest VIF

## Step#4: Modelling 

4.1 Modelling Objects:
   1) Find a model that can handle unbalanced data well. 
   2) Find a optimal threshold that balance recall and precision score to meet business goal(Precision + Recall = F1 Score)


4.2 Pre-work

4.2.1 Modelling Steps Overview

     1. Initially include all variables in the base model.
     2. Use Random Forest to Identify the most important features.
     3. Assess multicollinearity factors (VIF analysis) to exclude highly correlated variables.
     4. Exclude variables based on: 1) Previous Correlation analysis, 2) Initial model results, and 2) VIF analysis to Ensured all selected features were correlated with the target variable without multicollinearity concerns.
     5. Run correlation based on filtered dataset to ensure no highly correlatons between variables.
     6. Understand How Optimal Thresholds impact the model
     7. Test models using the top 20, 15, and 12 important features
     8. Found that the top 12 features provided the best F1 score.
     9. Conduct optimized threshold search and grid search to identify the best model for the Random Forest classifier.
     10. Run and Compare performance of all 6 classifiers using the top 12 selected features.

4.2.2 Split the final dataset
4.3.3 Scale the final dataset ( prevent data leakage) 


4.3 Base Model Development (using Random Forest Classifer to Understand/Select top significant features). 

4.4  Feature Importance Analysis (using Random Forest) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/52f100b1-1e64-4466-bfb1-b8aa0a4389b1)  


4.5 VIF Analysis ( Random Forest) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0122b9e2-fae2-464a-9a37-c53a343a7b69) 

4.6 Insights:

High VIF Values (Potential Multicollinearity Issues):

pdays_segment_100 days and above
euribor3m_segment_Below 3
euribor3m_segment_3 and above
pdays_segment_Below 100 days 

Moderate VIF Values:

pdays: VIF = 77435.77
emp.var.rate: VIF = 41292.00
nr.employed: VIF = 35629.81


Exclude Features with Inf VIF Values:

pdays_segment_100 days and above
euribor3m_segment_Below 100 days
euribor3m_segment_3 and above
pdays_segment_Below 100 days

Consider Excluding Features with Very High VIF Values (Above 10):
pdays
emp.var.rate
nr.employed

Highly Correlated Features:
emp.var.rate
nr.employed
euribor3m

4.7 Run Correlation Again (after filtering out features with high VIF values and Correlation Coefficients

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0e68c0ef-dd26-4f6f-997e-ffabfc64fdec)

4.8 Final Dataset for building a simple model (62 features and 1 response variable ) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b01a55e4-9f9f-44bd-8a92-111cf1c93b7c) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0984f9f0-bd6d-427b-90b3-8571b4b69cec)


4.9 Repeat 4.4 to Select top 20, top 15, and top 12. I found top 12 features perform best in term of F1-Score. 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/f292c144-eb0d-4571-b9b7-64e04fbb4330) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/27b9d63b-cea7-4800-97ca-2caeaf613d20) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9914640e-ba75-4c46-8cdc-2f427776a216) 

4.10 

4.6 Search Optimal threshold to meet business goal  
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/fadbdff6-69c8-4676-a70c-c5100303907d) 


4.7 Cross Validations and Grid Search to find the statble and best model

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/88dabc67-f048-4bb9-8f08-9b0588e6add5) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/14df72d3-9d11-4b74-b9f1-d8b74d0ff846) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a3803ddf-d15a-41b3-8329-acc60fcfe9ce) 


4.8 Model Hypermeters Specifications and Identify the Best Threshold for each of 5 classifers 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/5a866130-ab48-4ab0-9832-5a859df6d1cb) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/1bf057dc-1ac2-4142-b1c6-9dfda42c9e99) 



## Step#5: Model Evaluation Parameters

5.1 Threshold Consideration

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/23c751c6-c8ec-4aca-87f2-1163a315e3ff) 

5.2 Precision and Recall Curve with Optimal Threshhold
   

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/10feb1f5-205f-43a5-a6f3-2c9829d6eb55)


5.3 ROC Curve

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/808ed417-6825-4318-a70e-2ff657d4f23a) 


5.4 Training and Testing Accurancy and Computing Time


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/933c7dca-9e8a-446c-b3e5-696702eee519) 


## Conclusions:

The Random Forest classifier is recommended for deployment due to its strong performance in testing accuracy and reasonable computation time.  

Random Forest and SVM classifiers showed strong performance with high testing accuracy and decent F1 scores.
Random Forest is recommended due to its slightly higher testing accuracy (0.91) and reasonable computation time.
SVM could be considered if computation time is not a constraint, as it also showed strong performance with a high F1 score.
Decision Tree is easiler overfitted the training data. 


## Cost-Benefit Analysis:

Random Forest has a longer computation time compared to Logistic Regression but provides higher accuracy.
Logistic Regression offers a quick computation time with competitive performance.
SVM, while effective, has a very high computation time which may not be practical for regular updates.
Implementation Strategy:

Deploy the Random Forest model due to its balance of performance and computation time.
Monitor the Logistic Regression model as a backup or for scenarios where quick retraining is needed.
Use SVM model for batch processing or periodic analysis if computation resources allow.


## Further Actions:

Regular monitoring, threshold adjustments, and stakeholder communication are crucial for maintaining and improving model effectiveness.

Threshold Adjustment: Fine-tune the decision threshold periodically to align with business goals (cost reduction or maximizing conversions).
Regular Model Updates: Retrain models periodically with new data to maintain performance.
Customer Segmentation: Use model predictions to segment customers for targeted marketing campaigns, improving resource allocation and campaign efficiency.
Handling Imbalanced Datasets:

If dataset imbalance is a concern, consider using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
Continuously monitor the precision and recall metrics to ensure the model remains effective in identifying true positives without excessively increasing false positives.


Stakeholder Communication:

Highlight the high AUC and F1 scores to demonstrate the model's robustness.
Explain the trade-offs between precision and recall, especially for class 1 (subscribers).
Provide visualizations (ROC curves, precision-recall curves) to make the performance metrics more understandable.
Discuss the business implications of different threshold settings and the potential impact on campaign costs and customer experience.

Following CRISP-DM Methodology to Monitor the performance of classiders with new data and new features available. 








