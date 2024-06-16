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

2.4.1 Catogrical Variables and Response Variable.

Key Findings:
   
   1) Job, education, contact, previous outcome, and month appear to be more influential in determining the outcome.
   
      Non-significant Features: 
   
   2) Loan, housing, and day of the week do not significantly contribute to the subscription success. 
   
      Feature Engineering: 
   
   3) Encode all categorical variables for basic models. 
   
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
   



2.4.2 Numerical Variables and Response Variable
   
2.4.2.1 Group Histograms

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/bbd8bdbc-df34-4a07-908f-78b4bd476156) 

2.4.2.2 Group Statistics

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/86cfbedc-a94f-4e68-824b-b0fa1b430dd7) 

2.4.3 Group Boxplots

2.4.4 Individual Histogram with Boxplot
   
   1) Highly Informative Features: 
   
      Call duration, pdays, previous contacts, employment variation rate, and Euribor rate are strong indicators for predicting positive responses (see the titles of the charts below for detailed findings) 
   
   2) Moderately Informative Features:
      
      Campaign contacts, age, and consumer confidence index provide additional insights. 
   
   3) Economic Indicators: 
   
      Negative employment variation rates and lower Euribor rates are associated with "yes" responses, indicating less favorable economic conditions may increase receptiveness. 
   
   4) Customer Contact Patterns: 
   
      Effective campaigns involve fewer, more targeted contacts and longer call durations.
   

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/df4238af-28b7-46e3-9b77-27e344d43f44) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/3310e747-e38c-4de1-9d84-1791e75f7b02) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/ef850449-a748-41a4-8e1f-2e4a0394ace5) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/46e94d94-4842-4c3a-b961-a54527eaa63e) 

    
2.4.5 T-Test 


It determines whether the means of numerical variables are significantly different between the groups (e.g., "yes" vs. "no"). By doing so, I can understand which features are potentially influential in predicting the target variable. Since significant features can be more informative for machine learning models.  

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/f860e8f4-fea4-48a9-9086-5d94c37750c9) 


2.4.6 Correlation 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/022c9b88-af82-4b2b-9177-d396f40bddf0) 
   

2.4.7 Findings 

    #Age: Most customers are younger. 
    
    #Campaign: Highly skewed towards fewer contacts. 
    
    #Previous: Most customers have not been contacted in previous campaigns. 
    
    #Employment Variation Rate: Indicates relatively stable employment conditions. 
    
    
    #Pdays: Large gap between contacts for many customers. 
    
    #Consumer Price Index and Consumer Confidence Index: Clustered around specific periods.
    
    #Euribor 3 Month Rate: Indicates periods of higher interest rates. 
    
    #Number of Employees: Suggests periods with specific employment levels. 
    

## Step#3: Data Preparation

    1) Log Transformation: Correct skewed distributions (e.g., pdays, previous, campaign). 
    
    2) Binning Numerical Variables: Represent clustered values for features like emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, and nr.employed. 
    
    3) Customer Contact Patterns: Implement feature engineering actions for categorical and numerical variables. 
    
    

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

1) pdays and previous are moderately negatively correlated.
2) emp.var.rate, euribor3m, and nr.employed are highly positively correlated with each other.
3) euribor3m_segment_3 and above is highly correlated with euribor3m.
4) poutcome_nonexistent and previous_segment_0 previous are highly correlated.
5) previous_segment_1 previous and poutcome_success are highly correlated.
6) previous_segment_2 or more previous is highly correlated with previous.


3.2 Feature Selection Strategy

Instead of dropping any variables outright, use a Random Forest to identify and select the most significant features. Given the robustness of Random Forest and the limited computational resources available, this approach will help ensure that the most impactful variables are retained for further analysis. 


## Step#4: Modelling 

4.1 Modelling Objectives: 

  1) Identify a model that effectively handles imbalanced data.
     
  2) Determine an optimal threshold that balances recall and precision scores to meet the business objective, ensuring a high F1 score (Precision + Recall = F1 Score).

     
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

4.2.2 Split Dataset

4.2.3 Scale the Dataset (performed after splitting to prevent data leakage) 

4.3 Base Model Development (Using Random Forest Classifer to Understand/Select Top Important Features). 

4.4  Feature Importance Analysis (Using Random Forest) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/52f100b1-1e64-4466-bfb1-b8aa0a4389b1)  


4.5 VIF Analysis (Random Forest) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0122b9e2-fae2-464a-9a37-c53a343a7b69) 

4.6 Insights/Actions:

   1) High VIF Values (Potential Multicollinearity Issues):

     pdays_segment_100 days and above
     euribor3m_segment_Below 3
     euribor3m_segment_3 and above
     pdays_segment_Below 100 days 

   2) Moderate VIF Values:

     pdays: VIF = 77435.77
     emp.var.rate: VIF = 41292.00
     nr.employed: VIF = 35629.81


   3) Exclude Features with Inf VIF Values:

     pdays_segment_100 days and above
     euribor3m_segment_Below 100 days
     euribor3m_segment_3 and above
     pdays_segment_Below 100 days

   4) Consider Excluding Features with Very High VIF Values (Above 10):

     pdays
     emp.var.rate
     nr.employed

   5) Consider Excluding One or Two Highly Correlated Features:

     emp.var.rate
     nr.employed
     euribor3m

4.7 Run Correlation Again (after filtering out features with high VIF values and Correlation Coefficients

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0e68c0ef-dd26-4f6f-997e-ffabfc64fdec)

4.8 Final Dataset for building a simple model (62 features and 1 response variable ) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b01a55e4-9f9f-44bd-8a92-111cf1c93b7c) 



4.9 Repeat 4.4 to Select top 20, top 15, and top 12. I found top 12 features perform best in term of F1-Score. 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/e644c0aa-fbcc-4ddf-905c-ea4e6d175589) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0c3c7f61-2b65-42a4-a626-918a66411b07) 



4.6 Search Optimal thresholds to Meet the Modelling Objective

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/6812e545-7904-423a-a8a0-0fe6a393d1fa) 



4.7 Use Cross Validations and Grid Search to find the Stable and Best Model with Optimal Threshold for F1 Score

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/06accc7d-0b17-448c-989c-0981ee5d030e) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/4be2bb78-affc-4e9f-9a4b-ac154f71f78c) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/20cb5219-4245-4d77-8c78-a6577b49b49e) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/8762ca96-46b8-4270-a371-b18be7c2ebb0) 




4.8 Model Hypermeters Specifications and Identify the Best Threshold for each of 5 classifers 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/c6a6207d-eaaf-4618-81b3-212aff84e278) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a714cdda-5f7e-4343-819b-78eef2d48045) 




## Step#5: Model Evaluation Parameters

5.1 Threshold Consideration

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/d2f7cbac-45b2-4427-8423-d8204aecc844) 


5.2 Precision and Recall Curve with Optimal Threshhold
   

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/8e48e757-8b14-46ae-9b13-4c18a2880222) 


5.3 ROC Curve

!![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/f4b5593f-ee8b-4c48-842e-6c810efd64b2) 


5.4 Training and Testing Accurancy and Computing Time


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/f7a41c3a-6f12-4720-8362-91be82701cf8) 

 

## Recommendations:

1. Model Selection:

Deploy Random Forest due to its high testing accuracy (0.91) and reasonable computation time.
Monitor Logistic Regression as a quick backup option for scenarios requiring rapid retraining.
Consider SVM for batch processing or periodic analysis if computation resources allow.

2. Cost-Benefit Analysis:

Random Forest offers a balance of performance and computation time.
Logistic Regression provides a quick computation time with competitive performance.
SVM has a high computation time which may not be practical for regular updates but can be used for detailed analysis.
Implementation Strategy:

3. Deploy Random Forest as the primary model.
Keep Logistic Regression as an alternative for quick updates.
Use SVM selectively for in-depth analysis.

Further Actions:

1. Threshold Adjustment:
Regularly fine-tune the decision threshold to align with business goals.

2. Regular Model Updates: Retrain models periodically with new data.

3. Customer Segmentation: Utilize model predictions for targeted marketing campaigns.
Handling Imbalanced Datasets:

4. Dataset Balance Use SMOTE or other techniques to balance the dataset.
5. Continuously monitor precision and recall metrics to ensure the model effectively identifies true positives.
6. Stakeholder Communication:

   1) Highlight high AUC and F1 scores.
   2) Explain trade-offs between precision and recall.
   3) Provide visualizations (ROC curves, precision-recall curves).
   4) Discuss business implications of different threshold settings on campaign costs and customer experience using imbalanced dataset


## Conclusion:

The Random Forest classifier is recommended for deployment due to its strong performance and reasonable computation time. Logistic Regression can serve as a quick alternative, while SVM can be used for periodic detailed analysis. Regular monitoring, threshold adjustments, and effective stakeholder communication are crucial for maintaining and improving model performance.
















