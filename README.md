# Practical Application III: Apply and Compare 5 Common Classifiers to Predict Subscription/Direct Deposit Outcomes from Banking Marketing Campaigns

Overview/Learning Goal: 

In this practical application, the goal is to compare the performance of classifiers including K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines, and Random Forest. 
A dataset related to marketing bank products over the telephone is used for the applicaiton. in additional 

## Methodology: CRISP-DM Framework 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/26a9f4ba-b87d-4319-ba53-fb00675ec3eb) 



## Business Understandings: Context, Goal and Benefits

In today's competitive market, optimizing the efficiency and effectiveness of direct marketing campaigns is crucial for increasing customer engagement and maximizing returns on business investment.

### Business/ML Goal: Use ML techniques to predict which clients are most likely to subscribe to a deposit after being contacted. 

### Business Benefits: 

1. Targeted Campaigns: By predicting which clients are most likely to respond positively, we can focus our efforts on high-potential leads. This means more effective use of our marketing budget and resources.
2. Resource Optimization: ML helps us understand the factors that drive successful client subscriptions. This insight allows us to allocate our resources—such as human effort, phone calls, and time—more efficiently. Instead of a broad approach, we can tailor our strategies to where they are most effective. 
3. Improved Customer Engagement: By identifying the key characteristics of clients who are likely to subscribe, we can personalize our marketing messages and offers. This leads to higher engagement rates and better customer satisfaction.For example, targeting specific groups like retirees with tailored offers.
4. Cost Efficiency: Focusing on the most promising leads reduces wasted efforts and costs associated with less effective campaigns. ML helps us find the balance between quality and affordability when selecting potential customers ensuring high precision and recall in marketing efforts.
5. Strategic Insights: The data-driven insights from ML provide a deeper understanding of our customer base and campaign performance. These insights enable us to refine our strategies continually, leading to ongoing improvements in our marketing effectiveness by evaluating, deploying, monitoring, and tuning our models to ensure optimal performance and continuous improvement.

By leveraging machine learning and CRIP methodology, we can continousely transform our marketing campaigns from broad, generalized efforts to highly targeted and efficient operations, even highly personalized campaigns. This not only boosts our successful bank product subsribers but also ensures that we are using our resources in the most impactful way possible.

## Data Understandings: Backgound, Data Dictonoary, Quality Check, and EDA 

### Background: 

The dataset comes from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. We will make use of the article accompanying the dataset here for more information on the data and features. To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the Materials and Methods section of the paper. How many marketing campaigns does this data represent? 

The dataset was collected from a Portuguese bank that conducted direct marketing campaigns using their contact center. The primary marketing channel was telephone calls by human agents, sometimes supplemented by online banking via the Internet. Each campaign was managed integrally, with combined results from all channels. 

The dataset encompasses 17 campaigns from May 2008 to November 2010, totaling 41,188 contacts. These campaigns promoted a long-term deposit application with attractive interest rates. For each contact, numerous attributes were recorded, including whether the campaign was successful as the target variable. Out of the entire dataset, there were 6,499 successful subscriptions, resulting in an 11 % success rate. 

### Data Dictionary: 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/e95d693c-385e-41ee-9cdf-797b3bffbf47)


### Data Quality Check: 

1) 0 null values
2) 12 duplicates
3) unbalanced dataset (see the chart below)
   
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/2498bc86-9725-4d65-be43-3874b9fe068d) 


### EDA
#### 1)Explore the relationships among Categorical Varaibles and Response Variable 


##### Findings 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/4d07a0f4-a351-45c5-9ed7-aa2e49cd1d9b) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b833d6bf-625f-431f-bda2-5816ce588909) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/85c30490-8156-4c8a-8289-cfa2c32f5842)  
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a4fb81eb-7235-4e11-9483-d9c7ffaeaf2e) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/ad89503c-3ebe-4a40-a6c1-29f35530ee39) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/d5596456-0f64-4319-b402-e4f7fec9f7f3) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/e915aed8-7287-4967-a830-a205f1fccf95) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b87e8b89-4f86-4a18-906e-2f726900d0c4)





##### Key Insights and Next Steps for Data Preparation or Feature Engineering 

1) Job, education, contact, previous outcome, and month appear to be more influential in determining the outcome.

2) Loan, housing, and day of the week do not contribute to the success at all; their success rates are equal to the overall success rate.

   ![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/d26270bc-94c7-4bde-a34a-9e07a585cfba)

   ![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9434a7fd-bbea-4007-8df2-d1ee89509509)

   ![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9260ba25-a3a6-48e6-83bf-daec649f60b1)

3) Marital status and default have a slight influence on the outcome.


#### Categorical Variables: Next Steps/Actions fpr Data Preparation and Feature Engineering. 

**Job Category: Encoding 
Why? Different job categories have varying subscription rates.

**Marital Status: Encoding

Why? Marital status shows some influence on subscription rates.

**Education: Encoding

Why: Different education levels have varying subscription rates.

**Default: Feature Importance: (removed)

Why? The default status has minimal influence on subscription rates.

**Housing: Encoding

Why? Housing loan status shows some influence on subscription rates.

** Loan: Encoding

Why? Personal loan status shows some influence on subscription rates.

** Contact: Encoding

Why? Different contact types have varying subscription rates.

** Month: Encoding

Why? The month of contact has a significant influence on subscription rates.

**Day of the Week: Feature Importance (removed) 

Why? The day of the week of contact has minimal influence on subscription rates.

**Poutcome (Previous Outcome): Encoding

Why? Previous campaign outcomes significantly influence subscription rates. 



#### EDA 2)Explore the relationships among Numerical Vraible and Response Variable 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/bbd8bdbc-df34-4a07-908f-78b4bd476156) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/86cfbedc-a94f-4e68-824b-b0fa1b430dd7) 


#### Key Findings frome Numerical Variables

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/df4238af-28b7-46e3-9b77-27e344d43f44) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/3310e747-e38c-4de1-9d84-1791e75f7b02) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/ef850449-a748-41a4-8e1f-2e4a0394ace5) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/46e94d94-4842-4c3a-b961-a54527eaa63e) 




T-TESTS 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/f860e8f4-fea4-48a9-9086-5d94c37750c9) 


Correlations 



![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/022c9b88-af82-4b2b-9177-d396f40bddf0) 


## Data Preparation Key Actions/ Feature Engineering 

1. log Duration (Logistic Regression)
   

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/c577a06e-a329-405a-a7cb-ff459c08f6fa) 


2. segementation of variables/Create new cateogircal variables. (explorations)
   
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/af067870-fc14-4b9c-a93e-c3735aa4d265)

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a3beb1a5-244d-4463-b943-b04cae5214db) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/5bece7af-2f3a-4673-8923-c367d34521a5) 


3. encoding all categorical variables. Now the full dataset including 67 coded variables.

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9c421e84-fa1b-499f-a050-2847d776319f) 



4. run a correlation based on a filter ( when correlation coefficient >0.1) to identify relative important variables


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/b0cf467c-20f7-4d0b-88ba-da039c97131c) 



5. Final Dataset for Modelling (

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/0b0d5531-76c4-4d27-9c32-f4a5468cdc70) 


6. Spliting the final dataset
7. Scaling the final dataset 

## Modelling 

### 1) Base Model using Random Forest Classifer
One of my key goal is to find a good balance of Precision and Recall indicating by F1 Score.  The F1 score is low despite I using class_weight = 'balanced' I
Guess I have to do resampling or some other types of sampling skill. overall F1 scorea are all low. 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/617b5fa5-8f56-4927-9119-8508ecb00bee)

Due to my limited computation power. I preselect features to run the models. 

Before running cross validation and Grid Search to find the best model
I analyze the feature importance using Random Forest.
List of Important Features (using Random Forest)

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/285becbc-96f6-4ecd-b8a1-787877a3c004) 

### 2) Then I chose top 20, top 15, and top 12. I found top 15 is the best model in term of F1-Score. 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/27b9d63b-cea7-4800-97ca-2caeaf613d20) 

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/9914640e-ba75-4c46-8cdc-2f427776a216) 


### 3) Find Optimal threshold to meet business goal  
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/fadbdff6-69c8-4676-a70c-c5100303907d) 


### 4) Cross Validations and Grid Search to find the statble and best model

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/88dabc67-f048-4bb9-8f08-9b0588e6add5) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/14df72d3-9d11-4b74-b9f1-d8b74d0ff846) 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/a3803ddf-d15a-41b3-8329-acc60fcfe9ce) 


### 5) Model Hypermeters Specifications and Identify the Best Threshold for each of 5 classifers 


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/5a866130-ab48-4ab0-9832-5a859df6d1cb) 
![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/1bf057dc-1ac2-4142-b1c6-9dfda42c9e99) 



## Model Evaluation Criteria

1) Threshold Consideration

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/23c751c6-c8ec-4aca-87f2-1163a315e3ff) 

2) Precision and Recall Curve with Optimal Threshhold
   

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/10feb1f5-205f-43a5-a6f3-2c9829d6eb55)


3) ROC Curve

![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/808ed417-6825-4318-a70e-2ff657d4f23a) 


4) Training and Testing Accurancy and Computing Time


![image](https://github.com/Sandysmile/Marketing-Campaigns/assets/20648423/933c7dca-9e8a-446c-b3e5-696702eee519) 



Recomended Model 
Random Forest


## Next Steps





