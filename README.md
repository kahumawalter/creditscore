# About the Project
In this project, I used a multi-class classification project to help a finance company predict their consumers' credit score based on the consumers' credit-related information. The project is split into 2 parts which show data visualization and a deep learning model that were used to make predictions to the credit score. 

#Data Visualization

The heatmap shows there is a high positive correlation between Credit_score and features like Credit_mix and Credit_history_age. There is also a high negative correlation between Credit_score and features like Outstanding Debt, Num_credit_inquiries, Num_of Delayed_Payment, Delay_from_due_date, Num_of_loan, Interest_rate, Num_credit_card and Num_Bank_accounts. 


The box plot shows that those with a credit score of 0 (Poor) have an outstanding debt of about 2,000. According to the interquartile range (IQR) it ranges between 1500 and 2500. Those with a credit score of 1 (Standard) have an outstanding debt that ranges from 500 to 1500 according to the interquartile range. While as those with a credit score of 2 (Good) have an outstanding debt that ranges from 400 to 1200. 


# Summary of the Project

After analysing and preprocessing the data, I made predictions to the credit score using a multi-class classification model (Keras) and this gave me accuracy score of 83 on the predicted data. However the model was able to predict the standard class better because of the class imbalance in the data provided. 

