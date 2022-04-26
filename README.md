# Jobathon, September 2021
by Analytics Vidhya 

Problem Statement:

# Supplement Sales Prediction

Your Client WOMart is a leading nutrition and supplement retail chain that offers a comprehensive range of products for all your wellness and fitness needs. WOMart follows a multi-channel distribution strategy with 350+ retail stores spread across 100+ cities. Effective forecasting for store sales gives essential insight into upcoming cash flow, meaning WOMart can more accurately plan the cashflow at the store level. Sales data for 18 months from 365 stores of WOMart is available along with information on Store Type, Location Type for each store, Region Code for every store, Discount provided by the store on every day, Number of Orders everyday etc. Your task is to predict the store sales for each store in the test set for the next two months.

# 3-fold Cross-Validation definition :

train1: January - November 2018
test1: December 2018 - January 2019

train2: January - January 2019
test2: February - March 2019

train3: January 2018 - March 2019
test3: April - May 2019`    `

# Models used :
After initial data exploration, I used XGBoost, LightGBM and CatBoost regressor to make predictions for the test data. 
