# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Churn Prediction - Single click deployment with AutoML
# MAGIC 
# MAGIC Now that our data is available, our first step as Data Scientist is to analyze and build the features we'll use to train our model. Let's see how this can be done.
# MAGIC 
# MAGIC The users enriched with spend data has been saved as gold_table within our Delta Live Table pipeline. All we have to do is read this information, analyze it and start an Auto-ML run.
# MAGIC 
# MAGIC Make sure you switched to the "Machine Learning" persona on the top left menu.
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-churn-ml-flow.png" width="800px">
# MAGIC 
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fretail%2Flakehouse_churn%2Fautoml&dt=LAKEHOUSE_RETAIL_CHURN">

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

import bamboolib as bam

# COMMAND ----------

database = 'rac_retail_demo'
spark.sql(f"USE {database}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data exploration and analysis
# MAGIC 
# MAGIC Let's review our dataset and start analyze the data we have to predict our churn

# COMMAND ----------

# DBTITLE 1,Read our churn gold table
# Read our churn_features table
churn_dataset = spark.table("churn_features")
display(churn_dataset)

# COMMAND ----------

# DBTITLE 1,Data Exploration and analysis
import seaborn as sns
g = sns.PairGrid(churn_dataset.sample(0.01).toPandas()[['age_group','gender','order_count']], diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)
g.map_upper(sns.regplot)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Further data analysis and preparation using pandas API
# MAGIC 
# MAGIC Because our Data Scientist team is familiar with Pandas, we'll use `pandas on spark` to scale `pandas` code. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC 
# MAGIC Typicaly Data Science project would involve more advanced preparation and likely require extra data prep step, including more complex feature preparation. We'll keep it simple for this demo.
# MAGIC 
# MAGIC *Note: Starting from `spark 3.2`, koalas is builtin and we can get an Pandas Dataframe using `pandas_api()`.*

# COMMAND ----------

# DBTITLE 1,Custom pandas transformation / code on top of your entire dataset - KOALAS
# Convert to koalas
dataset = churn_dataset.pandas_api()
dataset.describe()  
# Drop columns we don't want to use in our model
dataset = dataset.drop(columns=['address', 'email', 'firstname', 'lastname', 'creation_date', 'last_activity_date', 'last_event'])
# Drop missing values
dataset = dataset.dropna()   

# COMMAND ----------

# DBTITLE 1,Bamboolib - Exploration
pdf = churn_dataset.toPandas()
pdf

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Write to Feature Store (Optional)
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC 
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC 
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC 
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features. It also simplify realtime serving.
# MAGIC 
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

try:
  #drop table if exists
  fs.drop_table(f'{database}.churn_user_features')
except:
  pass
#Note: You might need to delete the FS table using the UI
churn_feature_table = fs.create_table(
  name=f'{database}.churn_user_features',
  primary_keys='user_id',
  schema=dataset.spark.schema(),
  description='These features are derived from the churn_bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=dataset.to_spark(), name=f'{database}.churn_user_features', mode='overwrite')
features = fs.read_table(f'{database}.churn_user_features')
display(features)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Accelerating Churn model creation using MLFlow and Databricks Auto-ML
# MAGIC 
# MAGIC MLflow is an open source project allowing model tracking, packaging and deployment. Everytime your datascientist team work on a model, Databricks will track all the parameter and data used and will save it. This ensure ML traceability and reproductibility, making it easy to know which model was build using which parameters/data.
# MAGIC 
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC 
# MAGIC While Databricks simplify model deployment and governance (MLOps) with MLFlow, bootstraping new ML projects can still be long and inefficient. 
# MAGIC 
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC 
# MAGIC 
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC 
# MAGIC 
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC 
# MAGIC <br style="clear: both">
# MAGIC 
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC 
# MAGIC ### Using Databricks Auto ML with our Churn dataset
# MAGIC 
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`churn_features`)
# MAGIC 
# MAGIC Our prediction target is the `churn` column.
# MAGIC 
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC 
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------

from databricks import automl
train_dataset = fs.read_table(f'{database}.churn_user_features')

# COMMAND ----------

display(train_dataset)

# COMMAND ----------

# summary = automl.classify(train_dataset.select('gender', 'age_group', 'churn'), target_col="churn")
summary = automl.classify(train_dataset, target_col="churn")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### The model generated by AutoML is ready to be used in our DLT pipeline to detect customers about to churn.
# MAGIC 
# MAGIC Our Data Engineer can now easily retrive the model `dbdemos_customer_churn` from our Auto ML run and predict churn within our Delta Live Table Pipeline.<br>
# MAGIC Re-open the DLT pipeline to see how this is done.
# MAGIC 
# MAGIC #### Track churn impact over the next month and campaign impact
# MAGIC 
# MAGIC This churn prediction can be re-used in our dashboard to analyse future churn, take actiond and measure churn reduction. 
# MAGIC 
# MAGIC The pipeline created with the Lakehouse will offer a strong ROI: it took us a few hours to setup this pipeline end 2 end and we have potential gain for $129,914 / month!
# MAGIC 
# MAGIC <img width="800px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-churn-dbsql-prediction-dashboard.png">
# MAGIC 
# MAGIC <a href='/sql/dashboards/1e236ef7-cf58-4bfc-b861-5e6a0c105e51'>Open the Churn prediction DBSQL dashboard</a> | [Go back to the introduction]($../00-churn-introduction-lakehouse)
# MAGIC 
# MAGIC #### More advanced model deployment (batch or serverless realtime)
# MAGIC 
# MAGIC We can also use the model `dbdemos_custom_churn` and run our predict in a standalone batch or realtime inferences! 
# MAGIC 
# MAGIC Next step:  [Explore the generated Auto-ML notebook]($./04.2-automl-generated-notebook) and [Run inferences in production]($./04.3-running-inference)
