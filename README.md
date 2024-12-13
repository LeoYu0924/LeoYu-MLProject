# LeoYu-MLProject

### Description
For this semester-long project, students will build a complete machine learning pipeline that incorporates big data technologies using a cloud infrastructure.  The project is split into 6 milestones that will be due throughout the semester.  The project must be original (e.g., not copied from a previous semester, competition or other source). However, code examples can be referenced (proper citations required).


### Code

Appendix A

N/A


Appendix B
from google.cloud import storage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Setting up Google Cloud Storage client
bucket_name = 'my-bucket-ly'
file_path = 'landing/all_reviews.csv'
output_bucket_name = 'my-bucket-ly'
output_path = 'cleaned/steam_game_reviews_top_200_games.parquet'

client = storage.Client()
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(file_path)

#Download the CSV file to a local file
blob.download_to_filename('/tmp/all_reviews.csv')

#Load data into a DataFrame
df = pd.read_csv('/tmp/all_reviews.csv')



#Exploratory Data Analysis (EDA)

num_records = len(df)
columns = list(df.columns)

missing_values = df.isnull().sum().to_dict()

descriptive_stats = df.describe().toPandas().transpose()

min_values = descriptive_stats.loc['min']
max_values = descriptive_stats.loc['max']
mean_values = descriptive_stats.loc['mean']
std_values = descriptive_stats.loc['std']

if df.select_dtypes(include=['datetime']).shape[1] > 0:
    date_columns = df.select_dtypes(include=['datetime']).columns.tolist()
    min_dates = {col: df[col].min() for col in date_columns}
    max_dates = {col: df[col].max() for col in date_columns}

text_columns = [field for field in df.columns if df[field].dtype == 'object']
num_words_stats = {}
for col in text_columns:
    num_words_stats[col] = {
        'avg_words': df[col].dropna().apply(lambda x: len(x.split())).mean(),
        'stddev_words': df[col].dropna().apply(lambda x: len(x.split())).std(),
        'min_words': df[col].dropna().apply(lambda x: len(x.split())).min(),
        'max_words': df[col].dropna().apply(lambda x: len(x.split())).max()
    }

#Print results for highlights
print(f"Number of records: {num_records}")
print(f"Columns: {columns}")
print(f"Number of missing values per column:\n{missing_values}")
print(f"Descriptive statistics for numeric variables:\n{descriptive_stats}")

if any([df.schema[field].dataType.simpleString() == 'timestamp' for field in df.columns]):
    print(f"Min dates:\n{min_dates}")
    print(f"Max dates:\n{max_dates}")

if len([field for field in df.columns if df.schema[field].dataType.simpleString() == 'string']) > 0:
    print(f"Number of words statistics for text data:\n{num_words_stats}")

#Creating Histograms for Categorical Variables
categorical_columns = [field for field in df.columns if df.schema[field].dataType.simpleString() == 'string']

#Limit the number of histograms generated
max_histograms = 3
for idx, col in enumerate(categorical_columns):
    if idx >= max_histograms:
        break
    sample_df = df.sample(fraction=0.05)  # Reduced sampling fraction to 5%
    sample_df[col].value_counts().plot(kind='bar', title=f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

Appendix C
#Data Cleaning Notebook
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, BooleanType, TimestampType

#Initialize Spark session
spark = SparkSession.builder \
    .appName("Steam Reviews Data Cleaning") \
    .getOrCreate()

#Define input and output paths
input_path = "gs://my-bucket-ly/landing/all_reviews.csv"
output_path = "gs://my-bucket-ly/cleaned/cleaned_reviews.parquet"

#Load data from landing folder
df = spark.read.csv(input_path, header=True, inferSchema=True)

#Drop records with missing values for important columns
df_cleaned = df.dropna(subset=["recommendationid", "appid", "author_steamid", "review"])

# Fill in missing values for less critical columns
df_cleaned = df_cleaned.fillna({
    "author_playtime_forever": 0,
    "author_playtime_last_two_weeks": 0,
    "votes_up": 0,
    "votes_funny": 0,
    "comment_count": 0
})

#Drop unnecessary columns
columns_to_drop = [
    "steam_purchase", "received_for_free", "written_during_early_access", 
    "hidden_in_steam_china", "steam_china_location"
]
df_cleaned = df_cleaned.drop(*columns_to_drop)

#Rename columns to remove spaces and make them more Python-friendly
column_renames = {
    "author_playtime_forever": "playtime_forever",
    "author_playtime_last_two_weeks": "playtime_last_two_weeks",
    "author_playtime_at_review": "playtime_at_review",
    "timestamp_created": "review_created",
    "timestamp_updated": "review_updated"
}
for old_name, new_name in column_renames.items():
    df_cleaned = df_cleaned.withColumnRenamed(old_name, new_name)

#Cast columns to appropriate data types
df_cleaned = df_cleaned.withColumn("appid", col("appid").cast(IntegerType())) \
    .withColumn("author_num_games_owned", col("author_num_games_owned").cast(IntegerType())) \
    .withColumn("author_num_reviews", col("author_num_reviews").cast(IntegerType())) \
    .withColumn("playtime_forever", col("playtime_forever").cast(IntegerType())) \
    .withColumn("playtime_last_two_weeks", col("playtime_last_two_weeks").cast(IntegerType())) \
    .withColumn("playtime_at_review", col("playtime_at_review").cast(IntegerType())) \
    .withColumn("review_created", col("review_created").cast(TimestampType())) \
    .withColumn("review_updated", col("review_updated").cast(TimestampType())) \
    .withColumn("votes_up", col("votes_up").cast(IntegerType())) \
    .withColumn("votes_funny", col("votes_funny").cast(IntegerType())) \
    .withColumn("comment_count", col("comment_count").cast(IntegerType())) \
    .withColumn("voted_up", col("voted_up").cast(BooleanType()))

#Write cleaned data to cleaned folder as a Parquet file
df_cleaned.write.mode("overwrite").parquet(output_path)

Appendix D

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from google.cloud import storage
import numpy as np
from pyspark.sql.functions import when, col, year, month, dayofweek
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder # Import for CrossValidator

#Initialize Spark session
spark = SparkSession.builder \
    .appName("Feature Engineering and Modeling") \
    .config("spark.master", "local[*]") \
    .getOrCreate()

#Load cleaned data from GCS into Spark DataFrame
bucket_name = 'my-bucket-ly'
cleaned_file_path = 'cleaned/all_reviews_cleaned.parquet'
gcs_url = f"gs://{bucket_name}/{cleaned_file_path}"
df_cleaned = spark.read.parquet(gcs_url)

#Convert `voted_up` to a binary value (0.0 or 1.0)
df_cleaned = df_cleaned.withColumn("voted_up", 
                                  when(col("voted_up") >= 1.0, 1.0)
                                  .otherwise(0.0))

#Ensure `voted_up` is a double
df_cleaned = df_cleaned.withColumn("voted_up", df_cleaned['voted_up'].cast('double'))

#Convert `received_for_free` to double if it exists
if 'received_for_free' in df_cleaned.columns:
    df_cleaned = df_cleaned.withColumn('received_for_free', when(col('received_for_free') == 'True', 1.0).otherwise(0.0))
    df_cleaned = df_cleaned.withColumn('received_for_free', df_cleaned['received_for_free'].cast('double'))

#Drop unnecessary columns
features_to_drop = ['recommendationid', 'appid', 'author_steamid']
df_cleaned = df_cleaned.drop(*features_to_drop)

#Handle missing values
for col in df_cleaned.columns:
    df_cleaned = df_cleaned.na.fill({col: 0})

#Extract useful features from timestamp columns
if 'timestamp' in df_cleaned.columns:
    df_cleaned = df_cleaned.withColumn('year', year(df_cleaned['timestamp']))
    df_cleaned = df_cleaned.withColumn('month', month(df_cleaned['timestamp']))
    df_cleaned = df_cleaned.withColumn('dayofweek', dayofweek(df_cleaned['timestamp']))
    df_cleaned.select('timestamp', 'year', 'month', 'dayofweek').show(truncate=False)

#Manually select columns for feature engineering based on their role
categorical_columns = ['steam_purchase', 'early_access']
categorical_columns = [col for col in categorical_columns if col in df_cleaned.columns]

numeric_columns = ['playtime_forever', 'playtime_last_two_weeks', 'playtime_at_review']
numeric_columns = [col for col in numeric_columns if col in df_cleaned.columns]

binary_columns = [col for col in ['received_for_free', 'is_featured'] if col in df_cleaned.columns]


#Index and OneHotEncode categorical columns
indexer = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid='keep') for col in categorical_columns]
encoder = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded", dropLast=True) for col in categorical_columns]

#Scale numeric columns using MinMaxScaler
scaled_numeric_cols = []
num_assembler_scaler = []
for num_col in numeric_columns:
    num_assembler = VectorAssembler(inputCols=[num_col], outputCol=f"{num_col}_vector")
    num_scaler = MinMaxScaler(inputCol=f"{num_col}_vector", outputCol=f"{num_col}_scaled")
    num_assembler_scaler.extend([num_assembler, num_scaler])
    scaled_numeric_cols.append(f"{num_col}_scaled")

#Assemble features
feature_columns = [f"{col}_encoded" for col in categorical_columns] + scaled_numeric_cols + binary_columns
assembler = VectorAssembler(inputCols=feature_columns, outputCol="assembled_features")

#Define the pipeline for feature engineering
pipeline_stages = indexer + encoder + num_assembler_scaler + [assembler]
feature_pipeline = Pipeline(stages=pipeline_stages)

#Apply the feature engineering pipeline
feature_model = feature_pipeline.fit(df_cleaned)
df_features = feature_model.transform(df_cleaned)

#Filter dataset to remove rows with missing `voted_up` before Train/Test Split
df_features = df_features.dropna(subset=['voted_up'])

#Train/Test Split
df_train, df_test = df_features.randomSplit([0.8, 0.2], seed=42)

#Logistic Regression Model to predict `voted_up`
lr = LogisticRegression(featuresCol="assembled_features", labelCol="voted_up", maxIter=10)

#Training the Logistic Regression model
lr_model = lr.fit(df_train)
lr_predictions = lr_model.transform(df_test)

#Define a parameter grid for hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 20, 50]) \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

#Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol='voted_up', rawPredictionCol='rawPrediction', metricName='areaUnderROC')

#Define CrossValidator
crossval = CrossValidator(estimator=lr, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator, 
                          numFolds=5)  # 5-fold cross-validation

#Train the model using cross-validation
cv_model = crossval.fit(df_train)

#Get the best model from cross-validation
best_model = cv_model.bestModel

#Evaluate the best model on the test set
cv_predictions = best_model.transform(df_test)
roc_auc_cv = evaluator.evaluate(cv_predictions)
print(f"Area Under ROC for the best model (Cross-Validated): {roc_auc_cv}")

#Save the best model
best_model.write().overwrite().save(f"gs://{bucket_name}/models/steam_game_review_best_lr_model")

#Save cross-validation results
cv_results_content = f"Area Under ROC (Cross-Validated): {roc_auc_cv}\n"
cv_results_content += f"Best Model Parameters: {best_model.extractParamMap()}\n"
cv_results_content += f"Model trained on: {df_train.count()} rows\n"
cv_results_content += f"Test data used for evaluation: {df_test.count()} rows\n"
cv_results_content += f"Cross-validation completed successfully.\n"

print(cv_results_content)

#Upload cross-validation results to GCS
client = storage.Client()
bucket = client.get_bucket(bucket_name)
cv_blob = bucket.blob('results/cross_validation_results.txt')
cv_blob.upload_from_string(cv_results_content)



#Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol='voted_up', rawPredictionCol='rawPrediction', metricName='areaUnderROC') 
roc_auc = evaluator.evaluate(lr_predictions)
print(f"Area Under ROC for Logistic Regression: {roc_auc}")

#Save the model
lr_model.write().overwrite().save(f"gs://{bucket_name}/models/steam_game_review_lr_model")

#Show predictions
lr_predictions.select('*').show(truncate=False)

#Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol='voted_up', rawPredictionCol='rawPrediction', metricName='areaUnderROC') 

roc_auc = evaluator.evaluate(lr_predictions)

train_rows = df_train.count()
test_rows = df_test.count()

#Prepare results content
results_content = f"Area Under ROC for Logistic Regression: {roc_auc}\n"
results_content += f"Model trained on: {train_rows} rows\n"
results_content += f"Test data used for evaluation: {test_rows} rows\n"
results_content += f"Model evaluation completed successfully.\n"

print(results_content) 


#Upload the results to GCS

client = storage.Client() bucket = client.get_bucket(bucket_name) blob = 

bucket.blob('results/feature_engineering_modeling_results.txt') 

blob.upload_from_string(results_content) print("Results successfully uploaded to: 

gs://{bucket_name}/results/feature_engineering_modeling_results.txt")n as e:




Appendix E

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from pyspark.sql import SparkSession

#Initialize Spark session
spark = SparkSession.builder.appName("DataProcessing").getOrCreate()

#Load the cleaned data from Spark
sdf = spark.read.parquet('gs://my-bucket-ly/cleaned/cleaned_reviews.parquet')

#Convert Spark DataFrame to Pandas DataFrame
if sdf.count() > 0:
    df = sdf.limit(10000).toPandas()
else:
    df = pd.DataFrame()  

#Simulate a logistic regression model and training/testing data
np.random.seed(42)  # For reproducibility
if not df.empty:
    # Check if the required column is present
    if 'playtime_at_review' in df.columns:
        X_train = df[['playtime_at_review']]  # Use the correct column name
    else:
        raise ValueError("The expected column 'playtime_at_review' is not found in the dataset.")

    
y_train = df['voted_up'].fillna(0).astype(int)  
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
X_test = X_train  
y_test = y_train
y_pred = log_reg_model.predict(X_test)
feature_names = X_train.columns
feature_importance = np.abs(log_reg_model.coef_[0])





#Initialize Google Cloud Storage client
client = storage.Client()
bucket = client.get_bucket('my-bucket-ly')

#Playtime vs Review Sentiment
if not df.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['playtime_at_review'], y=df['voted_up'], alpha=0.3)
    plt.xlabel('Playtime at Review (minutes)')
    plt.ylabel('Review Sentiment (0 = Negative, 1 = Positive)')
    plt.title('Playtime vs. Review Sentiment')
    plt.tight_layout()
    playtime_vs_sentiment_path = '/tmp/playtime_vs_sentiment.png'
    plt.savefig(playtime_vs_sentiment_path, bbox_inches='tight')
    plt.close()
    blob = bucket.blob('results/playtime_vs_sentiment.png')
    blob.upload_from_filename(playtime_vs_sentiment_path)


#Correlation Heatmap for Numerical Features
if not df.empty:
    plt.figure(figsize=(12, 10))
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    correlation_heatmap_path = '/tmp/correlation_heatmap.png'
    plt.savefig(correlation_heatmap_path, bbox_inches='tight')
    plt.close()
    blob = bucket.blob('results/correlation_heatmap.png')
    blob.upload_from_filename(correlation_heatmap_path)

#Class Distribution Pie Chart
if not df.empty:
    voted_up_counts = df['voted_up'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(voted_up_counts, labels=['Positive Reviews', 'Negative Reviews'], autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'salmon'])
    plt.title('Distribution of Review Sentiments (Positive vs. Negative)')
    plt.tight_layout()
    class_distribution_path = '/tmp/class_distribution_pie_chart.png'
    plt.savefig(class_distribution_path, bbox_inches='tight')
    plt.close()
    blob = bucket.blob('results/class_distribution_pie_chart.png')
    blob.upload_from_filename(class_distribution_path)

#Confusion Matrix Plot for Model Predictions
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix of Model Predictions')
confusion_matrix_path = '/tmp/confusion_matrix.png'
plt.savefig(confusion_matrix_path, bbox_inches='tight')
plt.close()
blob = bucket.blob('results/confusion_matrix.png')
blob.upload_from_filename(confusion_matrix_path)





