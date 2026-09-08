from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator 
import pyspark.sql.functions as F


spark = SparkSession.builder \
    .appName("ALS_Batch_Training") \
    .master("spark://spark-master:7077") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("--- [1/5] Ingesting and Cleaning Edge List (Parquet) ---")


parquet_path = "hdfs://namenode:9000/precision_medicine/production_data/model_training_data"
raw_df = spark.read.parquet(parquet_path)


df = raw_df.select(
    F.col("targetId").alias("geneSymbol"),
    F.col("diseaseId").alias("diseaseName"),
    F.col("associationScore").cast("float").alias("associationScore")
)

df_clean = df.dropna(subset=["diseaseName", "geneSymbol", "associationScore"])

print(f"Total valid rows for training: {df_clean.count()}")

print("--- [2/5] Building Machine Learning Pipeline ---")


gene_indexer = StringIndexer(inputCol="geneSymbol", outputCol="gene_id", handleInvalid="skip")
disease_indexer = StringIndexer(inputCol="diseaseName", outputCol="disease_id", handleInvalid="skip")

als = ALS(
    maxIter=15, 
    regParam=0.01, 
    userCol="gene_id", 
    itemCol="disease_id", 
    ratingCol="associationScore", 
    coldStartStrategy="drop"
)


pipeline = Pipeline(stages=[gene_indexer, disease_indexer, als])

print("--- [3/5] Train/Test Split & Model Evaluation ---")

training_data, testing_data = df_clean.randomSplit([0.8, 0.2], seed=42)


model = pipeline.fit(training_data)

# Test the model on the 20% unseen split
predictions = model.transform(testing_data)

# Calculate the RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="associationScore", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print(f"🎯 Model RMSE (Root Mean Square Error): {rmse:.4f}")
print("*(Note: An RMSE closer to 0 indicates higher precision in association score predictions)*")

print("--- [4/5] Retraining Production Model on Full Dataset ---")

production_model = pipeline.fit(df_clean)

print("--- [5/5] Saving Production Model & Disease Catalog to HDFS ---")
# Save the full trained AI pipeline
production_model.write().overwrite().save("hdfs://namenode:9000/models/production_als_pipeline")

# Save a simple list of all diseases so the inference script knows what to predict against
diseases_df = df.select("diseaseName").distinct()
diseases_df.write.mode("overwrite").parquet("hdfs://namenode:9000/models/disease_catalog.parquet")

print("✅ SUCCESS: Evaluated Model and Catalog persisted to HDFS.")
spark.stop()