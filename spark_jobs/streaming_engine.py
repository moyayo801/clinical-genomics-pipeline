from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, row_number
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel

# Import your translation dictionaries
from ontology_mappings import GENE_TO_ENSEMBL, DISEASE_TO_NAME

# ----------------------------------------
# 1. INITIALIZE STREAMING SESSION
# ----------------------------------------
spark = SparkSession.builder \
    .appName("RealTime_Clinical_Genomics_AI") \
    .config("spark.mongodb.output.uri", "mongodb://mongodb:27017/precision_medicine.patient_risks") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ----------------------------------------
# 2. PRE-LOAD MODEL & CATALOG (Run once on startup)
# ----------------------------------------
print("Loading Production ALS Model and Disease Catalog from HDFS...")
model = PipelineModel.load("hdfs://namenode:9000/models/production_als_pipeline")
diseases_df = spark.read.parquet("hdfs://namenode:9000/models/disease_catalog.parquet")
print("✅ AI Engine Loaded and Ready.")

# ----------------------------------------
# 3. DEFINE THE KAFKA PAYLOAD SCHEMA
# ----------------------------------------
schema = StructType([
    StructField("patient_id", StringType(), True),
    StructField("gene_target", StringType(), True),
    StructField("raw_dna", StringType(), True)
])

# ----------------------------------------
# 4. CONNECT TO KAFKA (readStream)
# ----------------------------------------
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "clinical_genomics") \
    .option("startingOffsets", "latest") \
    .load()

parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# ----------------------------------------
# 5. AI INFERENCE MICRO-BATCH PROCESSOR
# ----------------------------------------
# Create UDFs (User Defined Functions) to translate strings using your dictionaries
map_gene_udf = udf(lambda g: GENE_TO_ENSEMBL.get(g, g), StringType())
map_disease_udf = udf(lambda d: DISEASE_TO_NAME.get(d, d), StringType())

def process_and_write_batch(batch_df, epoch_id):
    """Applies the ALS model to incoming streaming data and writes to Mongo"""
    
    # Skip processing if the micro-batch is empty
    if batch_df.count() == 0:
        return

    # A. Map the incoming common gene name to the Ensembl ID (e.g., BRCA1 -> ENSG0...)
    mapped_df = batch_df.withColumn("geneSymbol", map_gene_udf(col("gene_target")))

    # B. Cross-join with the disease catalog (creates a row for every possible disease for this patient)
    patient_disease_df = mapped_df.crossJoin(diseases_df)

    # C. Run the ALS Model to predict the association score for every pair
    predictions = model.transform(patient_disease_df)

    # D. Isolate the top predicted disease for each patient using a Window function
    windowSpec = Window.partitionBy("patient_id").orderBy(col("prediction").desc())
    top_predictions = predictions.withColumn("rank", row_number().over(windowSpec)) \
                                 .filter(col("rank") == 1)

    # E. Translate the winning disease ID back to English and format the output
    final_output = top_predictions \
        .withColumn("predicted_disease", map_disease_udf(col("diseaseName"))) \
        .withColumn("risk_score", col("prediction") * 100) \
        .select("patient_id", "gene_target", "predicted_disease", "risk_score")

    # F. Push the actual AI predictions to MongoDB
    final_output.write \
        .format("mongo") \
        .mode("append") \
        .save()

# ----------------------------------------
# 6. START THE STREAM
# ----------------------------------------
print("🚀 Spark Structured Streaming Engine Online.")
print("Listening to Kafka Topic: 'clinical_genomics'...")

query = parsed_stream.writeStream \
    .foreachBatch(process_and_write_batch) \
    .outputMode("append") \
    .start()

query.awaitTermination()