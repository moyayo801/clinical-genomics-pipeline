import sys
import zlib
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pyspark.sql.functions as F
from ontology_mappings import GENE_TO_ENSEMBL, DISEASE_TO_NAME

patient_hdfs_path = sys.argv[1]
gene_target = sys.argv[2]
reference_hdfs_path = f"hdfs://namenode:9000/precision_medicine/production_data/true_reference_{gene_target}.fasta"

spark = SparkSession.builder \
    .appName("Single_Patient_Inference") \
    .master("spark://spark-master:7077") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def clean_fasta(rdd):
    return rdd.filter(lambda line: not line.startswith(">")).reduce(lambda a, b: a + b)

def apply_bwt(sequence):
    sequence += "$"
    n = len(sequence)
    suffix_array = sorted(range(n), key=lambda i: sequence[i:])
    return ''.join(sequence[n - 1] if i == 0 else sequence[i - 1] for i in suffix_array)

def get_compressed_size(sequence):
    return len(zlib.compress(apply_bwt(sequence).encode('utf-8')))

try:
    # Phase 1: NCD Calculation
    ref_rdd = spark.sparkContext.textFile(reference_hdfs_path)
    patient_rdd = spark.sparkContext.textFile(patient_hdfs_path)
    
    ref_dna = clean_fasta(ref_rdd)
    patient_dna = clean_fasta(patient_rdd)
    
    c_ref = get_compressed_size(ref_dna)
    c_pat = get_compressed_size(patient_dna)
    c_concat = get_compressed_size(ref_dna + patient_dna)
    
    ncd_score = (c_concat - min(c_ref, c_pat)) / max(c_ref, c_pat)
    
    print(f"$$NCD_SCORE={ncd_score:.4f}")
    
    if ncd_score < 0.1:
        print("$$DIAGNOSIS=Healthy Baseline / Benign Variation")
        print("$$RISKS=None")
    else:
        print("$$DIAGNOSIS=Severe Structural Divergence Detected")
        
        # Phase 2: Instant Inference (No Training)
        model = PipelineModel.load("hdfs://namenode:9000/models/production_als_pipeline")
        diseases_df = spark.read.parquet("hdfs://namenode:9000/models/disease_catalog.parquet")
        
        mapped_gene_id = GENE_TO_ENSEMBL.get(gene_target, gene_target)
        patient_query = diseases_df.withColumn("geneSymbol", F.lit(mapped_gene_id))
        predictions = model.transform(patient_query)
        
        top_risks = predictions.orderBy(F.col("prediction").desc()).limit(3).collect()
        
        if top_risks:
            risk_list = []
            for row in top_risks:
                raw_disease = row['diseaseName']
                
                clean_name = DISEASE_TO_NAME.get(raw_disease, raw_disease)
                score = min(99.9, row['prediction'] * 100)
                
                
                risk_list.append(f"{clean_name}|{score:.1f}")
                
            
            print(f"$$RISKS={','.join(risk_list)}")
        else:
            print("$$RISKS=Target Gene not found in DisGeNET Graph")

except Exception as e:
    print(f"$$ERROR={str(e)}")

spark.stop()