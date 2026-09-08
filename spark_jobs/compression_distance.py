from pyspark.sql import SparkSession
import zlib
import sys  # <-- Added to catch terminal arguments

# 1. INITIALIZATION & DATA EXTRACTION

spark = SparkSession.builder \
    .appName("Genomics_NCD_Engine") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR") 

# Catch the argument passed from Streamlit (default to 1.0 / 100% if none is provided)
try:
    decimal_fraction = float(sys.argv[1])
except IndexError:
    decimal_fraction = 0.1 

hdfs_base = "hdfs://namenode:9000/precision_medicine/production_data/"
ref_rdd = spark.sparkContext.textFile(hdfs_base + "true_reference_BRCA1.fasta")
patient_rdd = spark.sparkContext.textFile(hdfs_base + "chromosome_22_stress.fasta")

def clean_fasta(rdd):
    return rdd.filter(lambda line: not line.startswith(">")).reduce(lambda a, b: a + b)

print(f"\n[1/3] Extracting DNA from HDFS (Using {decimal_fraction * 100}% of patient sequence)...")
ref_dna = clean_fasta(ref_rdd)
full_patient_dna = clean_fasta(patient_rdd)

# <-- SLICE THE STRING based on the fraction from Streamlit
subset_length = int(len(full_patient_dna) * decimal_fraction)
patient_dna = full_patient_dna[:subset_length]


# 2. DEFINING THE ALGORITHMS (BWT & NCD)

def apply_bwt(sequence):
    """
    Applies the Burrows-Wheeler Transform using a memory-efficient Suffix Array.
    Achieves O(N) spatial complexity by sorting integer pointers instead of matrix rows.
    """
    sequence += "$"
    n = len(sequence)
    suffix_array = sorted(range(n), key=lambda i: sequence[i:])
    bwt_list = []
    for i in suffix_array:
        if i == 0:
            bwt_list.append(sequence[n - 1]) 
        else:
            bwt_list.append(sequence[i - 1])
            
    return ''.join(bwt_list)

def get_compressed_size(sequence):
    """Compresses the BWT string using zlib (dictionary compression) and returns byte size."""
    bwt_string = apply_bwt(sequence)
    compressed_bytes = zlib.compress(bwt_string.encode('utf-8'))
    return len(compressed_bytes)


# 3. DISTRIBUTED EXECUTION

print("[2/3] Broadcasting BWT logic to Spark Workers...")

patients_rdd = spark.sparkContext.parallelize([("Patient_A", patient_dna)])

# Broadcast the healthy reference sequence to all worker nodes so they don't have to re-download it
broadcast_ref = spark.sparkContext.broadcast(ref_dna)

def calculate_ncd(patient_record):
    patient_id, p_dna = patient_record
    r_dna = broadcast_ref.value
    
    # Calculate compressed sizes
    c_ref = get_compressed_size(r_dna)
    c_pat = get_compressed_size(p_dna)
    c_concat = get_compressed_size(r_dna + p_dna) # Compressing them together
    
    # Apply NCD Formula
    ncd_score = (c_concat - min(c_ref, c_pat)) / max(c_ref, c_pat)
    
    return (patient_id, round(ncd_score, 4))

print("[3/3] Calculating Normalized Compression Distance...\n")

# Execute the mapping function across the cluster
results = patients_rdd.map(calculate_ncd).collect()


# 4. RESULTS

print("========================================")
print("     GENOMIC COMPRESSION RESULTS        ")
print("========================================")
for patient_id, score in results:
    print(f"{patient_id} vs Reference NCD Score: {score}")
    
    # Clinical Logic (Back to hardcoded thresholds)
    if score < 0.1:
        print("Diagnosis: Normal / Benign Variation")
    elif 0.1 <= score < 0.4:
        print("Diagnosis: Moderate Structural Mutation Detected")
    else:
        print("Diagnosis: Severe Genetic Divergence Detected!")
print("========================================\n")

spark.stop()