from kafka import KafkaProducer
import json
import time
import random

# Connect to the local Kafka container
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    api_version=(2, 8, 1), 
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# A list of genes our MLlib model knows how to process
target_genes = ["BRCA1", "TP53", "EGFR"]

print("🧬 Hospital Sequencer Online. Initializing real-time stream...")
print("Press Ctrl+C to stop the sequencer.\n")

patient_counter = 1

try:
    while True:
        patient_id = f"Patient_{patient_counter}"
        gene = random.choice(target_genes)
        
        # Simulate generating a raw DNA sequence string
        sequence = ''.join(random.choices(['A', 'C', 'T', 'G'], k=150))
        
        payload = {
            "patient_id": patient_id,
            "gene_target": gene,
            "raw_dna": sequence
        }
        
        # Fire the data into the Kafka 'clinical_genomics' topic
        producer.send('clinical_genomics', payload)
        print(f"[+] Sequenced {patient_id} -> Target: {gene} -> Pushed to Kafka Broker")
        patient_counter += 1
        time.sleep(3) 
        

except KeyboardInterrupt:
    print("\n🛑 Hospital Sequencer Offline.")