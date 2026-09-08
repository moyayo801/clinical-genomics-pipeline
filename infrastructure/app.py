import streamlit as st
import pandas as pd
from pymongo import MongoClient
import subprocess
import time
import os
import sys
import zlib
import os
import re
import subprocess

# ----------------------------------------------------------
# 1. PAGE CONFIGURATION & MINIMALIST AESTHETIC
# ----------------------------------------------------------
st.set_page_config(page_title="Diagnostics Studio", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
        /* Minimalist UI Theme */
        .stApp { background-color: #fbfbfd; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px; color: #1d1d1f; font-weight: 500; font-size: 16px; }
        .stTabs [aria-selected="true"] { color: #0066cc; border-bottom-color: #0066cc; }
        div[data-testid="stMetricValue"] { color: #1d1d1f; font-weight: 600; font-size: 32px; letter-spacing: -0.005em; }
        div[data-testid="stMetricLabel"] { color: #86868b; font-weight: 500; font-size: 14px; }
        .block-container { padding-top: 3rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #1d1d1f; letter-spacing: -0.015em; font-weight: 600; }
        .stButton>button { border-radius: 20px; font-weight: 500; padding: 4px 22px; transition: all 0.2s ease; border: 1px solid #d2d2d7; }
        .stButton>button:hover { border-color: #0066cc; color: #0066cc; background-color: #fbfbfd; }
    </style>
""", unsafe_allow_html=True)

st.title(" Precision Diagnostics")
st.markdown("Distributed Genomic Alignment & ML Pathological Inference Engine - Mohamed Abadine")
st.markdown("---")

# ----------------------------------------------------------
# 2. DATABASE CONNECTION & SESSION STATE
# ----------------------------------------------------------
@st.cache_resource
def init_connection():
    return MongoClient("mongodb://127.0.0.1:27017/?directConnection=true")

client = init_connection()
db = client["precision_medicine"]
collection = db["patient_risks"]

if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'kafka_process' not in st.session_state:
    st.session_state.kafka_process = None

# ----------------------------------------------------------
# 3. TAB ROUTING
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload (Edge Compute)", 
    "Live Ward (Kafka Stream)", 
    "Patient Database", 
    "Hardware Stress Test"
])

# ==========================================================
# TAB 1: MANUAL UPLOAD (TRUE AI INFERENCE)
# ==========================================================
with tab1:
    st.subheader("Single Patient Distributed Diagnostic")
    st.markdown("Upload a `.fasta` sequence to trigger a dedicated Apache Spark job and MLlib matrix factorization.")
    
    uploaded_file = st.file_uploader("Upload Patient DNA Sequence", type=["fasta"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        
        # Determine target gene from filename (default to BRCA1)
        gene_target = "BRCA1"
        for potential_gene in ["BRCA1", "TP53", "CFTR", "APOE", "EGFR"]:
            if potential_gene in filename.upper():
                gene_target = potential_gene
                break
                
        # 1. Save the file temporarily to the mapped /data folder
        local_data_path = os.path.join(os.path.dirname(__file__), '../data/uploaded_patient.fasta')
        with open(local_data_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Initializing Apache Spark & ALS AI Engine (approx. 15-20s)..."):
            
            # 2a. Copy the file from your local Windows machine into the namenode container
            docker_cp_cmd = [
                "docker", "cp", 
                local_data_path, 
                "namenode:/tmp/uploaded_patient.fasta"
            ]
            subprocess.run(docker_cp_cmd, capture_output=True)

            # 2b. Push the file from the container's /tmp folder into HDFS
            hdfs_put_cmd = [
                "docker", "exec", "namenode", 
                "hdfs", "dfs", "-put", "-f", 
                "/tmp/uploaded_patient.fasta", 
                "/precision_medicine/production_data/"
            ]
            subprocess.run(hdfs_put_cmd, capture_output=True)
            
            # 3. Trigger the PySpark Job
            spark_cmd = [
                "docker", "exec", "spark-master", 
                "/spark/bin/spark-submit", 
                "/spark_jobs/single_patient_inference.py", 
                "hdfs://namenode:9000/precision_medicine/production_data/uploaded_patient.fasta",
                gene_target
            ]
            
            try:
                result = subprocess.run(spark_cmd, capture_output=True, text=True, check=True)
                output = result.stdout
                
                # 4. Parse the printed output from Spark
                ncd_match = re.search(r'\$\$NCD_SCORE=(.+)', output)
                diag_match = re.search(r'\$\$DIAGNOSIS=(.+)', output)
                risks_match = re.search(r'\$\$RISKS=(.+)', output)
                
                if ncd_match and diag_match:
                    ncd_score = float(ncd_match.group(1))
                    diagnosis = diag_match.group(1)
                    risks = risks_match.group(1) if risks_match else "None"
                    
                    st.success(f"Distributed Analysis Complete for Target: {gene_target}")
                    st.metric("Normalized Compression Distance (NCD)", f"{ncd_score:.4f}")
                    
                    if ncd_score < 0.1:
                        st.info(f"✅ **Diagnosis:** {diagnosis}")
                    else:
                        st.error(f"⚠️ **Diagnosis:** {diagnosis}")

                    if risks == "Target Gene not found in DisGeNET Graph":
                        st.warning("🧬 **Predicted Pathology Risks (DisGeNET AI):** Target Gene not found in DisGeNET Graph")
                    else:
                        st.markdown("### 🧬 Predicted Pathology Risks")
                        
                        cols = st.columns(3)
                        
                        risk_items = risks.split(',')
                        
                        for i, item in enumerate(risk_items):
                            if "|" in item and i < 3:
                                disease_name, score_str = item.split('|')
                                score_float = float(score_str)
                                
                                with cols[i]:
                                    st.metric(label=disease_name, value=f"{score_str}%")
                                    st.progress(score_float / 100.0)
                        
                    with st.expander("View Raw Spark Output"):
                        st.code(output, language="bash")
                else:
                    st.error("Failed to parse Spark output.")
                    st.code(output, language="bash")
                    
            except subprocess.CalledProcessError as e:
                st.error("Apache Spark execution failed. Ensure containers are running.")
                st.code(e.stderr, language="bash")
# ==========================================================
# TAB 2: LIVE WARD (HOSPITAL SIMULATION)
# ==========================================================
with tab2:
    st.subheader("Real-Time Hospital Sequencer")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶ Start Kafka Sequencer"):
            if not st.session_state.streaming:
                st.session_state.streaming = True
                script_path = os.path.join(os.path.dirname(__file__), 'hospital_sequencer.py')
                st.session_state.kafka_process = subprocess.Popen([sys.executable, script_path])
                st.rerun()
                
    with col2:
        if st.button("⏹ Stop Kafka Sequencer"):
            if st.session_state.streaming and st.session_state.kafka_process:
                st.session_state.kafka_process.terminate()
                st.session_state.streaming = False
                st.session_state.kafka_process = None
                st.rerun()

    # The Live Dashboard Loop
    if st.session_state.streaming:
        st.markdown("""
            <div style="background-color:#e8f5e9; padding:10px; border-radius:8px; color:#2e7d32; font-weight:500; margin-bottom: 20px;">
                🟢 Streaming Connection Active
            </div>
        """, unsafe_allow_html=True)
        
        latest_record = list(collection.find({}, {"_id": 0}).sort("_id", -1).limit(1))
        
        if latest_record:
            df = pd.DataFrame(latest_record)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Patient ID", df['patient_id'].iloc[0])
            with c2:
                st.metric("Predicted Pathology", df['predicted_disease'].iloc[0])
            with c3:
                st.metric("Risk Severity", f"{df['risk_score'].iloc[0]:.1f}%")
                
        time.sleep(2)
        st.rerun()
    else:
        st.markdown("""
            <div style="background-color:#f5f5f7; padding:10px; border-radius:8px; color:#86868b; font-weight:500; margin-bottom: 20px;">
                ⚪ Stream Inactive. Waiting for sequencer start...
            </div>
        """, unsafe_allow_html=True)

# ==========================================================
# TAB 3: PATIENT DATABASE (MONGODB)
# ==========================================================
with tab3:
    st.subheader("Historical Clinical Diagnostics")
    
    if st.button("🔄 Refresh Database"):
        st.rerun()
        
    records = list(collection.find({}, {"_id": 0}))
    
    if records:
        # Flatten the data if it is nested
        df_history = pd.json_normalize(records) 
        st.dataframe(df_history, use_container_width=True) # Use the new 'width' syntax
    else:
        st.info("Database is currently empty. Run the Live Stream to populate patient risks.")

# ==========================================================
# TAB 4: HARDWARE STRESS TEST (SPARK CLUSTER)
# ==========================================================
with tab4:
    st.subheader("Distributed CPU Scaling Benchmark")
    st.markdown("Trigger an alignment job on the massive Chromosome 22 payload inside the Apache Spark cluster.")
    
    data_fraction = st.slider("Select Data Payload Size", min_value=10, max_value=100, value=10, step=10, format="%d%%")
    
    if st.button("⚡ Run Distributed NCD Alignment"):
        decimal_fraction = data_fraction / 100.0
        
        with st.spinner(f"Submitting Spark Job with {decimal_fraction} data fraction..."):
            # Command specifically targeted for the spark-master Docker container
            spark_command = [
                "docker", "exec", "spark-master", 
                "/spark/bin/spark-submit", 
                "/spark_jobs/compression_distance.py", 
                str(decimal_fraction)
            ]
            
            try:
                result = subprocess.run(spark_command, capture_output=True, text=True, check=True,encoding='utf-8', errors='replace')
                st.success("Spark Compute Job Completed.")
                
                # Show the terminal output from PySpark
                st.markdown("**Spark Terminal Output:**")
                st.code(result.stdout, language="bash")
                
            except subprocess.CalledProcessError as e:
                st.error("Apache Spark cluster execution failed. Ensure Docker containers are running.")
                st.code(e.stderr, language="bash")