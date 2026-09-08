import urllib.request
import csv
import random


print("Downloading Chromosome 22 Stress Data...")
stress_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000022.11&seq_start=15000000&seq_stop=16000000&rettype=fasta&retmode=text"
urllib.request.urlretrieve(stress_url, "chromosome_22_stress.fasta")
print("Done!")