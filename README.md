# Network Traffic Classifier

The goal of this project is to train a model that can identify the type of malicious IoT network activity.

### Dataset
Our Dataset can be Downloaded from: [Kaggle | CIC-BCCC-NRC-TabularIoTAttacks-2024](https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024?resource=download)

### How to Run
This repo contains two machine learning models: SVM found in src/svm.py and Logistic Regression found in src/log_reg.py.
- Clone the repo
- Download the dataset: [Kaggle | CIC-BCCC-NRC-TabularIoTAttacks-2024](https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024?resource=download)
- *(Optional)* Inside root, create a folder named data, then inside that another named loose_csv, and place all csv files there. 
- If dataset .csv files are in data/loose_csv/, run from src:
```console
python log_reg.py
```
```console
python svm.py
```
- If dataset is located somewhere else, run from src:
```console
python log_reg.py <location of csv files>
```
```console
python svm.py <location of csv files>
```

---

### Problems loading the dataset?
To ensure the data is loaded properly, either:

#### Method 1
* Inside root, create a folder named data, then inside that another named loose_csv, and place all csv files there. 
* When running the program through a terminal, do not add any cli arguments.

#### Method 2

* When running the program through terminal, specify the path into the folder where the csv files have been downloaded. 
* Example: files located in "../data/loose_csv/"
```bash
python log_reg.py ../data/loose_csv/
```
### Missing modules in python environment?
- requirements.txt is located in project root

---

#### Preliminary classifications of attack types:
 * Benign Traffic
 * DDoS ICMP Flood
 * DDoS UDP Flood
 * DoS ICMP Flood
 * DoS TCP Flood
 * DoS UDP Flood
 * MITM ARP Spoofing
 * MQTT DDoS Publish Flood
 * MQTT DoS Connect Flood
 * MQTT DoS Publish Flood
 * MQTT Malformed
 * Recon OS Scan
 * Recon Ping Sweep
 * Recon Port Scan
 * Recon Vulnerability Scan