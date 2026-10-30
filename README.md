# Network Traffic Classifier

The goal of this project is to train a model that can identify network activity.

#### Dataset
Our Dataset can be Downloaded from: [Kaggle | CIC-BCCC-NRC-TabularIoTAttacks-2024](https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024?resource=download)

#### How to Run
1. Clone the repo
Our classifications are:
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


To run and evaluate the training models:
Clone the repo. 3 files are

To ensure the data is loaded properly, either:

### Method 1
* Inside root, create a folder named data, then inside that another named loose_csv, and place all csv files there. 
* When running the program through a terminal, do not add any cli arguments.

### Method 2

* When running the program through terminal, specify the path into the folder where the csv files have been downloaded. 
* Example: "../data/loose_csv/"
* Example: 
```bash
python log_reg.py ../data/loose_csv/
```