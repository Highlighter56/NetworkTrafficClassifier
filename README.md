# Network Traffic Classifier

The goal of this project is to train a model that can identify the type of malicious IoT network activity.

### Dataset
Our Dataset can be Downloaded from: [Kaggle | CIC-BCCC-NRC-TabularIoTAttacks-2024](https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024?resource=download)

### How to Run
This repo contains two machine learning models: Decision Tree and Logistic Regression.
- Clone the repo
- Download the dataset: [Kaggle | CIC-BCCC-NRC-TabularIoTAttacks-2024](https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024?resource=download)
- *(Optional)* Inside root, create a folder named `data`, then inside that another named `raw_data`, and place all CSV files there.
- Run from the project root:
```console
python src/main.py
```
To force cleaned data and models to be regenerated:
```console
python src/main.py -i
```
The same command works from inside `src`:
```console
python main.py
```
If the dataset is located elsewhere, provide its raw-data directory:
```console
python src/main.py C:\path\to\raw_data
```
Add `-i` to force regeneration with a custom directory:
```console
python src/main.py -i C:\path\to\raw_data
```

---

### Problems loading the dataset?
To ensure the data is loaded properly, either:

#### Method 1
* Inside root, create a folder named `data`, then inside that another named `raw_data`, and place all CSV files there.
* Run `python src/main.py` from the project root or `python main.py` from `src`.

#### Method 2

* Specify the path into the folder where the CSV files have been downloaded.
* Example:
```bash
python src/main.py C:\data\raw_data
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