![image](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/59cb946a-9fe1-4b68-bfee-31a15e58788e)## Running code
## Current methods drawbacks
Current methods use reconstructive/generative approach. They require hand crafted post processing techniques for anomaly localisation.
![current methods](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/7b9c8a35-da2a-4402-acc0-082dc13e463e)
## Problem Statement
**Unsupervised discriminatively trained reconstruction method for text anomaly detection**
## Our Solution
Use a **reconstructive + discriminative** approach. 
Model will inherently capture the information about the certain portions of text which makes it anomalous by learning **joint representation of input data** which is both anomalous and non-anomalous(reconstructed) along with learning the **decision boundary** between them. 
Hence, facilitating **direct anomaly localisation**.
Training data contains only the non-anomalous data, its anomalous part is generated using **pseudo text anomaly generation model**.
![our solution](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/85296d5a-a3cb-40bf-93d0-b64cd7be87f1)
## Architecture Diagram
![Training](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/fe891fea-fceb-4028-be4c-6d5e9b16fe45)
![Testing](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/8b8f5f01-bd08-4e27-8f76-1f442a133767)
## Pipeline for Pseudo Text Anomaly Generation Model
![Pipeline Pseudo Anomaly Generation](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/1e3806ca-8d63-43df-87db-7e418260002e)
## Components of the Model
![Components of the Model](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/863d6c7a-65d4-42a1-8a7b-68402e9ec179)
![Pseudo Text Anomaly Generation](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/f9f2ac5a-6bf5-4750-a899-f048f664febe)
![Anomaly Detection and Localisation + mask prediction](https://github.com/kalbishnoi/Unsupervised-text-anomaly-detection/assets/140685270/7e339b2e-71fd-4383-ac14-59cc1de7a00f)


