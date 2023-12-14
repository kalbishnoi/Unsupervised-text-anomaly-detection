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
