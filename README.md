# IAPET
This repository contains the code for reproducing the results with the training file, in the following paper:  

**IAPET: Illumination Adaptive Prompt Enhancement Transformer for Robust Object Detection in Traffic Surveillance**

This work addresses the challenges posed by varying illumination conditions in traffic surveillance using a novel transformer-based model. We utilize two widely-used datasets: UA-DETRAC and BDD100K, to evaluate the performance of our approach under different conditions.

For more details on the methodology and experiments, please refer to the full paper.

## 1. Dataset Preparation

The following datasets are used for training and evaluation:

1. **UA-DETRAC**: A real-world dataset focusing on vehicle detection and tracking in traffic scenes.
   - Download from [UA-DETRAC](https://paperswithcode.com/dataset/ua-detrac).

2. **BDD100K**: A large-scale dataset for diverse road scenes, covering various conditions.
   - Access from [BDD100K](https://www.vis.xyz/bdd100k/).

Ensure both datasets are placed in the appropriate directories for the training scripts.

## 2. Model Training

To train the IAPET model, you can use the following command:

```bash
 waiting 
```
You may adjust the hyperparameters according to your system's capabilities to optimize the model performance based on available resources.

## 3. Results

### UA-DETRAC Results

The following figures show the comparison of IAPET's performance with other models on the UA-DETRAC dataset. The results highlight the robustness of IAPET under various illumination conditions, showcasing its competitive performance.

<img src="https://github.com/user-attachments/assets/a7e6da23-d1e7-449b-93b0-6ed93a1d6d9d" alt="UA-DETRAC metrics/mAP50(B)" width="600"/>
<br>
<div style="text-align: center;">  
<strong>Figure 1</strong>: Comparison of mAP@50 (mean Average Precision at IoU 50) on the UA-DETRAC dataset.
</div>

As shown in **Figure 2**, the model's performance on metrics like mAP@50-95 further highlights IAPET's adaptability across various precision thresholds.

<img src="https://github.com/user-attachments/assets/0093caca-93e8-4b6f-8b30-f1a44d43b353" alt="UA-DETRAC metrics/metrics/mAP50-95(B)" width="600"/>
<br>
<div style="text-align: center;">  
<strong>Figure 2</strong>: Performance comparison for mAP@50-95 on the UA-DETRAC dataset.
</div>

### BDD100K Results

Next, we evaluate IAPET's performance on the BDD100K dataset, which encompasses more diverse road scenes. The results presented in the following figures reflect the model's strong performance across different environmental conditions.

<img src="https://github.com/user-attachments/assets/315388d5-ff77-4c53-8d4a-659dfe0fc628" alt="BDD100K metrics/mAP50(B)" width="600"/>
<br>
<div style="text-align: center;">  
<strong>Figure 3</strong>: mAP@50 results on the BDD100K dataset.
</div>

Similarly, the comparison in **Figure 4** demonstrates the model's precision across multiple IoU thresholds on the BDD100K dataset.

<img src="https://github.com/user-attachments/assets/59a558ac-4aa0-491d-9c52-aa8e4babf95b" alt="BDD100K metrics/metrics/mAP50-95(B)(B)" width="600"/>
<br>
<div style="text-align: center;">  
<strong>Figure 4</strong>: Comparison on BDD100K for mAP@50-95 across various metrics.
</div>

### Additional Metrics

To further understand the performance of IAPET, we examine the F1-Score and Recall metrics on the BDD100K dataset. **Figure 5** and **Figure 6** illustrate these metrics and highlight the model's balance between precision and recall across different confidence levels.

<img src="https://github.com/user-attachments/assets/2e92c74a-9edb-4533-9f8e-a21c4f4dc08e" alt="BDD100K F1-confidence" width="600"/>
<br>
<div style="text-align: center;">  
<strong>Figure 5</strong>: F1-Score analysis on BDD100K dataset.
</div>

<img src="https://github.com/user-attachments/assets/818411f5-869a-4510-85ab-1b603dfe1b1b" alt="BDD100K Recall-confidence" width="600"/>
<br>
<div style="text-align: center;">  
<strong>Figure 6</strong>: Recall analysis on BDD100K dataset.
</div>

Finally, **Figure 7** presents a comparison of the number of model parameters between IAPET and other approaches, demonstrating the model's efficiency given its competitive performance.

<img src="https://github.com/user-attachments/assets/2ddcd81e-804b-4b15-83a5-cf49b7f65fae" alt="model/parameters" width="400"/>
<br>
<div style="text-align: center;">  
<strong>Figure 7</strong>: Comparison of model parameters across different approaches.
</div>



