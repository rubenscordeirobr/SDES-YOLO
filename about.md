# SDES-YOLO: A High-Precision and Lightweight Model for Fall Detection in Complex Environments

**Authors:** Xiangqian Huang, Xiaoming Li, Limengzi Yuan, Zhao Jiang, Hongwei Jin, Wanghao Wu, Ru Cai, Meilian Zheng, Hongpeng Bai.  
**Journal:** Scientific Reports (Nature Portfolio)  
**Published:** January 15, 2025  
**DOI:** [10.1038/s41598-025-86593-9](https://doi.org/10.1038/s41598-025-86593-9)  
**PMID:** 39814943 | **PMCID:** PMC11735607  

## 1. Abstract Summary

Falling is an emergency situation that can result in serious injury or even death, especially in the absence of immediate assistance. Therefore, developing a model that can accurately and promptly detect falls is crucial for enhancing quality of life and safety. In the field of object detection, while YOLOv8 has recently made notable strides in detection accuracy and speed, it still faces challenges in detecting falls due to variations in lighting, occlusions, and complex human postures.

To address these issues, this study proposes the **SDES-YOLO** model, an improvement based on YOLOv8. 

**Core Architectural Innovations:**
1. **SDFP (Multi-scale Feature Extraction Pyramid):** Enhances feature scaling and hierarchical extraction.
2. **SEAM (Occlusion-aware Attention Mechanism):** Helps the model maintain focus on target subjects even when partially obstructed.
3. **ES3 (Edge and Spatial Information Fusion Module):** Fuses spatial details to handle complex human postures.
4. **WIoU-Shape Loss Function:** A bounding box regression loss tailored for precise fall detection.

### Key Performance Metrics
- **Parameters:** 2.9M (1.33% reduction vs YOLOv8n)
- **Computation:** 7.2 GFLOPs (11.11% reduction vs YOLOv8n)
- **Precision (mAP@0.5):** 85.1% (3.41% improvement over YOLOv8n)

---

## 2. Mathematical Formulations (Converted to LaTeX)

*Note: Due to publisher access restrictions (Nature's robots.txt blocking automated scraping), the complete full-text equations could not be fully extracted directly from the webpage. Based on the abstract's identification of the YOLOv8 and WIoU-Shape loss architecture, the standard generalized mathematical formulas for these modules are reconstructed below in LaTeX.*

### 2.1 Wise-IoU (WIoU) with Shape Penalty
The SDES-YOLO model leverages the **WIoU-Shape loss function** for bounding box regression. The general formulation for WIoU combined with a shape penalty is given by:

$$
\mathcal{L}_{WIoU} = R_{WIoU} \times \mathcal{L}_{IoU}
$$

Where the distance penalty $R_{WIoU}$ uses a dynamic non-monotonic focusing mechanism:

$$
R_{WIoU} = \exp \left( \frac{(x - x_{gt})^2 + (y - y_{gt})^2}{(W_g^2 + H_g^2)^*} \right)
$$

### 2.2 Shape Loss Integration
To capture geometric deformations (human falls and complex postures), the Shape-IoU component modifies the regression loss by penalizing discrepancies in width ($w$) and height ($h$):

$$
\mathcal{L}_{shape} = 1 - IoU + \omega_{shape} \sum_{i \in \{w, h\}} \left( 1 - e^{-\rho_i} \right)
$$

Where $\rho_w$ and $\rho_h$ denote the normalized differences between predicted and ground-truth bounding box dimensions:

$$
\rho_w = \frac{|w - w_{gt}|}{\max(w, w_{gt})}, \quad \rho_h = \frac{|h - h_{gt}|}{\max(h, h_{gt})}
$$

### 2.3 Total Objective Function
The overall training objective $\mathcal{L}_{total}$ integrates the bounding box regression, confidence (objectness), and classification losses:

$$
\mathcal{L}_{total} = \lambda_{box} \mathcal{L}_{WIoU-Shape} + \lambda_{cls} \mathcal{L}_{cls} + \lambda_{dfl} \mathcal{L}_{dfl}
$$

*(Where $\mathcal{L}_{dfl}$ represents the Distribution Focal Loss inherent to YOLOv8 architectures).*
