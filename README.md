# Segmentation of Lung CT via Python/PyTorch for Tumor Identification

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Lightning: PyTorch](https://img.shields.io/badge/-PyTorch%20Lightning-792ee5.svg)](https://www.pytorchlightning.ai/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

**Developer:** Carlos Eduardo Mendes  
**Contact:** [carloseduardo.mendesf@gmail.com](mailto:carloseduardo.mendesf@gmail.com)

---

## 🔍 Introduction
This project focuses on the automated identification and segmentation of lung tumors in **Computed Tomography (CT)** scans. Utilizing deep learning, the goal is to provide a precise tool for quantifying tumor regions, which is essential for diagnosis and treatment planning.

The implementation uses a **U-Net** architecture—a gold standard in medical imaging—built with **PyTorch** and managed by **PyTorch Lightning** for modularity and scalability.

### Project Highlights
* **Architecture:** U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)).
* **Framework:** PyTorch Lightning (Research-oriented structure).
* **Data Source:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (Task06_Lung).
* **Objective:** 2D Slice-based binary segmentation of tumor vs. lung tissue.

---

## 📂 Dataset
The data is provided by the Medical Segmentation Decathlon challenge.
* **Direct Download:** [NIfTI Dataset Link](https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing)
* **License:** CC-BY-SA 4.0.

---

## 🛠️ Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/carlos-emendes/Lung-Tumor-Segmentation.git](https://github.com/carlos-emendes/Lung-Tumor-Segmentation.git)
cd Lung-Tumor-Segmentation
pip install -r requirements.txt
```


## 🚀 Usage Guide
1. Data Preparation
Extract the downloaded dataset folder into the repository root. Ensure the structure follows the Medical Decathlon standard (NIfTI format).

2. Preprocessing
   Run the following script to convert 3D volumes into 2D slices, apply Hounsfield Unit (HU) normalization, and resize images to $256 \times 256$.
   ```bash
   python Preprocessing.py
   ```
3. Training
Execute the training script to optimize the U-Net. PyTorch Lightning will automatically save the best model checkpoints in the /logs directory.
   ```bash
   python Training.py
   ```

4. Evaluation & Visualization
The Evaluate.py script calculates the Dice Loss and allows you to visualize the CT slices, comparing the Ground Truth (Expert Label) vs. the Model Prediction.
     ```bash
   python Evaluate.py
     ```
**Note**: To load your trained weights, edit line 13 in Evaluate.py to point to your specific .ckpt file path.

---

## 📈 Results & Evaluation

The model performance is evaluated using the **Dice Coefficient** (also known as the F1-score for pixels). This metric measures the spatial overlap between the predicted tumor region ($X$) and the ground truth ($Y$), providing a value between 0 and 1, where 1 represents a perfect match.

$$Dice = \frac{2 |X \cap Y|}{|X| + |Y|}$$

### Qualitative Analysis (Model in Action)
Below is a demonstration of the U-Net model segmenting the lung tumor across multiple CT slices:

<p align="center">
  <video src="ct_with_prediction_id45.mp4" width="800" autoplay loop muted playsinline></video>
  <br>
  <em>U-Net Prediction vs. Ground Truth (Axial View)</em>
</p>

Sequential CT slices showing the original tumor mask (left) vs predicted tumor mask (right).

---
| Metric | Value |
| :--- | :--- |
| **Best Validation Dice Loss** | **0.157** |
| **Optimizer** | Adam |
| **Loss Function** | Binary Dice Loss |

## 🤝 Acknowledgments

* **Coursework:** This project was developed as part of the [Deep Learning with PyTorch for Medical Image Analysis](https://www.udemy.com/course/deep-learning-with-pytorch-for-medical-image-analysis/) course on Udemy.
* **Dataset:** We thank the **Medical Segmentation Decathlon** (Task06_Lung) for providing the open-source data.
* **Architecture:** Based on the original U-Net paper: *U-Net: Convolutional Networks for Biomedical Image Segmentation* by Ronneberger et al.
* **License:** This project is licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
