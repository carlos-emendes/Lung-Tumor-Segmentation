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
