
<h1 style="color:red"> Segmentation of Lung CT via Python/Pytorch for Tumor identification </h1>
<p>
<hgroup>
<h5>Developer: Carlos Eduardo Mendes<br/>
e-mail:carloseduardo.mendesf@gmail.com</h5>
</hgroup>
</p>
<h2>Introduction:</h2>
<p>This Repository consist in a project available on the course https://www.udemy.com/course/deep-learning-with-pytorch-for-medical-image-analysis/ <br/>
The main goal of this project is to identify Lung Tumor in Computed Tomography(CT) images. For this taks, an Machine Learning (ML) was developed, using Pytorch for ML structure development and Pytorch Lightning for ML training. The structure is a UNET, a stablished ML architecture for segmentation, defined in the following paper : https://arxiv.org/abs/1505.04597

</p>
<p>
The data is provided by the medical segmentation decathlon challenge(http://medicaldecathlon.com/) <br />
You can directly download the full body cts and segmentation maps from: <br />
https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing

(Data License: CC-BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/)
</p>

  <h2>Installing Required Libraries:</h2>
  <p>Libraries can be installed by prompting on the command window: <br/>
  pip install -r .\requirements.txt </p>

  <h2>Usage:</h2>
  <p>Donwload and extract the Dataset folder inside the repository folder. The following part consist in running the python files in the order: <br/>
  1-Preprocessing.py <br/>
  2-Training.py </p>
  <p>The Evaluate.py calculate the Dice Loss and allows visualization of the CT image, comparing the tumor region provided by the dataset and the tumor region identified by the ML.<br/> 
    To load the trained the ML, put the .ckpt checkpoint path on the line 13 ( replace the best_model.ckpt from your checkpoint path). checkpoints are located on folder /logs.</p>

  <p>The repository can be modified for your own purpose/dataset.</p>

  
  

