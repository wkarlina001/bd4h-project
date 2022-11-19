<a name="readme-top"></a>

<div>
<h2 align="center">BDH Reproducibility Challenge</h3>
  <h3 align="center">
   Paper Title : Application of deep and machine learning techniques for multi-label
classification performance on psychotic disorder diseases
    <br />
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
     <li>
      <a href="#usage">Usage</a>
    </li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
This project is about reproducing the work published by the authors. Authors proposed models (deep learning and machine learning models) to classify psychotic disorder diseases. We have obtained the same dataset used by the authors and recreated the work as outlined in the paper.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
1. First, you will need to install [git](https://git-scm.com/), if you don't have it already.

2. Next, clone this repository by opening a terminal and typing the following command:
  ```
  git clone https://github.com/wkarlina001/bd4h-project.git
  ```

3. Create conda environment using environment.yml in the directory by running the following command:
  ```
  conda env create -f environment.yml
  conda activate bdh-project
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
The main code of this reproducibility challenge is main.py. The script includes cleaning and preparing data as well as training models. There is also an option to train models with and without Synthetic Minority Oversampling Technique (SMOTE). All plots/figures are saved in figure folder. The best deep neural network model is saved in model folder.

To run the script, please run the following command :
  ```
  python main.py --smote # with SMOTE
  python main.py --no-smote # without SMOTE
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgements

* I. Elujide, S.G. Fashoto, B. Fashoto, E. Mbunge, S.O. Folorunso, J.O. Olamijuwon, et al. (2021) Application of deep and machine learning techniques for multi-label classification performance on psychotic disorder diseases. Informatics Med Unlocked, 23: Article 100545. https://doi.org/10.1016/j.imu.2021.100545


<p align="right">(<a href="#readme-top">back to top</a>)</p>


