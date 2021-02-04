[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Twitter][twitter-shield]][twitter-url]
[![Facebook][facebook-shield]][facebook-url]


<br />
<p align="center">
  <a href="https://github.com/miccio-dk/itsadive_hybrid_structural_model">
    <img src="logo.png" alt="Logo" width="240" height="240">
  </a>

  <h3 align="center">Hybrid Structural Model for HRTF individualization</h3>
</p>


<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about-the-project">About</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#anthropometric-data-collection">Anthropometric data collection</a></li>
        <li><a href="#prtf-generation">PRTF generation</a></li>
        <li><a href="#hrtf-generation">HRTF generation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


## About
This repository includes the code for a hybrid structural HRTF model combining measured, synthesised, and selected components. In particular, its three components are:
* A generic head-and-torso component, taken from the "pinna-less" KEMAR set included in the Viking HRTF dataset v2 [1] with ITD removed (measured component);
* A fully customized pinna component, built using features related to the shape of the user’s pinnae through deep learning [2] (synthesized component);
* The best-match ITD from an available HRTF dataset [3] obtained by regression on anthropometric parameters of the head and torso [4] (selected component).

The model, implemented in MATLAB/Python, directly outputs a SOFA file.


## Getting Started
To get a local copy up and running follow these simple steps.

### Prerequisites
To run this project you'll need a copy of Python 3.x (3.7 or later recommended) and MATLAB.
If you want to use the generated SOFA file with Steam Audio, it is recommended to use MATLAB 2016.
See [this issue](https://github.com/ValveSoftware/steam-audio/issues/129) for more details.

Make sure to download and install the following MATLAB toolboxes:
* SOFA API (1.1.x): https://sourceforge.net/projects/sofacoustics/files
* Auditory Modeling Toolbox (latest): http://amtoolbox.sourceforge.net

Furthermore, you'll need a copy of the HUTUBS [3] dataset, which can be found [here](https://depositonce.tu-berlin.de/handle/11303/9429) (file `HRIRs.zip`).

Finally, it is recommended to install the necessary Python dependencies within a [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or using [`conda`](https://docs.conda.io/en/latest/miniconda.html).
In the latter case, make sure to install `pip` within your Conda environment first.

### Installation
1. Clone the repo
   ```sh
   git clone git@github.com:miccio-dk/itsadive_hybrid_structural_model.git
   ```
2. Install Python packages
   ```sh
   cd ./itsadive_hybrid_structural_model/python
   pip install -r requirements.txt
   ```


## Usage
In order to generate an individualized HRTF set from user's data, follow these steps.

### Anthropometric data collection
* Place an absolute reference (such as a ruler) behind the ear and take a photo of the pinna from ~30 cm afar.
* Extract the pinna contours
  * Using an image editing program such as GIMP, use the measuring tool to derive the quantity _W_, equivalent to 9.5 cm in pixels.
  * Use the selection tool to mark a square of side _W_ centered at the entrance of the ear canal and crop to selection.
  * Trace the pinna contours manually or with the help of MATLAB's `edge()` function; the areas of interest are the contour of the concha (cymba and cavum), the ear canal, the tragus, and the inner and outer edges of the helix.
  * Scale the image down to 256x256 pixels and store at PNG. 
* Measure the _head width_ from above the ears, the _head depth_ between the forehead and the outermost part of the back, and the _shoulder circumference_ using a soft measuring tape around the chest

### PRTF generation
From within the `python/` directory, run the following command:
```sh
python ear_to_prtf.py configs/edges_median.json /path/to/pinna_contours.png --nfft 512 --output_path /path/to/prtf.mat
```

For more info regarding further arguments and options:
```sh
python ear_to_prtf.py --help
```

### HRTF generation
Open MATLAB and call `generateHrtfSet()` with the following arguments:
* _head\_width_, _head\_depth_, and _shoulder\_circumference_ measured earlier, in centimeters
* the path to the PRTF generated using the Python script
* the output path to the generated SOFA file


## License
Distributed under the MIT License. See `LICENSE` for more information.


## Contact
Simone Spagnol - [@itsadive](https://twitter.com/itsadive) - `ssp (@) create.aau.dk`

Project Link: [https://itsadive.create.aau.dk](https://itsadive.create.aau.dk)



## References
* [1] S. Spagnol, R. Miccini, and R. Unnthórsson (2020). The Viking HRTF dataset v2. DOI: `10.5281/zenodo.4160401`

* [2] R. Miccini and S. Spagnol. HRTF individualization using deep learning. In: _Proceedings of the 2020 IEEE Conference on Virtual Reality and 3D User Interfaces Workshops (VRW 2020)_, pages 390-395, Atlanta, GA, USA, March 2020.

* [3] F. Brinkmann, M. Dinakran, R. Pelzer, J.J. Wohlgemuth, F. Seipl, and Stefan Weinzierl. The HUTUBS HRTF database. DOI: `10.14279/depositonce-8487`

* [4] S. Spagnol. HRTF selection by anthropometric regression for improving horizontal localization accuracy. _IEEE Signal Processing Letters_ 27, pages 590-594, April 2020.


#### This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 797850.





<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/miccio-dk/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/miccio-dk/itsadive_hybrid_structural_model/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/miccio-dk/repo.svg?style=for-the-badge
[forks-url]: https://github.com/miccio-dk/itsadive_hybrid_structural_model/network/members
[stars-shield]: https://img.shields.io/github/stars/miccio-dk/repo.svg?style=for-the-badge
[stars-url]: https://github.com/miccio-dk/itsadive_hybrid_structural_model/stargazers
[issues-shield]: https://img.shields.io/github/issues/miccio-dk/repo.svg?style=for-the-badge
[issues-url]: https://github.com/miccio-dk/itsadive_hybrid_structural_model/issues
[license-shield]: https://img.shields.io/github/license/miccio-dk/repo.svg?style=for-the-badge
[license-url]: https://github.com/miccio-dk/itsadive_hybrid_structural_model/blob/master/LICENSE.txt
[twitter-shield]: https://img.shields.io/badge/-Twitter-black.svg?style=for-the-badge&logo=twitter&colorB=555
[twitter-url]: https://twitter.com/itsadive
[facebook-shield]: https://img.shields.io/badge/-Facebook-black.svg?style=for-the-badge&logo=facebook&colorB=555
[facebook-url]: https://www.facebook.com/itsadive