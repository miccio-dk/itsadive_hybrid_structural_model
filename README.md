# Hybrid Structural Model for HRTF individualization 




#### ITS A DIVE, 2020

This repository includes the code for a hybrid structural HRTF model combining measured, synthesised, and selected components. In particular, its three components are:
-	a generic head-and-torso component, taken from the "pinna-less" KEMAR set included in the Viking HRTF dataset v2 [1] with ITD removed (measured component);
-	a fully customized pinna component, built using features related to the shape of the user’s pinnae through deep learning [2] (synthesized component);
-	the best-match ITD from an available HRTF dataset [3] obtained by regression on anthropometric parameters of the head and torso [4] (selected component).

The model, implemented in MATLAB/Python, directly outputs a SOFA file.

#### More technical info...

## References

[1] S. Spagnol, R. Miccini, and R. Unnthórsson (2020). The Viking HRTF dataset v2. DOI: 10.5281/zenodo.4160401

[2] R. Miccini and S. Spagnol. HRTF individualization using deep learning. In: Proceedings of the 2020 IEEE Conference on Virtual Reality and 3D User Interfaces Workshops (VRW 2020), pages 390-395, Atlanta, GA, USA, March 2020.

[3] F. Brinkmann, M. Dinakran, R. Pelzer, J.J. Wohlgemuth, F. Seipl, and Stefan Weinzierl. The HUTUBS HRTF database. DOI: 10.14279/depositonce-8487

[4] S. Spagnol. HRTF selection by anthropometric regression for improving horizontal localization accuracy. IEEE Signal Processing Letters 27, pages 590-594, April 2020.

#### This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 797850.
