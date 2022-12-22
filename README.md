# Sea-Thru

Sea-Thru is an algorithm that removes the veiling effects of water in images taken underwater. It is presented by Derya Akkaynak and Tali Treibitz at CVPR 2019.

Open Access: [Sea-Thru: A Method for Removing Water From Underwater Images](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf)

Bibliographic Info: [10.1109/CVPR.2019.00178](https://doi.org/10.1109/CVPR.2019.00178)

It takes the original image and a depth map of the same size for inputs, and output the recovered image. 

[MiDaS](https://github.com/isl-org/MiDaS) (by Intel Intelligent Systems Lab) is a project for predicting depth map from an arbitrary image. In this implementation, we explore the possibility of using monocular depth interpreting techniques to facilitate the recovery by Sea-thru.

The authors did not provide any code accompanying the original paper, hence all of the implementation is done by myself. The images and depth maps are provided by the authors (see `Data/Data.md`).

## Implementation Details
For more details about the implementation, and brief explanation of the original paper, please refer to my [blog post](https://teragion.github.io/sea-thru).

## TODO
I will write about the command line APIs later.
