# Sea-Thru

Sea-Thru is an algorithm that removes the veiling effects of water in images taken underwater. It is presented by Derya Akkaynak and Tali Treibitz at CVPR 2019.

Open Access: [Sea-Thru: A Method for Removing Water From Underwater Images](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf)

Bibliographic Info: [10.1109/CVPR.2019.00178](https://doi.org/10.1109/CVPR.2019.00178)

It takes the original image and a depth map of the same size for inputs, and output the recovered image. 

Monodepth is a project for predicting depth map from an arbitrary image. 

One potential issue is the accuracy of monodepth for images taken underwater. 

## Imaging Underwater
Water's effects on an image taken underwater can be divided into two categories: **wideband attenuation** and **backscatter**. Specifically, underwater image $I$ is represented as 
$$I_c = D_c + B_c$$ 
where $c\in \{r,g,b\}$ representing each color channel. $D_c$ is the direct signal containing information of the attenuated scene, and $B_c$ is the backscatter due to water. They can be further decomposed as 
$$
\begin{aligned}
  D_c &= J_c^{-\beta_c^D(\textbf{v}_D)z} \\
  B_c &= B_c^\infty(1-e^{-\beta_c^B(\textbf{v}_B)z})
\end{aligned}
$$
