Based on VideoPose3D.
First, use the detectron2 module to detect 2D keypoints in each frame, which contain the position and confidence of each body part.
Then, use an affinity matrix to measure the similarity between different keypoints, and cluster the keypoints that belong to the same body part according to a similarity threshold, and connect them with lines to form the skeleton of the body.
Next, feed the obtained 2D keypoint sequence into a pre-trained 3D pose estimation model, which uses a spatio-temporal convolutional network to capture the dynamic information in the video, and outputs the 3D keypoint coordinates for each frame.
Finally, project the predicted 3D keypoint coordinates onto the 2D plane, and compare them with the original 2D keypoints, calculate the error between them, and update the model parameters according to the error backpropagation, to make the prediction more accurate.

(1) Bottom-up joint detection. The bottom-up human pose estimation approach is a methodology that eliminates the requirement for prior human detection. It directly identifies body joints from the entire image through joint detection, subsequently employing algorithmic processes to assemble these joints into complete skeletal structures. Compared with top-down approaches, this method effectively mitigates errors and distortions caused by detector imperfections and image cropping operations.

(2) Affinity matrix-based clustering for joint association. An affinity matrix is utilized to represent the likelihood of different joints belonging to the same anatomical structure. Clustering algorithms are then applied to group joints into coherent anatomical segments, which are subsequently connected to form partial skeletal structures.

(3) Pretrained model integration for depth inference. A pretrained deep learning model is employed to infer three-dimensional depth coordinates for each anatomical segment, leveraging both the partial skeletal structures and image features [3]. This process enables the estimation of spatial positions in the three-dimensional coordinate system.

(4) Projection-based iterative refinement with 3D-to-2D optimization. The framework implements:
- An optimization function that updates camera extrinsic parameters to minimize projection loss
- A projection function that maps 3D joint coordinates to 2D image planes
- An iterative optimization loop that computes projection errors between reprojected 2D coordinates and original image annotations, subsequently refining parameters through gradient-based optimization.

This systematic approach enables progressive refinement of 3D pose estimation through cyclic projection comparison and parameter adjustment.


Known bugs：	Unexpected key(s) in state_dict: "layers_conv.4.weight", "layers_conv.5.weight", "layers_conv.6.weight", "layers_conv.7.weight", "layers_bn.4.weight", "layers_bn.4.bias", "layers_bn.4.running_mean", "layers_bn.4.running_var", "layers_bn.4.num_batches_tracked", "layers_bn.5.weight", "layers_bn.5.bias", "layers_bn.5.running_mean", "layers_bn.5.running_var", "layers_bn.5.num_batches_tracked", "layers_bn.6.weight", "layers_bn.6.bias", "layers_bn.6.running_mean", "layers_bn.6.running_var", "layers_bn.6.num_batches_tracked", "layers_bn.7.weight", "layers_bn.7.bias", "layers_bn.7.running_mean", "layers_bn.7.running_var", "layers_bn.7.num_batches_tracked". 
