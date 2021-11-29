# ParticleDetectorDetector
This AI is meant for use with a cloud chamber. It takes an input frame of the cloud chamber and will output the number of Alpha Particles in the cloud chamber.

Two different methods were employed to count the number of alpha traces in the cloud chamber. The first was a simple CNN network. This method provided high accuracy (~98% when optimized) however it had the issue of overfitting to the training data. The other method used was training a YOLO model. This method worked well and would locate individual particles in the frame, providing a nice visual.
