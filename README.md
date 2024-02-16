# MPSC_video_classifier_demo

## Install instruction
- Clone this git:  git clone https://github.com/chughgarvit/mpsc_video_classifier_demo.git
- Install all the dependencies (PyTorch, Flask)
- Go to the directory cd mpsc_video_classifier_demo
- Run the py file python app.py
- Open the web browser
- Go to the following address localhost:8080

## Storyline
Welcome to our interactive demo on self-supervised learning from multiview video data for activity recognition and pose detection. In this demo, we'll showcase two approaches to classifying activities in video data: using a single-view model trained in a supervised manner and our innovative multi-view model trained through self-supervised learning.

Imagine a scenario where we can access multiview data captured from cameras placed on the ground and drones flying overhead. This setup gives us rich, multi-perspective views of the environment and the activities taking place within it.

Let's dive into our demo. We have 12 examples of different activities that can be selected from the dropdown menu. When we click "Classify using single-view model," we'll see the results based on a model trained from only one view in a supervised way. This model might struggle with variations in viewpoint and occlusions, typical challenges in single-view activity recognition.

Now, let's try "Classify using the multi-view model." This time, we're leveraging our innovative approach, where we've developed a view-invariant model using partially labeled data. By exploiting optimal losses given partially labeled video, we've built a machine learning architecture to recognize activities across different views and handle occlusions more effectively.

As we switch between the single-view and multi-view models, pay attention to how each model performs on the selected examples. You'll notice that the multi-view model offers more robust and accurate predictions, thanks to its ability to leverage information from multiple perspectives.

Our objectives in developing this system were clear: to design a view/occlusion-invariant human activity and pose detection model, to create scalable video frame classification techniques using unlabeled multiview video data, and to validate the performance of our system using both public and in-house multiview data streams.

Through self-supervised deep learning methods, we've learned to separate video frames in low-dimensional spaces, allowing us to extract meaningful features and representations that generalize well across different views.

In conclusion, our interactive demo showcases the power of self-supervised learning and multiview data in activity recognition and pose detection. By harnessing the richness of multiview data and developing innovative learning techniques, we're pushing the boundaries of what's possible in understanding and analyzing human activities in complex environments.

Thank you for exploring our demo, and we hope you enjoyed witnessing the capabilities of our approach firsthand.
