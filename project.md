Project title: SoccerVision: Automated Player and Ball Tracking for Soccer Analytics
Students: Abel Dagne, Laszlo Bollyky
One sentence summary of your project:
We are developing a cv tool that processes soccer game images/footage to automatically
detect, track, and classify players, identify teams, and locate the ball then generate a birds-eye
2d mapping of the game from the input image/video.
Background (prior works and methods, existing benchmarks…etc):
Player and object tracking in sports is well-studied, but most existing systems rely on expensive
multi-camera setups or GPS trackers that only capture limited data (e.g., distance covered,
heart rate). These methods don’t provide full player positioning or tactical insights. Existing
computer vision techniques, including feature tracking (KLT), Histogram of Oriented Gradients
(HOG), and deep learning-based object detection (CNNs), have been applied to soccer, but
often fail to deliver the accuracy and features teams need—especially when relying on
broadcast footage with changing angles and occlusions. Our unique access to panoramic and
3D footage allows for more complete player tracking, consistent field coverage, and improved
team identification. We also have manual play-pattern mappings created by players and
coaches, providing valuable ground truth for evaluation.
Your proposed method and evaluation:
1. Extract Images from current game footage: We are going to first get a sufficient
number of images from different games from which we have existing footage to ensure
that our project will work with different colors/teams.
2. Isolate the field: We will then use the same process we used for lane detection in HW2
(canny edge detector & Hough transform) to isolate just the field to simplify and ensure
consistency.
3. Feature Detection & Tracking: We use Harris Corners and SIFT to detect distinctive
player and ball features.
4. Team Classification: We employ color-based segmentation (HSV color space analysis)
to distinguish teams based on jersey colors.
5. 2-dimensional output: We map the tracked coordinates of the field in the footage to a
2D plane (using some of the same techniques employed in HW4), enabling a birds-eye
view of the tracked information.
6. Evaluation: We will evaluate the system using existing manually annotated game
footage, comparing tracked player and ball positions to existing interpretations to assess
accuracy and potential improvements.
Anything else you’d like us to know:
If time permits, we would ideally begin to try and string together a number of images in a row to
begin to process video. We realize that this will likely not be feasible within the time frame,
especially as there are a number of additional features that would need to be incorporated, but
we would love to explore it outside the scope of this course as it could prove useful to the team.
TA you’d like to work with (Abi/Shruti/Ayush/Ian): Ayush