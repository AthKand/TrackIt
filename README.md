# TrackIt: Offline Mouse Tracking System for Hex-Maze Behavioral Analysis

TrackIt was designed for the meticulous tracking of rodent navigation within a maze the Genzel Lab's Hex-Maze project. Developed as part of a master's thesis internship in 2023, TrackIt streamlines Hex-Maze data analysis by automating tracking to save time and enabling quantification of key metrics like dwell-times and node crossings during probe trials. 

<img src="/resources/hex_upper.png" width="250"> <img src="/resources/hex_lower.png" width="250">

## System Overview
TrackIt employs multi-threaded operations to concurrently capture and analyze video frames, ensuring real-time processing without latency. The system's core functionality relies on contour detection within bounding boxes to accurately monitor mouse movements. TrackIt tracks and logs proximate nodes, essential for mapping the subject's navigational trajectory.

### Key Functionalities and User Commands
- **'t'**: Initiates the tracking process, logging the movement data in real-time.
- **'b'**: Activates a monochrome mode, enhancing contrast until movement is detected.
- **'q'**: Terminates the application.
- **'spacebar'**: Pauses the video feed, allowing ongoing data acquisition and processing.

### Data Logging and Analysis

TrackIt is equipped to save experimental data in two distinct formats for comprehensive analysis:
- A `.log` file, documenting bi-cameral tracker outputs at intervals of 0.066 seconds, facilitating a granular view of the subject's movements.
- A `.txt` file, cataloging the nodes navigated by the subject during each trial, providing insights into spatial memory engagement and strategy.

#### Heatmaps and Dwell-Time Analysis

The project features tools for creating heatmaps based on subject routes and performing dwell-time analysis through node proximity assessment. (see /analysis for more)

<img src="/resources/path.png" width="250">

For path-based analysis individual camera views were stitched and the resulting Homography matrix calculated through perspective warp and image stitching. Manual selection of points adjusts the lower camera view's perspective to match the upper one, complementing the analysis tools like heatmaps and dwell-time assessment.

<img src="/resources/path_stitched.png" width="250">

## Application Usage

To initiate the TrackIt system, the following commands can be used:

For standard operation:
```python
python trackit.py
```

For a graphical user interface (GUI):
```
python qt.py
```
(Note: The GUI version culminates in a standalone executable file for ease of use.)

