# Dataset - Classroom studies

Author: Mirko Raca <racamirko@gmail.com> <racamirko@protonmail.com>
        2014-2015  in Chili Lab, EPFL

## XML synchronization format

Synchronization files start with N tags (N=number of cameras), listing:
 - filename - video filename
 - heads - head detection, obsolete, not provided
 - offset - offset of the synchronization frame
 - fps - speed of the recordin in frames per second

Persons section repeats per person:
 - id - identity number of the person
 - bestCam - in case of multiple cameras, the index of the best camera
   used for assessing the head orientation of the student
 - row, seat - location of the student
 - and contain *region* subelements for each camera in which the student
   was visible.
 
Region element contains:
 - camIdx - identifies the filename of the video file which was annotated
 - x1, y1, x2, y2 - rectangle coordinates of the person
 - camAngle - estimated horizontal angle at which the camera was
   recording the person
 
In *timeline* section each *mark* element marks the frame at which an
event happened. In case when the event had a duration, there is a matching
Begin/End pair of marks (e.g. BeginAnswer, followed by EndAnswer).

In *period* section we marked the begin and end frame of evaluated periods.
The periods have a time-gap between them which represents the time needed to
fill out the questionnaire (which was not taken into account).
 
## Face detection files

Face detection files are located in the subfolder 05_head_orient/02c_detection_smoothed.
Each row of the CSV file contains:
 - frame number
 - components
 - estimated head angle
 - mixture index (best detector, roughly models the head angle at steps of 90, 60,
   45, 0, -45, -60, -90
 - detection score 
 - bounding box x1, y1, x2, y2 coordinates

## Questionnaire format

The header of the .csv files is summarizes the content:
 - id - id of the participant
 - gender - (m)ale/(f)emale
 - row, seat - location of the person
 - pre_interest, pre_knowledge - self-declared pre-test interest and knowledge
   on the subject of the lecture
 - att, class, material, teacher - per period marks
   - personal attention
   - percieved class attention
   - material interest
   - teacher perception (pasive/uninteresting - energetic/intersting)
 - post-test score - if available
 - notes

There are additional files
 - big_table_class_level.csv - represents the same fields but from all classes
   concatinated. Additional field *is_post* indicates whether the questionnaire
   for that class was filled in-class (False) or post-class (True)
 - big_table_period_level.csv - represents the same data, with each row represents
   a single period. To identify the class/period, additional fields were added
   (class_id, period_id)

## Code

The base Python classes for analysis of data are included with the code. The classes
have different utility functions to load the files. You can find:
 - classroom_layouts - which generates the x,y coordinates of the each seat's
   location in the classroom, as well as the wall coordinates and teacher locations
   (also see the folder "classroom_layout")
 - data_providers - different loading and labeling classes
