# Car Counter

This project is created to count car objects in a traffic video using yolov5 and tensorflowlite model. Frames are saved into the working directory when the center of the car object's bounding boxes touch the threshold line. Then, frame information is saved in an images table in PostgreSQL (default postgres database with postgres user).

## Instructions

###Create a virtual environment
#### For Linux:

Run: 
	$ virtualenv virtualenv_name
	$ source virtualenv_name/bin/activate


### Install requirements

Navigate to the project directory: $ cd yolov5vehiclecount

Clone yolov5: $ git clone https://github.com/ultralytics/yolov5 

Run: $ pip install -r requirements.txt

### Launch the program

Run: $ python main.py

## Demo

