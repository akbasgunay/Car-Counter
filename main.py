import numpy as np
import torch
import cv2
import os
import psycopg2
import datetime


def create_connection():
    # Connect to the database
    # using the psycopg2 adapter.
    # Pass your database name ,# username , password , 
    # hostname and port number
    # I used the postgres user with password '123'
    conn = psycopg2.connect(dbname='postgres',
                            user='postgres',
                            password='123',
                            host='localhost',
                            port='5432')
    curr = conn.cursor()
    return conn, curr
  
def create_table():
    try:
        conn, curr = create_connection()
        try:
            # Create table to store the image information
            curr.execute("CREATE TABLE IF NOT EXISTS \
            images(id INTEGER PRIMARY KEY, line_name TEXT, obj_name TEXT, frame_time TEXT, frame_path TEXT)")
              
        except(Exception, psycopg2.Error) as error:
            # Print exception
            print("Error while creating cartoon table", error)
        finally:
            # Close the connection object
            conn.commit()
            conn.close()
    finally:
        pass
  
def write_blob(id, line_name, obj_name, frame_time, frame_path):
    try:
        conn, cursor = create_connection()
        try:           
            # Execute the INSERT statement
            # Convert the image data to Binary
            cursor.execute("INSERT INTO images\
            (id, line_name, obj_name, frame_time, frame_path) " +
                    "VALUES(%s,%s,%s,%s, %s)",
                    (id, line_name,obj_name, frame_time, frame_path))
            # Commit the changes to the database
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while inserting data in images table", error)
        finally:
            # Close the connection object
            conn.close()
    finally:
        # Since we do not have to do
        # anything here we will pass
        pass



def main():
    create_table()

    cap=cv2.VideoCapture("video.mp4")

    #model path
    path='yolov5/yolov5m-fp16.tflite'

    model = torch.hub.load('yolov5', 'custom', path,source='local')

    obj = model.names[2] = 'car' # The object I would like to count


    size=640

    color=(0,0,255)

    offset=5
    counter = 0
    count = 0

    while True:
        ret,img=cap.read()
        

        img=cv2.resize(img,(800,800)) # Resize the frame
        cv2.line(img, (79, 400), (799, 400), (0, 255, 0), 2)

        results = model(img, size)
        bbox = results.pandas().xyxy[0] 
        for index, row in bbox.iterrows():
            cv2.putText(img, f'Araba nesnesi: {str(counter)} adet', (76, 51), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            cl = (row['class'])

           # If the object is car draw a rectanle around it
            if cl==2:
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                center = int((x1+x2)/2), int((y1+y2)/2)
                center_x = center[0]
                center_y = center[1]

                cv2.circle(img, (center_x, center_y), 3, (0, 255, 0), -1)

                if center_y < (400 + offset) and center_y > (400 - offset):
                    counter = counter + 1
                    cv2.line(img, (79, 400), (799, 400), (0, 255, 255), 2)
                    path = os.getcwd()
                    cv2.imwrite(os.path.join(path , f'{counter}.jpg'), img)
                    dt = str(datetime.datetime.now())
                    write_blob(counter, "threshold line", 'car', dt, os.path.join(path ,f'{counter}.jpg'))

        cv2.imshow("IMG",img)
        
        if cv2.waitKey(1)&0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()