#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import time


# In[3]:


thres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# In[4]:


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


# In[5]:


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# In[6]:


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# In[7]:


start_time = time.time()
time_limit = 15


# In[8]:


fps_start_time = time.time()
fps_frame_count = 0


# In[9]:


speed_start_time = time.time()
frame_count = 0
total_detection_time = 0


# In[10]:


while True:
    elapsed_time = time.time() - start_time
    if elapsed_time > time_limit:
        print("Time limit reached. Exiting.")
        break
        
    success, img = cap.read()
    if not success:
        print("Error: Unable to read a frame from the video capture.")
        break

    # Measure detection time
    detection_start_time = time.time()
    
    # Detect objects
    try:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        detection_time = time.time() - detection_start_time
        total_detection_time += detection_time

    except cv2.error as e:
        print(f"Error during detection: {e}")
        break

    # Feature added: Total count of objects
    Total_object_count = 0
    class_count = {name: 0 for name in classNames}

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            Total_object_count += 1
            class_name = classNames[classId - 1]
            class_count[class_name] += 1

    # Display the total object count on the image
    cv2.putText(img, f"Objects: {Total_object_count}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    # Display individual object counts
    y_offset = 80
    for class_name, count in class_count.items():
        if count > 0:
            cv2.putText(img, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            y_offset += 30
    # Show the output
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1


# In[11]:


# Release the video capture object and close windows

cap.release()

time.sleep(15)

cv2.destroyAllWindows()


# In[12]:


# # Calculate performance metrics
# end_time = time.time()
# total_time = end_time - start_time
# fps = frame_count / total_time
# average_detection_time = total_detection_time / frame_count


# In[13]:


# print(f"Objects: {Total_object_count}")

# print(f"Total Frames: {frame_count}")

# print(f"Total Time: {total_time:.2f} seconds")

# print(f"FPS: {fps:.2f}")

# print(f"Average Detection Time: {average_detection_time:.4f} seconds")


# In[ ]:





# In[ ]:




