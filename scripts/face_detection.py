# import cv2
# # import face_recognition
# import os

# def detect_faces_in_video(video_path):
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return

#     while True:
#         # Grab a single frame of video
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_frame = frame[:, :, ::-1]

#         # Find all the faces in the current frame of video
#         face_locations = face_recognition.face_locations(rgb_frame)

#         for top, right, bottom, left in face_locations:
#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Display the resulting image
#         cv2.imshow('Video', frame)

#         # Wait for Enter key to stop
#         if cv2.waitKey(25) == 13:
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# detect_faces_in_video('/Users/toheed/PanduAI/backend/scripts/office.mp4')
# # print("file exists?", os.path.exists('/Users/toheed/PanduAI/backend/scripts/office.mp4'))

