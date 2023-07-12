import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN

def cluster(points, epsilon, min_samples):
    # Convert the list of tuples to a NumPy array
    points_array = np.array(points)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(points_array)
    
    # Get the cluster labels assigned to each point
    cluster_labels = dbscan.labels_
    
    # Identify the cluster label with the largest number of points (excluding outliers)
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    
    # Check if any significant cluster exists
    if np.any(unique_labels != -1):
        strong_cluster_label = unique_labels[np.argmax(label_counts[unique_labels != -1])]
        
        # Get the coordinates of the points in the strong cluster
        strong_cluster_points = points_array[cluster_labels == strong_cluster_label]
        
        # Calculate the center of the strong cluster
        strong_cluster_center = np.mean(strong_cluster_points, axis=0)
        
        return strong_cluster_center
    else:
        return None
    
def euclid_dist(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])

def find_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                continue

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            intersection = (x, y)
            
            # Check if the intersection lies within the line segments
            def lies_between(a, b, c):
                return min(a, b) <= c <= max(a, b)

            if (lies_between(line1[0][0], line1[1][0], intersection[0]) and
                lies_between(line1[0][1], line1[1][1], intersection[1]) and
                lies_between(line2[0][0], line2[1][0], intersection[0]) and
                lies_between(line2[0][1], line2[1][1], intersection[1])):
                intersections.append(intersection)
    
    return intersections

cap = cv2.VideoCapture(0)

# Parameters for corner detection
feature_params = dict(
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

# Parameters for optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Take the first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# List to store annotation colors
color = np.random.randint(0, 255, (100, 3))

# Moving average parameters
num_points = 5  # Number of points to consider for moving average
point_buffer = []  # Buffer to store past points

start = time.time()
frames = 1
frame_counter = 0
update_interval = 30  # Update corners every 30 frames

prev_center = (0, 0)
while cv2.waitKey(10) != 27:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_counter += 1
    if frame_counter == update_interval:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update the corner points using goodFeaturesToTrack
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        # Reset the frame counter
        frame_counter = 0
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # Create a new mask image for each frame
        mask = np.zeros_like(frame)
        lines = []
        # Draw the tracks and collect intersection points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # Only take action if the movement is greater than 8px
            if euclid_dist(a, b, c, d) > 1:
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 8)

                # Calculate the direction vector of the ray
                direction = (c - a, d - b)

                # Extend the ray to cover the entire frame width and height
                extended_point = (int(a + direction[0] * frame.shape[1]), int(b + direction[1] * frame.shape[0]))

                # Draw the extended ray
                mask = cv2.line(mask, (int(a), int(b)), extended_point, color[i].tolist(), 5)
                lines.append((
                    (int(a), int(b)), extended_point
                ))
                mask = cv2.circle(mask, (int(a), int(b)), 5, color[i].tolist(), -1)

                # Store the point in the buffer
                point_buffer.append((a, b))

                # Apply moving average to smooth the points
                if len(point_buffer) > num_points:
                    point_buffer.pop(0)  # Remove the oldest point

                # Calculate the average point from the buffer
                avg_point = np.mean(point_buffer, axis=0).astype(int)
                mask = cv2.circle(mask, tuple(avg_point), 5, color[i].tolist(), -1)

    points = find_intersections(lines)
    if len(points) > 0:
        approach = cluster(points, 3, 9)
        if approach is not None:
						# only take action if there is a cluster that is consistent with the previous approach
            if euclid_dist(approach[0], approach[1], prev_center[0], prev_center[1]) < 50:
								
								# appraoch is a tuple representing the point the camera is approaching

                cv2.circle(frame, (int(np.round(approach[0])), int(np.round(approach[1]))), 6, (255, 255, 255), 3)
                cv2.circle(frame, (int(np.round(approach[0])), int(np.round(approach[1]))), 22, (255, 255, 255), 3)
                cv2.circle(frame, (int(np.round(approach[0])), int(np.round(approach[1]))), 38, (255, 255, 255), 3)
                
            prev_center = approach
    
    img = cv2.add(frame, mask)  # Combine frame and mask
    fps = frames / (time.time() - start)
    cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Rays', img)
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    frames += 1

cap.release()
cv2.destroyAllWindows()