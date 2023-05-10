import cv2
import numpy as np
import vlc

# Define the parameters of the concentric rings
x = 500
y = 500
num_rings = 5
radius_step = 50

# Create the rings on a blank image
img = np.zeros((1000, 1000, 3), np.uint8)
for i in range(num_rings):
    cv2.circle(img, (x, y), radius_step * (i+1), (255, 255, 255), 2)

# Define the media file to be played
media_file = 'demo.mp4'

# Create the VLC instance and media player
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()


# Define a function to handle the mouse click event
def handle_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click was inside a ring
        for i in range(num_rings):
            if (i+1)*radius_step > np.sqrt((x-500)**2 + (y-500)**2) > i*radius_step:
                # Play the media file using VLC
                media = vlc_instance.media_new(media_file)
                player.set_media(media)
                player.play()
                break


# Show the concentric rings on the screen and wait for a mouse click
cv2.imshow("Concentric Rings", img)
cv2.setMouseCallback("Concentric Rings", handle_click)
cv2.waitKey(0)
cv2.destroyAllWindows()
