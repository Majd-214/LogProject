import cv2
import numpy as np
import vlc
import mouse

# Define the parameters of the concentric rings
x = 500
y = 500
num_rings = 10
radius_step = 25

# Create the rings on a blank image
img = np.zeros((1000, 1000, 3), np.uint8)
for i in range(num_rings):
    cv2.circle(img, (x, y), radius_step * (i+1), (255, 255, 255), 2)

# Define the media files to be played
media_files = ['demo.mp4'.format(i+1) for i in range(num_rings)]

# Create the VLC instances and media players for each ring
vlc_instances = [vlc.Instance() for i in range(num_rings)]
players = [instance.media_player_new() for instance in vlc_instances]


# Define a function to highlight a ring
def highlight_ring(input_ring):
    cv2.addWeighted(input_ring, 0.5, input_ring, 0.5, 0, dst=input_ring)


# Define a function to un-highlight a ring
def un_highlight_ring(input_ring):
    cv2.addWeighted(input_ring, 1, input_ring, 0, 0, dst=input_ring)


# Create the rings for each media file on separate images
ring_images = []

for i in range(num_rings):
    ring_img = np.zeros((1000, 1000, 3), np.uint8)
    cv2.circle(ring_img, (x, y), radius_step * (i+1), (255, 255, 255), 2)
    ring_images.append(ring_img)


# Define a function to handle the mouse move event
def handle_mouse_move(event):
    for j, ring_image in enumerate(ring_images):
        distance = ((mouse.get_position()[0] - x) ** 2 + (mouse.get_position()[1] - y) ** 2) ** 0.5
        if distance <= (j+1)*radius_step:
            highlight_ring(ring_image)
            media = vlc_instances[j].media_new(media_files[j])
            players[j].set_media(media)
            players[j].play()
        else:
            un_highlight_ring(ring_image)
            players[j].stop()


# Show the concentric rings on the screen and wait for mouse events
for ring_img in ring_images:
    cv2.imshow("Concentric Rings", ring_img)


while True:
    mouse.on_click(handle_mouse_move)
    if cv2.waitKey(1) == 27:
        break

# Release the resources
for player in players:
    player.release()
cv2.destroyAllWindows()
