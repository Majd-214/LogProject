import cv2
import numpy as np
import vlc

# Define the parameters of the concentric rings
page = (600, 600, 3)
ring_x = int(page[0] / 2)
ring_y = int(page[1] / 2)
num_rings = 10
radius_step = 25
ring_thickness = 20

# Define the media files to be played
media_files = [f'demo.mp4' for i in range(num_rings)]

# Create the VLC instance and media player
instance = vlc.Instance()
player = instance.media_player_new()

# Create a blank image to draw the rings on
ring_img = np.zeros(page, np.uint8)

# Draw the rings on the image
for i in range(num_rings):
    cv2.circle(ring_img, (ring_x, ring_y), radius_step * (i + 1), (255, 255, 255), ring_thickness)


# Define a function to highlight a ring
def highlight_ring(input_ring, index):
    return cv2.circle(input_ring.copy(), (ring_x, ring_y), radius_step * (index + 1), (0, 255, 0), ring_thickness)


# Define a function to un-highlight a ring
def un_highlight_ring(input_ring, index):
    return cv2.circle(input_ring.copy(), (ring_x, ring_y), radius_step * (index + 1), (255, 255, 255), ring_thickness)


# Define a callback function to handle mouse events
def handle_mouse_event(event, x, y, flags, param):
    global ring_img
    distance = ((x - ring_x) ** 2 + (y - ring_y) ** 2) ** 0.5
    # Check which ring the mouse is over and stop the player if needed
    for j in range(num_rings):
        if (((j + 1) * radius_step) + (ring_thickness / 2)) >= distance \
                >= (((j + 1) * radius_step) - (ring_thickness / 2)):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not player.is_playing():
                    player.set_media(instance.media_new(media_files[j]))
                    player.play()
                    player.set_fullscreen(True)
            ring_img = highlight_ring(ring_img, j)
        else:
            ring_img = un_highlight_ring(ring_img, j)


# Show the concentric rings on the screen
cv2.imshow('Concentric Rings', ring_img)

# Register the handle_mouse_event() function as a callback for mouse events
cv2.setMouseCallback('Concentric Rings', handle_mouse_event)

# Flag to indicate whether player is currently playing
is_playing = False

# Wait for the user to press the 'ESC' key
while True:
    # Check if player is playing
    if is_playing:
        if player.get_state() == vlc.State.Ended:
            player.stop()
            player.release()
            is_playing = False

    # Show the concentric rings on the screen
    cv2.imshow('Concentric Rings', ring_img)

    if cv2.waitKey(1) == 27:
        break

# Release the resources
player.release()
cv2.destroyAllWindows()
