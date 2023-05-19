import cv2
import numpy as np
import vlc
import mouse
import screeninfo

# Define screen dimension information
screen = screeninfo.get_monitors()[0]

# Define the parameters of the Log
num_rings = 20
radius = int(720 / 2)
radius_step = int(radius / num_rings)
ring_thickness = radius_step

# Create variables to store mouse position
x, y = 0, 0

# Define the media files to be played
media_files = [f'demo.mp4' for i in range(num_rings)]

# Create the VLC instance and media player
instance = vlc.Instance()
player = instance.media_player_new()

# Clear memory for new window
cv2.destroyAllWindows()

# Create the window
cv2.namedWindow('Log', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Log', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Find the center of the image
ring_x = int(screen.width / 2)
ring_y = int(screen.height / 2)

# Create a blank image to draw the rings on
ring_img = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)


# Define a function to highlight a ring
def highlight_ring(index):
    global ring_img
    clear_rings()
    if index >= 0:
        ring_img = cv2.circle(ring_img, (ring_x, ring_y), radius_step * (index + 1),
                              (255, 0, 0), ring_thickness, cv2.LINE_AA)
    cv2.putText(ring_img, str(index),
            (int(ring_x - cv2.getTextSize(str(index),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] / 2), int(ring_y / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


# Define a function to un-highlight a ring
def clear_rings():
    global ring_img
    ring_img = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)
    for i in range(num_rings):
        ring_img = cv2.circle(ring_img, (ring_x, ring_y), radius_step * (i + 1),
                              (0, 55, 55), ring_thickness, cv2.LINE_AA)


# Show the initial image
cv2.imshow('Log', ring_img)


def get_selected_ring():
    distance = ((x - ring_x) ** 2 + (y - ring_y) ** 2) ** 0.5

    # Check which ring the mouse is over and stop the player if needed
    for j in range(num_rings):
        if (((j + 1) * radius_step) + (ring_thickness / 2)) >= distance \
                >= (((j + 1) * radius_step) - (ring_thickness / 2)):
            return j
    return -1


# Define a callback function to handle mouse events
def handle_mouse_click(event, x, y, flags, param):
    index = get_selected_ring()
    if event == cv2.EVENT_LBUTTONDOWN and index >= 0:
        if not player.is_playing():
            player.set_media(instance.media_new(media_files[index]))
            player.play()
            player.set_fullscreen(True)


# Register the handle_mouse_event() function as a callback for mouse events
cv2.setMouseCallback('Log', handle_mouse_click)

# Wait for the user to press the 'ESC' key
while True:
    # Check if player is playing
    if player.get_state() == 6:
        player.stop()

    x, y = mouse.get_position()

    # Highlight the selected ring
    highlight_ring(get_selected_ring())

    # Show the Log on the screen
    cv2.imshow('Log', ring_img)

    if cv2.waitKey(1) == 27:
        break

# Release the resources
player.release()
cv2.destroyAllWindows()
