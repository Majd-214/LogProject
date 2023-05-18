import cv2
import numpy as np
import vlc
import mouse

# Define the parameters of the Log
num_rings = 12
radius_step = 25
ring_thickness = 20

# Define the media files to be played
media_files = [f'demo.mp4' for _ in range(num_rings)]

# Create the VLC instance and media player
instance = vlc.Instance()
player = instance.media_player_new()

# Clear memory for new window
cv2.destroyAllWindows()

# Create the window used for the ring
cv2.namedWindow('Log', cv2.WINDOW_FULLSCREEN)

# Find windows dimensions
window_rect = cv2.getWindowImageRect('Log')

# Find the center of the image
ring_x = int(window_rect[2] / 2)
ring_y = int(window_rect[3] / 2)

# Create a blank image to draw the rings on
ring_img = np.zeros((window_rect[3], window_rect[2], 3), dtype=np.uint8)

# Show the initial image
cv2.imshow('Log', ring_img)


# Define a function to highlight a ring
def highlight_ring(input_img, index):
    return cv2.circle(input_img.copy(), (ring_x, ring_y), radius_step * (index + 1), (0, 255, 0), ring_thickness)


# Define a function to un-highlight a ring
def un_highlight_ring(input_img, index):
    return cv2.circle(input_img.copy(), (ring_x, ring_y), radius_step * (index + 1), (255, 255, 255), ring_thickness)


def clicked(event):
    if isinstance(event, mouse.ButtonEvent) and player.is_playing:
        player.stop()


# Draw the rings on the image
for i in range(num_rings):
    ring_img = un_highlight_ring(ring_img, i)


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


# Register the handle_mouse_event() function as a callback for mouse events
cv2.setMouseCallback('Log', handle_mouse_event)

# Wait for the user to press the 'ESC' key
while True:
    # Check if player is playing
    if player.get_state() == vlc.State.Ended:
        player.stop()

    mouse.on_click(clicked)

    # Show the Log on the screen
    cv2.imshow('Log', ring_img)

    if cv2.waitKey(1) == 27:
        break

# Release the resources
player.release()
cv2.destroyAllWindows()
