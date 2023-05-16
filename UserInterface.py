import cv2
import numpy as np
import vlc
import mouse

# Define the parameters of the concentric rings
page = (600, 600, 3)
x = int(page[0] / 2)
y = int(page[1] / 2)
num_rings = 10
radius_step = 25
ring_thickness = 20

# Define the media files to be played
media_files = [f'demo.mp4' for i in range(num_rings)]

# Create the VLC instances and media players for each ring
players = [vlc.MediaPlayer() for _ in range(num_rings)]

# Create the media objects for each file and assign them to the players
for i in range(num_rings):
    media = vlc.Media(f'file://{media_files[i]}')
    players[i].set_media(media)

# Create a blank image to draw the rings on
ring_img = np.zeros(page, np.uint8)

# Draw the rings on the image
for i in range(num_rings):
    cv2.circle(ring_img, (x, y), radius_step * (i+1), (255, 255, 255), ring_thickness)


# Define a function to highlight a ring
def highlight_ring(input_ring, index):
    # Create a new image
    output_img = input_ring.copy()
    # Draw the highlighted ring on the new image
    cv2.circle(output_img, (x, y), radius_step * (index+1), (0, 255, 0), ring_thickness)
    return output_img


# Define a function to un-highlight a ring
def un_highlight_ring(input_ring, index):
    # Create a new image
    output_img = input_ring.copy()
    # Draw the un-highlighted ring on the new image
    cv2.circle(output_img, (x, y), radius_step * (index + 1), (255, 255, 255), ring_thickness)
    return output_img


# Define a function to handle the mouse click event
def handle_mouse_input(event):
    global current_ring_index
    global ring_img

    # Check which ring the mouse is over and stop all other players
    for j in range(num_rings):
        distance = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
        if distance <= (j+1)*radius_step:
            for k in range(num_rings):
                if k != j:
                    players[k].stop()
                    ring_img = un_highlight_ring(ring_img, k)

            # Highlight the selected ring and play its corresponding media file
            ring_img = highlight_ring(ring_img.copy(), j)
            if j == current_ring_index:
                players[j].stop()
                current_ring_index = None
            else:
                players[j].play()
                players[j].set_fullscreen(True)
                current_ring_index = j


current_ring_index = None

# Wait for the user to press the 'ESC' key
while True:
    if cv2.waitKey(1) == 27:
        break

    # Register the handle_mouse_click() function as a callback for mouse click events
    mouse.hook(handle_mouse_input)

    # Stack the ring images vertically and show them
    cv2.imshow("Concentric Rings", ring_img)

# Release the resources
for player in players:
    player.release()
cv2.destroyAllWindows()
