import vlc
import numpy as np
import cv2
import mouse
import requests
import screeninfo
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pytube import YouTube
import pafy

# Define screen dimension information
screen = screeninfo.get_monitors()[0]

# Define the parameters of the Log
num_rings = 52
radius = 1000 / 2
radius_step = int(radius / num_rings)
ring_thickness = int(radius_step)

# Define year counter parameters
starting_year = 1920
ending_year = 2024
num_years = ending_year - starting_year
year_increment = int(num_years / (num_rings - 1))
selected_year = 0

# Create variables to store mouse position
x, y = 0, 0

# Create the VLC instance and media player
instance = vlc.Instance('--no-video-title-show', '--extraintf', 'dummy', '--ffmpeg-hw')
player = instance.media_player_new()


# Create URL list
def is_valid_url(input_url):
    try:
        received_response = requests.head(input_url)
        return received_response.status_code == requests.codes.ok
    except requests.exceptions.RequestException:
        return False


# Set up the credentials
scope = ['https://www.googleapis.com/auth/spreadsheets']
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'dist/RUNTIME DATA/Resources/bchs-log-data.json', scope)

# Authenticate and create the client
client = gspread.authorize(credentials)

# Specify the document ID and sheet name
document_id = '1qDgvbcetqrXAYn734Y_LwQMqHgqy2mmq4H0agc5Z2A4'
sheet_name = 'Sheet1'

# Open the specific sheet in the document
spreadsheet = client.open_by_key(document_id)
worksheet = spreadsheet.worksheet(sheet_name)

# Get all media URLs from the specified column
media_urls = worksheet.col_values(7)
media_labels = worksheet.col_values(1)

# Remove empty cells and corresponding labels
while not str(media_urls[0]).startswith('https:'):
    del media_urls[0]
    del media_labels[0]

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
    global selected_year
    clear_rings()
    if index >= 0:
        selected_year = int((index * year_increment) + starting_year)
        output = 'Year: ' + str(selected_year)
        ring_img = cv2.circle(ring_img, (ring_x, ring_y), radius_step * (index + 1),
                              (130, 64, 37), ring_thickness, cv2.LINE_AA)
    else:
        output = 'Touch to Begin!'
    cv2.putText(ring_img, output,
                (int(ring_x - cv2.getTextSize(output,
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] / 2), int(ring_y / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Define a function to un-highlight a ring
def clear_rings():
    global ring_img
    ring_img = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)
    for i in range(num_rings):
        ring_img = cv2.circle(ring_img, (ring_x, ring_y), radius_step * (i + 1),
                              (27, 182, 252), ring_thickness, cv2.LINE_AA)


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
    video_index = media_labels.index(str(selected_year)) if starting_year <= selected_year <= ending_year else None
    if video_index is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= index < len(media_urls):
        if not player.is_playing():
            video_url = str(media_urls[video_index])
            start_time = int(video_url[video_url.find("?t=") + len("?t="):]) if video_url.find("?t=") != -1 else 0
            try:
                video = YouTube(video_url)
            except Exception as e:
                print(f"Error: {e}")
            else:
                best_stream = video.streams.get_highest_resolution()
                video_url = best_stream.url
                media = instance.media_new(video_url)
                media.add_option(f'start-time={start_time}')
                player.set_media(media)
                player.play()
                player.set_fullscreen(True)  # Set player to full screen


# Register the handle_mouse_event() function as a callback for mouse events
cv2.setMouseCallback('Log', handle_mouse_click)

# Wait for the user to press the 'ESC' key
while True:
    # Check if player is playing
    if player.get_state() == vlc.State.Ended:
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
