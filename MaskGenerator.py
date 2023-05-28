import numpy as np
import cv2
import screeninfo

# Get screen dimensions
screen = screeninfo.get_monitors()[0]
width = screen.width
height = screen.height

# Create blank image with an alpha channel
image = np.zeros((height, width, 4), dtype=np.uint8)

# Make the entire image transparent
image[:, :] = (0, 0, 0, 0)

# Define circle parameters
circle_radius = 470
circle_center = (int(width / 2), int(height / 2))

# Draw a transparent circle on the image
cv2.circle(image, circle_center, circle_radius, (0, 0, 0, 255), -1, cv2.LINE_AA)

# Invert the alpha channel to make the circle transparent and background red
image[:, :, 3] = cv2.bitwise_not(image[:, :, 3])

# Save the image
output_filename = 'blank_image.png'
cv2.imwrite(output_filename, image)
print(f"Image saved as {output_filename}")

# Show the image
cv2.imshow('Blank Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
