import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def predict(image_path):
    # Load an example image
    original_image = cv2.imread(image_path)
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Convert the image to YCrCb color space
    ycrcb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Apply initial segmentation (example: simple skin color segmentation in HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    initial_segmentation_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Pseudo ground truth
    lower_acne_color = np.array([0, 20, 70], dtype=np.uint8)
    upper_acne_color = np.array([20, 255, 255], dtype=np.uint8)
    pseudo_ground_truth_mask = cv2.inRange(hsv_image, lower_acne_color, upper_acne_color)

    # Apply additional processing for final segmentation if needed
    # For demonstration, let's use the initial segmentation as the final segmentation
    final_segmentation = initial_segmentation_mask

    # Display the images with different color maps
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    axs[0, 0].imshow(original_image_bgr)
    axs[0, 0].set_title('Original (BGR)')

    axs[0, 1].imshow(gray_image, cmap='gray')
    axs[0, 1].set_title('Grayscale')

    axs[0, 2].imshow(cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB))
    axs[0, 2].set_title('YCrCb')

    axs[0, 3].imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
    axs[0, 3].set_title('HSV')

    axs[1, 0].imshow(initial_segmentation_mask, cmap='viridis')  # Use 'viridis' colormap for segmentation
    axs[1, 0].set_title('Initial Segmentation')

    axs[1, 1].imshow(pseudo_ground_truth_mask, cmap='plasma')  # Use 'plasma' colormap for ground truth
    axs[1, 1].set_title('Pseudo Ground Truth')

    # Display the final segmentation with a custom colormap
    cmap = plt.cm.colors.ListedColormap(['#ffffff', '#ff0000'], name='custom_map', N=2)  # White for non-acne, Red for acne
    axs[1, 2].imshow(cv2.cvtColor(final_segmentation, cv2.COLOR_GRAY2RGB), cmap=cmap)
    axs[1, 2].set_title('Final Segmentation')

    # Display ground truth mask with a custom colormap
    axs[1, 3].imshow(pseudo_ground_truth_mask, cmap=cmap)
    axs[1, 3].set_title('Pseudo Ground Truth')

    # Convert the plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64

# Example usage:
image_path = "backend/static/test_image.jpeg"  # Update with your actual path
result_image_base64 = predict(image_path)

