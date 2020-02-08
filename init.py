from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


# Function to show array of images (intermediate results)
def show_images(images):
    for i, img in enumerate(images):
        cv2.imshow("image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img_path = "images/Optimized-3.2.jpg"
#
#
#
#
# # Read image and preprocess
# # image = cv2.imread(img_path)
# # image_scaled = cv2.resize(image, None, fx=1, fy=1)
# # cv2.imshow('Scaling - Linear Interpolation', image_scaled)
#

def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int

    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)

    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w
def unwarp(img, src, dst):
    """

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array

    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    # plot

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title('Original Image')

    x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]

    ax2.imshow(img)
    ax2.plot(x, y, color='yellow', linewidth=3)
    ax2.set_ylim([h, 0])
    ax2.set_xlim([0, w])
    ax2.set_title('Target Area')

    plt.show()
    return un_warped


def apply_filter(image):
    """
    Define a 5X5 kernel and apply the filter to gray scale image
    Args:
        image: np.array

    Returns:
        filtered: np.array

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Image')
    plt.show()
    return filtered

def apply_threshold(filtered):
    """
    Apply OTSU threshold

    Args:
        filtered: np.array

    Returns:
        thresh: np.array

    """
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    plt.title('After applying OTSU threshold')
    plt.show()
    return thresh

def detect_contour(img, image_shape):
    """

    Args:
        img: np.array()
        image_shape: tuple

    Returns:
        canvas: np.array()
        cnt: list

    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    plt.title('Largest Contour')
    plt.imshow(canvas)
    plt.show()

    return canvas, cnt

def detect_corners_from_contour(canvas, cnt):
    """
    Detecting corner points form contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list

    Returns:
        approx_corners: list

    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print('\nThe corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]

    plt.imshow(canvas)
    plt.title('Corner Points: Douglas-Peucker')
    plt.show()
    return approx_corners

def example_two():
    """
    Skew correction using homography and corner detection using contour points
    Returns: None

    """
    image = cv2.imread('images/m.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

    filtered_image = apply_filter(image)
    threshold_image = apply_threshold(filtered_image)

    cnv, largest_contour = detect_contour(threshold_image, image.shape)
    corners = detect_corners_from_contour(cnv, largest_contour)

    destination_points, h, w = get_destination_points(corners)
    un_warped = unwarp(image, np.float32(corners), destination_points)

    cropped = un_warped[0:h, 0:w]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    f.subplots_adjust(hspace=.2, wspace=.05)

    ax1.imshow(un_warped)
    ax2.imshow(cropped)


    plt.show()

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)


    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    #show_images([blur, edged])

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)

    show_images([image, edged])
    print(len(cnts))

    # Reference object dimensions
    # Here for reference I have used a 2cm x 2cm square
    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int8")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 3
    pixel_per_cm = dist_in_pixel/dist_in_cm

    # Draw remaining contours
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(cropped, [box.astype("int")], -1, (0, 0, 255), 1)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(tr, br)/pixel_per_cm
        cv2.putText(cropped, "{:.2f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(cropped, "{:.2f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    show_images([cropped])

if __name__ == '__main__':
	example_two()