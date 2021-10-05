import numpy as np
import cv2

# ============================================================================

CANVAS_SIZE = (600,800)

WORKING_LINE_THICKNESS = 2
FINAL_LINE_THICKNESS = 3
FINAL_LINE_COLOR = (0, 0, 255)
WORKING_LINE_COLOR = (0, 165, 255)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self, img0 = None):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        if img0.all() == None: img0 = np.zeros(CANVAS_SIZE, np.uint8)
        cv2.imshow(self.window_name, img0)    
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            #canvas = np.zeros((img.shape[1], img.shape[0]), np.uint8)
            img = img0.copy()
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(img, np.array([self.points]), False, FINAL_LINE_COLOR, FINAL_LINE_THICKNESS)
                # And  also show what the current segment would look like
                cv2.line(img, self.points[-1], self.current, WORKING_LINE_COLOR, WORKING_LINE_THICKNESS)
            # Update the window
            cv2.imshow(self.window_name, img)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        #canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(img, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, img)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return img

# ============================================================================

if __name__ == "__main__":
    imgPath = "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\20210824_180000_ch2_Moment.jpg"
    # Polygon = [(52, 69), (66, 335), (198, 361), (405, 340), (403, 177), (399, 75), (72, 69)]
    # Polygon = [(19, 29), (455, 29), (439, 382), (27, 371), (18, 31)]
    #imgPath = "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\20210824_180000_ch3_Moment.jpg" 
    # Polygon = [(251, 12), (216, 48), (38, 72), (15, 681), (411, 686), (388, 132), (408, 20)]   
    # Polygon = [(1, 74), (289, 78), (277, 649), (3, 645), (1, 285)]
    #imgPath = "D:\\CodeBucket\\Yolov5_DeepSort_Pytorch\\20210824_180000_ch5_Moment.jpg" 
    # Polygon = [(175, 73), (345, 78), (705, 301), (700, 445), (547, 570), (68, 568), (35, 362), (61, 224), (154, 216), (173, 76)]   
    # Polygon = [(300, 580), (210, 410), (437, 330), (642, 575), (478, 574)]
    
    img = cv2.imread(imgPath)
    pd = PolygonDrawer("Polygon")
    image = pd.run(img)
    cv2.imwrite("polygon_ch2.png", image)
    print("Polygon = %s" % pd.points)