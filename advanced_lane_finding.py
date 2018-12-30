from image_thresholds import *
import cv2
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        #Number of recent values to average
        self.n = 8
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
   
    def update_poly_fit(self, fit):
        if fit is not None:
            if self.best_fit is not None:
                # find how far away this fit from the best fit
                self.diffs = abs(fit - self.best_fit)
            # If it is too far from best fit, reject this fit
            if (self.diffs[0] > 0.001 or \
                self.diffs[1] > 1.0   or \
                self.diffs[2] > 100.0) and \
                len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > self.n:
                    # remove old values and keep recent 'n' values
                    self.current_fit = self.current_fit[len(self.current_fit)-self.n:]
                # Best fit is the average of recent good fits
                self.best_fit = np.average(self.current_fit, axis=0)
                #print("Best vs fit", self.best_fit, fit)
        else:
            self.detected = None

left_line = Line()
right_line = Line()
        
# This function finds lane lines using histogram 
# and sliding window method.
def sliding_window_lane_search(binary_warped, visualize=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # TO-DO : Following line is required only for visualization ??
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 30

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        # Ram  Commented.
        '''
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        '''
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
   
    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print("exception in sliding window poly fit")
        return None, None

    if visualize is True:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]     # RED  for left lane pixels
        out_img[righty, rightx] = [0, 0, 255]   # BLUE for left lane pixels

        # Plots the left and right polynomials on the lane lines in 'yellow' color
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        
        plt.imshow(out_img)
        plt.show()
    
    return (left_fit, right_fit)

# Search for lane lines around the given a polynomial.
def search_around_poly(binary_warped, left_fit, right_fit, visualize=False):
    #print("ssssssssssss")
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 80

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > left_fit[0]*nonzeroy**2+
                      left_fit[1]*nonzeroy+left_fit[2] - margin) &
                      (nonzerox < left_fit[0]*nonzeroy**2+
                      left_fit[1]*nonzeroy+left_fit[2] + margin))
    
    right_lane_inds = ((nonzerox > right_fit[0]*nonzeroy**2+
                      right_fit[1]*nonzeroy+right_fit[2] - margin) &
                      (nonzerox < right_fit[0]*nonzeroy**2+
                      right_fit[1]*nonzeroy+right_fit[2] + margin))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    #left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    ### Fit a second order polynomial to each with np.polyfit() ###
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print("exception in search around poly fit")
        return None, None
    
    
    if visualize is True:
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]
    
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(result)
        plt.show()
        
        ## End visualization steps ##
        
    return left_fit, right_fit

# Based on the given polynomial coefficients for left
# and right lanes, this function returns the predicted lane line.    
def get_predicted_lane(img_shape, left_fit, right_fit):

    # Generate y values
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    ### Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty
    
# Calculates the curvature of polynomial functions in meters.    
def measure_curvature_real(h, left_fitx, right_fitx):
    # Define conversions in x and y from pixels space to meters
    #ym_per_pix = 30/720 # meters per pixel in y dimension
    ym_per_pix = 3.0/100 # meters per pixel in y dimension
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension
    xm_per_pix = 3.7/380 # meters per pixel in x dimension
    
    y_points = np.linspace(0, h-1, h)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(y_points)
    
    # Fit polynomial based on real world space
    left_fit_world_space = np.polyfit(y_points*ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_world_space = np.polyfit(y_points*ym_per_pix, right_fitx * xm_per_pix, 2)
    
    left_curverad = ((1 + (2*left_fit_world_space[0]*y_eval*ym_per_pix + left_fit_world_space[1])**2)**1.5) / np.absolute(2*left_fit_world_space[0])
    right_curverad = ((1 + (2*right_fit_world_space[0]*y_eval*ym_per_pix + right_fit_world_space[1])**2)**1.5) / np.absolute(2*right_fit_world_space[0])
    
    #print(left_curverad, right_curverad)
    return (left_curverad, right_curverad)

def draw_lane_lines(undist, warped, left_fitx, right_fitx, ploty, src, dst, visualize=False):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    if visualize is True:
        plt.imshow(result)
        plt.show()
    
    return result

# Returns binary thresolded image.
def get_thresholded_image(img):
    ## Create Binary thresholded image
    hls_lselect_img = hls_lselect(img, thresh=(200,255))                          
    lab_bselect_img = lab_bselect(img, thresh=(200,255))
    
    # Using HLS-L Channel to pick up white lines and 
    # LAB-B channel to detect yellow lines.
    threshold_img = np.zeros_like(hls_lselect_img)
    threshold_img[(hls_lselect_img == 1) | (lab_bselect_img == 1)] = 1  
    return threshold_img
    
# This function implements the image pipeline.
# Given an image, it finds the lane lines, draws markers etc.    
def image_pipeline(img):    
    global left_line
    global right_line
    
    # undistort the image 
    undist_img = undistort(img, mtx, dist)
    
    # Get thresholded image
    threshold_img = get_thresholded_image(undist_img)

    ## Perspective transform

    # Hardcoded src and dst for perspective transform
    # TODO: Is there a way to find src and dst programatically ?
    h,w = threshold_img.shape[:2]
    src = np.float32([(575,464),
                      (707,464), 
                      (258,682), 
                      (1049,682)])
                      
    dst = np.float32([(450,0),
                      (w-450,0),
                      (450,h),
                      (w-450,h)])
                      
    binary_warped = warp(threshold_img, src, dst)

    if (not left_line.detected) or (not right_line.detected):
        ## Find lane lines using sliding window search method
        left_fit, right_fit = sliding_window_lane_search(binary_warped)
    else:
        # Search around polynimial computed in the previous iteration
        left_fit, right_fit = search_around_poly(binary_warped, left_line.best_fit, right_line.best_fit)
        #left_fit, right_fit = None, None
    
    # Compute the width of the lane based on the left and right x-intercepts.
    if left_fit is not None and right_fit is not None:
        left_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_width = abs(right_x_int - left_x_int)
        #print(lane_width)
        all_lane_width.append(lane_width)
        if (lane_width < 280 or lane_width > 480):
            left_fit = None
            right_fit = None
    
    left_line.update_poly_fit(left_fit)
    right_line.update_poly_fit(right_fit)

    # Get the predicted lane lines
    left_fitx = None
    right_fitx = None
    
    #print(left_line.best_fit, right_line.best_fit)
    if left_line.best_fit is not None and right_line.best_fit is not None:
        # compute the lane lnes based on best fit polynomial coefficients
        left_fitx, right_fitx, ploty = get_predicted_lane(binary_warped.shape, left_line.best_fit, right_line.best_fit)
        
        # Compute the radius of curvature
        left_radius, right_radius = measure_curvature_real(h, left_fitx, right_fitx)
        avg_radius = (left_radius + right_radius)/2
        radius_str = "Radius : %.3f m" % avg_radius
        
        # Compute offset from the center of lane
        center_of_lane = (right_fitx[h-1] + left_fitx[h-1])/2
        center_offset_meters = abs(w/2 - center_of_lane) * (3.7/700)
        center_offset_str = "Center offset: %.3f m" % center_offset_meters
        
        # Draw rectangle that covers the lane ahead of us
        out_img = draw_lane_lines(undist_img, binary_warped, left_fitx, right_fitx, ploty, src, dst)

        # Annotate the image with radius and center offset information
        cv2.putText(out_img, radius_str , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
        cv2.putText(out_img, center_offset_str, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    else:
        out_img = img

    return out_img
    
# Load the camera calibration coefficients
camera_cal_values = pickle.load(open("calibration.p", "rb"))
mtx = camera_cal_values["mtx"]
dist = camera_cal_values["dist"]

all_lane_width = []

'''
## Read an image
image_file = 'test_images/test6.jpg'
img = mpimg.imread(image_file)
out_img = image_pipeline(img)
plt.imshow(out_img)
plt.show()
'''

from moviepy.editor import VideoFileClip
import time

start_time = time.time()
output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(image_pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)
end_time = time.time()
print("Time taken to process the video = %d sec"% (end_time - start_time))

'''
print(all_lane_width)
from statistics import mean
print("Average lane width = ", mean(all_lane_width))
print("Max lane width = ", max(all_lane_width))
print("Min lane width = ", min(all_lane_width))
'''
###########################################################################
