from image_thresholds import *

# This function applies different thresholding
# techniques (Sobel edge, HLS/LAB color thresholding etc
# and displays to help determine which thresholds to use.
def test_thresholding():
    # Load the camera calibration coefficients
    camera_cal_values = pickle.load(open("calibration.p", "rb"))
    mtx = camera_cal_values["mtx"]
    dist = camera_cal_values["dist"]

    # Load all test images
    images = glob.glob('test_images/*.jpg')
    #images = ['test_images/straight_lines2.jpg']
    num_images = len(images) # Number of images
    thresh_types = 10        # Number of thresholding types

    # Setup the plot
    f, (axs) = plt.subplots(num_images, thresh_types, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    axs = axs.ravel()
        
    for idx, image_file in enumerate(images):    
                  
        # Read a test image
        img = mpimg.imread(image_file)

        # Undistort and save it.
        undist_img = undistort(img, mtx, dist)

        # Hardcoded src and dst for perspective transform
        # TODO: Find src and dst programatically 
        h,w = undist_img.shape[:2]
        src = np.float32([(575,464),
                          (707,464), 
                          (258,682), 
                          (1049,682)])
                          
        dst = np.float32([(450,0),
                          (w-450,0),
                          (450,h),
                          (w-450,h)])

        # warped Image                  
        warped_img = warp(undist_img, src, dst)
        #warped_img = undist_img

        # Sobel Thresholding
        abs_sobel_img = abs_sobel_thresh(warped_img, 'x', thresh=(30,200))
        mag_sobel_img = mag_thresh(warped_img, thresh=(30, 100))
        dir_sobel_img = dir_threshold(warped_img, sobel_kernel=15, thresh=(np.pi/3, np.pi/2))

        # HLS Thresholding
        hls_lselect_img = hls_lselect(warped_img, thresh=(200,255))
        hls_sselect_img = hls_sselect(warped_img, thresh=(180,255))
        
        # LAB Thresholding
        lab_lselect_img = lab_lselect(warped_img, thresh=(200,255))
        lab_aselect_img = lab_aselect(warped_img, thresh=(200,255))
        lab_bselect_img = lab_bselect(warped_img, thresh=(200,255))
        
        # Selected HLS-L Channel to pick up white lines and 
        # LAB-B channel to detect yellow lines.
        threshold_img = np.zeros_like(hls_lselect_img)
        threshold_img[(hls_lselect_img == 1) | (lab_bselect_img == 1)] = 1        
        #print(np.count_nonzero(lab_bselect_img))
        #print(lab_bselect_img)
        
        if idx == 0:
            axs[0].set_title('Unwarped Image', fontsize=10)
            axs[1].set_title('Abs Sobel', fontsize=10)
            axs[2].set_title('Mag Sobel', fontsize=10)
            axs[3].set_title('Dir Sobel', fontsize=10)
            axs[4].set_title('HLS L', fontsize=10)
            axs[5].set_title('HLS S', fontsize=10)
            axs[6].set_title('LAB L', fontsize=10)
            axs[7].set_title('LAB A', fontsize=10)
            axs[8].set_title('LAB B', fontsize=10)
            axs[9].set_title('HLS L | LAB B', fontsize=10)
            
        axs[(idx * thresh_types) + 0].imshow(warped_img)
        axs[(idx * thresh_types) + 1].imshow(abs_sobel_img, cmap='gray')
        axs[(idx * thresh_types) + 2].imshow(mag_sobel_img, cmap='gray')
        axs[(idx * thresh_types) + 3].imshow(dir_sobel_img, cmap='gray')
        axs[(idx * thresh_types) + 4].imshow(hls_lselect_img, cmap='gray')
        axs[(idx * thresh_types) + 5].imshow(hls_sselect_img, cmap='gray')
        axs[(idx * thresh_types) + 6].imshow(lab_lselect_img, cmap='gray')
        axs[(idx * thresh_types) + 7].imshow(lab_aselect_img, cmap='gray')
        axs[(idx * thresh_types) + 8].imshow(lab_bselect_img, cmap='gray')
        axs[(idx * thresh_types) + 9].imshow(threshold_img, cmap='gray')
        # End of For Loop

    plt.tight_layout()    
    plt.show()

#test_thresholding()
#exit(0)

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

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

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

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
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return (out_img, left_fit, right_fit)

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
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
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    print('Radius of curvature = %d' % (measure_curvature_real(ploty, left_fit, right_fit)))
    
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
    '''
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    '''
    ## End visualization steps ##
    
    return result, left_fitx, right_fitx, ploty

def measure_curvature_real(ploty, left_fitx, right_fitx):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
       
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    #left_curverad = 0  ## Implement the calculation of the left line here
    #right_curverad = 0  ## Implement the calculation of the right line here
    left_curverad = ((1 + (2*left_fitx[0]*y_eval*ym_per_pix + left_fitx[1])**2)**1.5) / np.absolute(2*left_fitx[0])
    right_curverad = ((1 + (2*right_fitx[0]*y_eval*ym_per_pix + right_fitx[1])**2)**1.5) / np.absolute(2*right_fitx[0])
    
    print(left_curverad, right_curverad)
    return (left_curverad + right_curverad) / 2

def draw_lane_lines(undist, warped, left_fitx, right_fitx, ploty, src, dst):
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
    plt.imshow(result)
    plt.show()


###########################################################################



def image_pipeline(img):
    ## Create Binary thresholded image
    hls_lselect_img = hls_lselect(undist_img, thresh=(200,255))                          
    lab_bselect_img = lab_bselect(undist_img, thresh=(200,255))
    # Selected HLS-L Channel to pick up white lines and 
    # LAB-B channel to detect yellow lines.
    threshold_img = np.zeros_like(hls_lselect_img)
    threshold_img[(hls_lselect_img == 1) | (lab_bselect_img == 1)] = 1        


    ## Perspective transform

    # Hardcoded src and dst for perspective transform
    # TODO: Is there a way to find src and dst programatically ?
    h,w = undist_img.shape[:2]
    src = np.float32([(575,464),
                      (707,464), 
                      (258,682), 
                      (1049,682)])
                      
    dst = np.float32([(450,0),
                      (w-450,0),
                      (450,h),
                      (w-450,h)])
    binary_warped = warp(threshold_img, src, dst)

                      
    ## Fit the polynomial using sliding window 
    (out_img, left_fit, right_fit) = fit_polynomial(binary_warped)
    
    # Search around polynimial found in the last steps
    out_img, left_fitx, right_fitx, ploty = search_around_poly(binary_warped, left_fit, right_fit)
    
    # Draw lane lines
    draw_lane_lines(undist_img, binary_warped, left_fitx, right_fitx, ploty, src, dst)
    
    return out_img

## Read an image
image_file = 'test_images/test6.jpg'
img = mpimg.imread(image_file)

## Undistort the image 

# Load the camera calibration coefficients
camera_cal_values = pickle.load(open("calibration.p", "rb"))
mtx = camera_cal_values["mtx"]
dist = camera_cal_values["dist"]

undist_img = undistort(img, mtx, dist)

out_img = image_pipeline(undist_img)

plt.imshow(out_img)
plt.show()

###########################################################################
