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

        # Unwarped Image                  
        unwarped_img = unwarp(undist_img, src, dst)

        # Sobel Thresholding
        abs_sobel_img = abs_sobel_thresh(unwarped_img, 'x', thresh=(30,100))
        mag_sobel_img = mag_thresh(unwarped_img, thresh=(30, 100))
        dir_sobel_img = dir_threshold(unwarped_img, sobel_kernel=15, thresh=(0.7, 1.3))

        # HLS Thresholding
        hls_lselect_img = hls_lselect(unwarped_img, thresh=(200,255))
        hls_sselect_img = hls_sselect(unwarped_img, thresh=(90,255))
        
        # LAB Thresholding
        lab_lselect_img = lab_lselect(unwarped_img, thresh=(200,255))
        lab_aselect_img = lab_aselect(unwarped_img, thresh=(200,255))
        lab_bselect_img = lab_bselect(unwarped_img, thresh=(200,255))
        
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
            
        axs[(idx * thresh_types) + 0].imshow(unwarped_img)
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

test_thresholding()