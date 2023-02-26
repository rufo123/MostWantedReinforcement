import cv2
from numpy import ndarray

from GPS import GPS
from strategy.gps_image_recognition.a_gps_ircgn_strategy import AGpsImageRecognitionStrategy


class GpsImageRecognitionStrategyGPU(AGpsImageRecognitionStrategy):
    import time

    def __init__(self):
        self.a_car_offset = None
        self.a_width: int = 640
        self.a_height: int = 481
        self.a_channels: int = 3


        # Create GpuMat objects using pinned memory
        self.a_screen_gpu = cv2.cuda.GpuMat()
        self.a_screen_gpu.create(self.a_height, self.a_width, cv2.CV_8UC3)

        # Create GpuMat objects using pinned memory
        self.a_gps_gpu: cv2.cuda.GpuMat = cv2.cuda.GpuMat()
        self.a_gps_gpu.create(self.a_height, self.a_width, cv2.CV_8UC3)

        self.a_gps_greyscale_helper: cv2.cuda.GpuMat = cv2.cuda.GpuMat()
        self.a_gps_greyscale_helper.create(self.a_height, self.a_width, cv2.CV_8UC3)

        self.a_gps_blur_helper: cv2.cuda.GpuMat = cv2.cuda.GpuMat()
        self.a_gps_blur_helper.create(self.a_height, self.a_width, cv2.CV_8UC3)

        self.a_gps_greyscale_gpu: cv2.cuda.GpuMat = cv2.cuda.GpuMat()
        self.a_gps_greyscale_gpu.create(self.a_height, self.a_width, cv2.CV_8UC3)

    def calc_car_offset(self, par_gps: GPS, par_image: ndarray) -> tuple[float, list, int]:
        time = self.time.time()
        # Convert to grayscale and equalize histogram
        self.a_gps_greyscale_gpu = self.make_grayscale(par_image)
        # tmp_gps_final = cv2.equalizeHist(tmp_grayscale)

        # tmp_equalized_gpu = cv2.cuda.equalizeHist(tmp_grayscale_gpu)

        # tmp_gps_final = tmp_equalized_gpu.download()

        # Get GPS mask
        tmp_gps, tmp_gps_center = par_gps.get_gps_mask(self.a_gps_greyscale_gpu.download())

        # Apply GPS mask to screenshot

        # tmp_gps = cv2.bitwise_and(self.a_screenshot, self.a_screenshot, mask=tmp_gps)

        self.a_screen_gpu.upload(par_image)

        self.a_gps_gpu.upload(tmp_gps)

        self.a_gps_gpu = cv2.cuda.bitwise_and(self.a_screen_gpu, self.a_gps_gpu)

        #print(str(self.time.time() - time))

        # cv2.imshow('Main DKO', self.a_gps_gpu.download())

        # Convert to HSV and create grayscale mask
        tmp_gps_hsv = cv2.cuda.cvtColor(self.a_gps_gpu, cv2.COLOR_BGR2HSV)

        tmp_lower_bound = (0, 0, 174)
        tmp_upper_bound = (197, 10, 255)

        tmp_gps_greyscale = cv2.cuda.inRange(tmp_gps_hsv, tmp_lower_bound, tmp_upper_bound)

        # Create GPS contour
        tmp_contour = par_gps.make_gps_contour(tmp_gps_greyscale, par_image, tmp_gps_center)

        # car position
        tmp_car_pos = par_gps.get_car_point(par_image, tmp_gps_center)

        # Get car offset and lap progress
        tmp_car_offset_dist: float = par_gps.polygon_contour_test(tmp_contour, tmp_car_pos)

        tmp_car_directional_offset: int = par_gps.check_direction_point_to_contour(tmp_contour, tmp_car_pos)

        self.a_car_offset = tmp_car_offset_dist

        return tmp_car_offset_dist, tmp_contour, tmp_car_directional_offset

    def make_grayscale(self, par_image: ndarray) -> cv2.cuda.GpuMat:
        # grayscale = cv2.cvtColor(par_image, cv2.COLOR_BGR2GRAY)
        # return cv2.GaussianBlur(grayscale, (5, 5), 0)

        # Upload input image to GpuMat helper object
        self.a_gps_greyscale_helper.upload(par_image)

        # Convert image to grayscale
        self.a_gps_greyscale_helper = cv2.cuda.cvtColor(self.a_gps_greyscale_helper, cv2.COLOR_BGR2GRAY)

        # Create Gaussian filter and apply to the grayscale image
        self.a_gps_blur_helper = cv2.cuda.createGaussianFilter(self.a_gps_greyscale_helper.type(),
                                                               self.a_gps_greyscale_helper.type(), (5, 5), 0).apply(
            self.a_gps_greyscale_helper)

        # Download the blurred image from GPU and return
        return self.a_gps_blur_helper
