#!/usr/bin/env python

# Built-in Modules
import time
import argparse
import logging
# Standard Modules
import cv2
import numpy
import math
# Custom Modules
import scripts
import SpeedySuperPixels

from leapcontrol import LeapListener
import Leap

logger = logging.getLogger('main')


class SkinDetector(object):

    def __init__(self, args):
        assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace'
        self.args = args
        self.mask = None
        logger.debug('SkinDetector initialised')

    @staticmethod
    def assert_image(img, grey=False):
        logger.debug('Applying assertions...')
        depth = 3
        if grey:
            depth = 2
        assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
        assert len(img.shape) == depth, 'skin detection can only work on color images'
        assert img.size > 100, 'seriously... you thought this would work?'

    def get_mask_hsv(self, img):
        logger.debug('Applying hsv threshold')
        self.assert_image(img)
        lower_thresh = numpy.array([0, 50, 0], dtype=numpy.uint8)
        upper_thresh = numpy.array([120, 150, 255], dtype=numpy.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        msk_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        if self.args.debug:
            scripts.display('input', img)
            scripts.display('mask_hsv', msk_hsv)
        self.add_mask(msk_hsv)

    def get_mask_rgb(self, img):
        logger.debug('Applying rgb thresholds')
        #lower_thresh = numpy.array([45, 52, 108], dtype=numpy.uint8)
        lower_thresh = numpy.array([40, 40, 90], dtype=numpy.uint8)
        upper_thresh = numpy.array([255, 255, 255], dtype=numpy.uint8)
        mask_a = cv2.inRange(img, lower_thresh, upper_thresh)
        mask_b = 255*((img[:, :, 2]-img[:, :, 1])/20)
        logger.debug('mask_b unique: {0}'.format(numpy.unique(mask_b)))
        mask_c = 255*((numpy.max(img, axis=2)-numpy.min(img, axis=2))/20)
        logger.debug('mask_d unique: {0}'.format(numpy.unique(mask_c)))
        msk_rgb = cv2.bitwise_and(mask_a, mask_b)
        msk_rgb = cv2.bitwise_and(mask_c, msk_rgb)
        if self.args.debug:
            scripts.display('input', img)
            scripts.display('mask_rgb', msk_rgb)
        self.add_mask(msk_rgb)

    def get_mask_ycrcb(self, img):
        self.assert_image(img)
        lower_thresh = numpy.array([90, 100, 130], dtype=numpy.uint8)
        upper_thresh = numpy.array([230, 120, 180], dtype=numpy.uint8)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)
        if self.args.debug:
            scripts.display('input', img)
            scripts.display('mask_ycrcb', msk_ycrcb)
        self.add_mask(msk_ycrcb)

    def grab_cut_mask(self, img_col, mask):
        kernel = numpy.ones((50, 50), numpy.float32)/(50*50)
        dst = cv2.filter2D(mask, -1, kernel)
        dst[dst != 0] = 255
        free = numpy.array(cv2.bitwise_not(dst), dtype=numpy.uint8)
        if self.args.debug:
            cv2.imshow('not skin', free)
            cv2.imshow('grabcut input', mask)
        grab_mask = numpy.zeros(mask.shape, dtype=numpy.uint8)
        grab_mask[:, :] = 2
        grab_mask[mask == 255] = 1
        grab_mask[free == 255] = 0
        if numpy.unique(grab_mask).tolist() == [0, 1]:
            logger.debug('conducting grabcut')
            bgdModel = numpy.zeros((1, 65), numpy.float64)
            fgdModel = numpy.zeros((1, 65), numpy.float64)
            if img_col.size != 0:
                mask, bgdModel, fgdModel = cv2.grabCut(img_col, grab_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                mask = numpy.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            else:
                logger.warning('img_col is empty')
        return mask

    @staticmethod
    def closing(msk):
        assert isinstance(msk, numpy.ndarray), 'msk must be a numpy array'
        assert msk.ndim == 2, 'msk must be a greyscale image'
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel, iterations=2)
        return msk

    def process(self, img):
        dt = time.time()
        self.assert_image(img)
        self.n_mask = 0
        self.mask = numpy.zeros(img.shape[:2], dtype=numpy.uint8)
        self.get_mask_rgb(img) # modify color space
        self.threshold(self.args.thresh)
        if self.args.debug:
            cv2.imshow('skin_mask', self.mask)
            cv2.imshow('input_img', img)
        dt = round(time.time()-dt, 2)
        hz = round(1/dt, 2)
        self.mask = self.closing(self.mask)
        self.mask = self.grab_cut_mask(img, self.mask)
        return self.mask

    def add_mask(self, img):
        logger.debug('normalising mask')
        self.assert_image(img, grey=True)
        img[img < 128] = 0
        img[img >= 128] = 1
        logger.debug('normalisation complete')
        logger.debug('adding mask to total mask')
        self.mask += img
        self.n_mask += 1
        logger.debug('add mask complete')

    def threshold(self, threshold):
        assert isinstance(threshold, float), 'threshold must be a float (current type - {0})'.format(type(threshold))
        assert 0 <= threshold <= 1, 'threshold must be between 0 & 1 (current value - {0})'.format(threshold)
        assert self.n_mask > 0, 'Number of masks must be greater than 0 [n_mask ({0}) = {1}]'.format(type(self.n_mask), self.n_mask)
        logger.debug('Threshold Value - {0}%'.format(int(100*threshold)))
        logger.debug('Number of Masks - {0}'.format(self.n_mask))
        self.mask /= self.n_mask
        self.mask[self.mask < threshold] = 0
        self.mask[self.mask >= threshold] = 255
        logger.debug('{0}% of the image is skin'.format(int((100.0/255.0)*numpy.sum(self.mask)/(self.mask.size))))
        return self.mask


def process(image, save=False, display=False, args=None, segment=False):
    assert isinstance(image, numpy.ndarray)
    if not args:
        args = scripts.gen_args()
    else:
        assert isinstance(args, argparse.Namespace), 'args must be an argparse.Namespace'
    args.save = save
    args.display = display
    detector = SkinDetector(args)
    if segment:
        slic = SpeedySuperPixels.SuperContour()
        skin = numpy.zeros(image.shape, dtype=image.dtype)
        for roi, contour in slic.process(image):
            pxl = cv2.bitwise_and(image, image, mask=contour)
            msk = detector.process(pxl)
            ret = msk.sum()/contour.sum()
            if ret > 0.8:
                skin = numpy.min(255, cv2.add(skin, contour))
        return skin
    else:
        return detector.process(image)


def getLongestSide(pt_arr, lenThresh=100000.0):
    pt_num = len(pt_arr)
    # print "pt_num"
    # print pt_num
    longestLen = 0
    retPt0 = []
    retPt1 = []
    for i in range(pt_num-1):
        p1 = pt_arr[i]
        p2 = pt_arr[i+1]

        dist = math.sqrt((p1[0][0]-p2[0][0])*(p1[0][0]-p2[0][0])+(p1[0][1]-p2[0][1])*(p1[0][1]-p2[0][1]))
        if dist > longestLen and dist < lenThresh:
            longestLen = dist
            retPt0 = p1
            retPt1 = p2
    # print "in func"
    # print retPt0[0], retPt1[0]
    return retPt0, retPt1, longestLen


def find_rot_angles(p00, p01, p10, p11):
    angle_in_radians = None
    vec0 = [p01[0] - p00[0],p01[1] - p00[1]]
    vec1 = [p11[0] - p10[0],p11[1] - p10[1]]

    if vec0[0] != 0:
        slope0 = float(vec0[1])/vec0[0]
        # print "slope0"
        # print slope0
        ang0_in_radians = math.atan(slope0)
    else:
        ang0_in_radians = math.pi/2

    if vec1[0] != 0:
        slope1 = float(vec1[1])/vec1[0]
        # print "slope1"
        # print slope1
        ang1_in_radians = math.atan(slope1)
    else:
        ang1_in_radians = math.pi/2

    return (ang1_in_radians-ang0_in_radians) * 180.0/math.pi


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)

    # reduce frame size to speed it up
    w = 640
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, w) 
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, w * 3/4) 
    camera.set(cv2.cv.CV_CAP_PROP_FPS, 10)


    # Create a sample listener and controller
    listener = LeapListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    area_cutoff = 0
    min_area = 100.0
    max_area = 3000.0
    match_thresh = 0.2
    rotAngleThresh = 10
    errThresh = 0.6
    matchEpsilon = 0.005
    shape_saved = False
    contours_saved = []
    centers_saved = []
    SHAPE_NAMES = ['Gun', 'Bird']
    savedPt0 = []
    savedPt1 = []
    lastPt0 = []
    lastPt1 = []
    savedShape = []
    translatn_vec = (0,0)
    shoudBind = True;

    while True:
        ret, frame = camera.read()
        # flip and display
        frame = cv2.flip(frame,1)
        img_msk = process(frame,segment=False)
        
        # scripts.display('img_msk', img_msk)
        cv2.imshow('img_msk', img_msk)
        c_frm = img_msk.copy()
        # cv2.imshow("contour", c_frm)
        
        # find_contour
        contours, hierarchy = cv2.findContours(c_frm,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        maxlength = 0
        largest_contour_index = -1
        
        # calculate areas and display at mass center        
        for i in range(len(contours)):
            c = contours[i]
            length = cv2.arcLength(c, False)
            if length > maxlength:
                maxlength = length
                largest_contour_index=i                  # bounding_rect=cv2.boundingRect(contours[i]);
        if largest_contour_index >= 0 and len(contours): 
            maxCMoment = cv2.moments(contours[largest_contour_index])
            cv2.drawContours(c_frm, contours, largest_contour_index, (255,0,0), 1)
            cv2.imshow("contour", c_frm)
        else:
            continue
        
        k = cv2.waitKey(5)
        if k == 27: # exit
            break

        elif k == 32: # ASCII code for space
            listener.clearDrawIndicator()
            centers_saved.append((int(maxCMoment['m10']/maxCMoment['m00']),int(maxCMoment['m01']/maxCMoment['m00'])))
            
            epsilon = matchEpsilon*cv2.arcLength(contours[largest_contour_index],True)
            approx = cv2.approxPolyDP(contours[largest_contour_index],epsilon,True)
            cv2.drawContours(c_frm, [approx], 0, (255,255,0), 2)
            # cv2.imshow("approxPolyDP", c_frm)
            contours_saved.append(approx)

            # savedPt0, savedPt1, longestLen = getLongestSide(approx)
            # lastPt0 = savedPt0
            # lastPt1 = savedPt1
            # print savedPt0, savedPt1
            # cv2.line(c_frm,(savedPt0[0][0],savedPt0[0][1]),(savedPt1[0][0],savedPt1[0][1]),(255,0,0),5)
            # cv2.imshow("saved_longest_side", c_frm)
            # savedShape = c_frm
            shape_saved = True
            print"shape %d saved" % len(centers_saved)
            continue

        
        if shape_saved == True:
            epsilon = matchEpsilon*cv2.arcLength(contours[largest_contour_index],True)
            approx = cv2.approxPolyDP(contours[largest_contour_index],epsilon,True)
            retVals = [cv2.matchShapes(c, approx, cv2.cv.CV_CONTOURS_MATCH_I2, 0.0) for c in contours_saved]
            # print retVals
            
            if (min(retVals) < errThresh):
                matchIndex = retVals.index(min(retVals))
                print "match shape: %s" % (SHAPE_NAMES[matchIndex]) 
                matchCenter = centers_saved[matchIndex]
                center = (int(maxCMoment['m10']/maxCMoment['m00']),int(maxCMoment['m01']/maxCMoment['m00']))
                translatn_vec = (center[0]-matchCenter[0],center[1]-matchCenter[1])
                shoudBind = True
            
                # rotation
                # redoCount = 0
                # longestLen = 100000

                # while(redoCount < 5):
                #     Pt0, Pt1, longestLen = getLongestSide(approx, longestLen)
                #     changeAngle = find_rot_angles(lastPt0[0], lastPt1[0], Pt0[0], Pt1[0])
                #     # print "changeAngle"
                #     # print changeAngle
                #     if (abs(changeAngle) > rotAngleThresh):
                #         redoCount += 1
                #     else:
                #         break

                # # print "redoCount:"
                # # print redoCount
                # rotAngle = find_rot_angles(savedPt0[0], savedPt1[0], Pt0[0], Pt1[0])
                # # print "rotAngle"
                # # print rotAngle
                
                # cv2.line(c_frm,(Pt0[0][0],Pt0[0][1]),(Pt1[0][0],Pt1[0][1]),(255,0,0),5)
                # cv2.imshow("longest_side", c_frm)
            else:
                shoudBind = False 
        
        if shoudBind:
            img_msk = listener.update_frame(img_msk, translatn_vec[0], translatn_vec[1])
        white = img_msk.copy()
        white = 255
        img_msk = white - img_msk
        cv2.imshow("img", img_msk);

    cv2.destroyAllWindows()
    camera.release()