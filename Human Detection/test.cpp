/*
-- Georgia Tech 2016 Spring
--
-- This is a sample code to show how to use the libfreenet2 with OpenCV
--
-- The code will streams RGB, IR and Depth images from an Kinect sensor.
-- To use multiple Kinect sensor, simply initial other "listener" and "frames"
-- This code refered from sample code provided from libfreenet2: Protonect.cpp
-- https://github.com/OpenKinect/libfreenect2
-- and another discussion from: http://answers.opencv.org/question/76468/opencvkinect-onekinect-for-windows-v2linuxlibfreenect2/
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>
*/

//! [headers]
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/objdetect.hpp"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
//! [headers]

using namespace std;
using namespace cv;

bool protonect_shutdown = false; // Whether the running application should shut down.

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

int main()
{
    std::cout << "Streaming from Kinect One sensor!" << std::endl;

    //! [context]
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;
    //! [context]

    //! [discovery]
    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    string serial = freenect2.getDefaultDeviceSerialNumber();

    std::cout << "SERIAL: " << serial << std::endl;

    if(pipeline)
    {
        //! [open]
        dev = freenect2.openDevice(serial, pipeline);
        //! [open]
    } else {
        dev = freenect2.openDevice(serial);
    }

    if(dev == 0)
    {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }

    signal(SIGINT, sigint_handler);
    protonect_shutdown = false;

    //! [listeners]
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |
                                                  libfreenect2::Frame::Depth |
                                                  libfreenect2::Frame::Ir);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    //! [listeners]

    //! [start]
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
    //! [start]

    //! [registration setup]
    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1280, 720 + 2, 4); // check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger
    //! [registration setup]

    Mat rgbmat, depthmat, depthmatUndistorted, irmat, rgbd, rgbd2;
    
    Size win_size(48,96);
    Size block_size(16,16);
    Size block_stride(8,8);
    Size cell_size(8,8);	
    Size win_stride(8,8);
    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, 		cell_size, 9);
    Mat detector = gpu_hog->getDefaultPeopleDetector();
    gpu_hog->setSVMDetector(detector);
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
 
    cuda::GpuMat gpu_img;
    //! [loop start]
    while(!protonect_shutdown)
    {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        //! [loop start]

        Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbmat);
        Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irmat);
        Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);
        vector<Rect> found, found_filtered;
	Mat img;
	Mat img_to_show;
	int64 hog_work_begin = getTickCount();
        resize(rgbmat, img, Size(640, 480));
	img_to_show = img;
	gpu_img.upload(img);
        gpu_hog->setNumLevels(13);
        gpu_hog->setHitThreshold(.9);
        gpu_hog->setWinStride(win_stride);
        gpu_hog->setScaleFactor(1.1);
        gpu_hog->setGroupThreshold(2);
        gpu_hog->detectMultiScale(gpu_img, found);
	int64 delta = getTickCount() - hog_work_begin;
    	double freq = getTickFrequency();
   	int64 hog_work_fps = freq / delta;
        //imshow("rgb", rgbmat);
        //cv::imshow("ir", irmat / 4096.0f);
        //cv::imshow("depth", depthmat / 4096.0f);

        //! [registration]
        //registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
        //! [registration]

        //cv::Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copyTo(depthmatUndistorted);
        //Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(rgbd);	
	//vector<Rect> found, found_filtered;
	//Mat dst;
	//cvtColor(rgbmat, dst, CV_BGRA2BGR);
        //hog.detectMultiScale(dst, found, 0, Size(8,8), Size(32,32), 1.05, 2);
        //cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data).copyTo(rgbd2);


        //cv::imshow("undistorted", depthmatUndistorted / 4096.0f);
        //imshow("registered", rgbd);
        //cv::imshow("depth2RGB", rgbd2 / 4096.0f);
	size_t i, j;
        for (i=0; i<found.size(); i++)
        {
            Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }
        for (i=0; i<found_filtered.size(); i++)
        {
	    Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
	    r.width = cvRound(r.width*0.8);
	    r.y += cvRound(r.height*0.06);
	    r.height = cvRound(r.height*0.9);
	    rectangle(img_to_show, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	}
	//for (size_t i = 0; i < found.size(); i++)
          // {
              // Rect r = found[i];
            //   rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
           //}
	std::cout << hog_work_fps << std::endl;
        imshow("video capture", img_to_show);
	if(found.size() > 0)
	{
	   imwrite("/images/person.jpg", rgbmat);
	}
        

        int key = waitKey(1);
        protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape

    //! [loop end]
        listener.release(frames);
    }
    //! [loop end]

    //! [stop]
    dev->stop();
    dev->close();
    //! [stop]

    delete registration;

    std::cout << "Streaming Ends!" << std::endl;
    return 0;
}
