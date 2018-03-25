#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#include <ApplicationServices/ApplicationServices.h>
#include <unistd.h>



using namespace cv;
using namespace std;

RNG rng(12345);


double mydist(cv::Point x, cv::Point y)
{
  return (x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y);
}



pair<cv::Point,double> circleFromPoints(cv::Point p1, cv::Point p2, cv::Point p3)
{
  double offset = pow(p2.x,2) +pow(p2.y,2);
  double bc =   ( pow(p1.x,2) + pow(p1.y,2) - offset )/2.0;
  double cd =   (offset - pow(p3.x, 2) - pow(p3.y, 2))/2.0;
  double det =  (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
  double TOL = 0.0000001;
  if (abs(det) < TOL) { cout<<"PointS TOO CLOSE"<<endl;return make_pair(cv::Point(0,0),0); }

  double idet = 1/det;
  double centerx =  (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
  double centery =  (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
  double radius = sqrt( pow(p2.x - centerx,2) + pow(p2.y-centery,2));

  return make_pair(cv::Point(centerx,centery),radius);
}






int main() {
  
  VideoCapture video(0);
  
  if(!video.isOpened()) {
    return 1;
  }
  
  Scalar color1 = Scalar(0,0,255);
  Scalar color2 = Scalar(0,255,0);
  Scalar color3 = Scalar(255,0,0);
  
  vector<Mat> channels;
  float totalHullSize=0;
  int h_mean, s_mean, v_mean;
  int low_h, low_s, low_v;
  int high_h, high_s, high_v;
  
  int lh1, lh2, hh1, hh2;
  char str[200];
  int spread = 20;

  Mat frame, fore, back, hsv, mesh1, mesh2, mask,cropROI;
  bool ok = video.read(frame);
  
  cv::Point roiTL = cv::Point(frame.cols/2-70, frame.rows/2-70);
  cv::Point roiBR = cv::Point(frame.cols/2+70, frame.rows/2+70);

  for(int i = 0; i<150; i++) {
    ok = video.read(frame);
    rectangle(frame, roiTL, roiBR, color1);
    cv::Rect myROI = cv::Rect(roiTL,roiBR);
    cropROI = frame(myROI);
    imshow("Prelim", frame);
    int c = waitKey ( 10 );
    if( c >= 0 ) {
      return -1;
    }

 }

  vector<vector<cv::Point> > contours;
  vector<Vec4i> hierarchy;
  vector<vector<int> > hullI(1);
  vector<vector<cv::Point> > hull(1);
  vector<pair<cv::Point,double> > palm_centers;

  
  cvtColor(cropROI, hsv, CV_BGR2HSV);
  split(cropROI, channels);
  h_mean = (int) mean(channels[0])[0];
  s_mean = (int) mean(channels[1])[0];
  v_mean = (int) mean(channels[2])[0];
  
  low_h = h_mean - spread;
  high_h = h_mean + spread;
  
  low_s = s_mean  - spread;
  high_s = s_mean + spread;
  low_v = v_mean - spread;
  high_v = v_mean + spread;
  
  if(low_s < 0) {
    low_s = 0;
  }
  if(high_s > 180) {
    high_s = 180;
  }
  if(low_v < 0) {
    low_v = 0;
  }
  if(high_v > 255) {
    high_v = 255;
  }


  /*
  if(low_h < 0) {
    low_h = 0;
  }
  if(high_h > 255) {
    high_h = 255;
    }*/

  
  //h wraps around 180
  if(low_h < 0) {
    lh1 = 180 + low_h;
    lh2 = 180;
    hh1 = 0;
    hh2 = h_mean + spread;
  } else {
    lh1 = low_h;
    if(high_h > 180) {
      lh2 = 180;
      hh1 = 0;
      hh2 = high_h - 180;
    } else {
      lh2 = high_h;
      hh1 = low_h;
      hh2 = high_h;
    }
  }
   

  Scalar WhiteSkinColor_min = Scalar(lh1 ,low_s , low_v );
  Scalar WhiteSkinColor_max = Scalar(lh2, high_s, high_v);
  Scalar WhiteSkinColor_min_high = Scalar(hh1, low_s,low_v );
  Scalar WhiteSkinColor_max_high = Scalar(hh2, high_s, high_v);

  //Scalar WhiteSkinColor_min_high = Scalar(low_h, low_s,low_v );
  //Scalar WhiteSkinColor_max_high = Scalar(high_h, high_s, high_v);
  
  Ptr<BackgroundSubtractor> bg = createBackgroundSubtractorMOG2();
  for(;;) {
    cv::Point palm_center;
    totalHullSize=0;
    ok = video.read(frame);

    cvtColor(frame, hsv, CV_BGR2HSV);
    inRange(frame, WhiteSkinColor_min, WhiteSkinColor_max, mesh1);
    inRange(frame, WhiteSkinColor_min_high, WhiteSkinColor_max_high, mesh2);
    mask = mesh1 | mesh2;

    erode(mask, mask, Mat());
    dilate(mask, mask, Mat());

    GaussianBlur(mask, mask, cv::Size(31,31), 0,0);
    //threshold(mask,mask, 128, 255, 0);

    bg->apply(mask, fore, 0.01);
    bg->getBackgroundImage(back);
    
    GaussianBlur(fore, fore, cv::Size(31,31), 0,0);
    //    erode(fore, fore, Mat());
    //    dilate(fore, fore, Mat());
    threshold(fore,fore, 128, 255, 0);
    
    findContours(fore, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    Mat drawing = Mat::zeros(fore.size(), CV_8UC3);
    
    for(int i=0; i<contours.size(); i++) {
      if(contourArea(contours[i]) >= 2000) {
  printf("%d\n", contourArea(contours[i]));
  drawContours(drawing, contours, i, color1, false);
  convexHull(Mat(contours[i]), hull[0], false);
  convexHull(Mat(contours[i]), hullI[0], false);
  drawContours(drawing, hull, -1, color2, false);
  
  RotatedRect rect = minAreaRect(Mat(contours[i]));

  vector<Vec4i> defects;

  if (hullI[0].size() > 0) {
    Point2f rect_Points[4];

    rect.points(rect_Points);
  
    for(int j=0; j < 4; j++) {
      line(drawing, rect_Points[j], rect_Points[(j+1)%4], color3, 1, 8);
    }

    cv::Point Palm;
  
    convexityDefects(contours[i], hullI[0], defects);
    totalHullSize += contourArea(hull[0]);
    //  sprintf(str, "%f - size of hull", contourArea(hull[0]));
    //putText(drawing, str, cv::Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
  
    if (defects.size() >= 3) {
      vector<cv::Point> p_Points;

      for (int k=0; k < defects.size(); k++) {
        int startidx=defects[k][0];
        cv::Point ptStart( contours[i][startidx] );
        int endidx=defects[k][1];
        cv::Point ptEnd( contours[i][endidx] );
        int faridx=defects[k][2];
        cv::Point ptFar( contours[i][faridx] );
        Palm += ptFar+ptStart+ptEnd;
        p_Points.push_back(ptFar);
        p_Points.push_back(ptStart);
        p_Points.push_back(ptEnd);
      }

      Palm.x/=defects.size()*3;
      Palm.y/=defects.size()*3;
      
      cv::Point cpt=  p_Points[0];
      
      vector<pair<double, int> > distvec;
      
      for (int k =0; k < p_Points.size(); k++) {
        distvec.push_back(make_pair(mydist(Palm, p_Points[k]), k));
      }
      
      sort(distvec.begin(), distvec.end());
      
      pair<cv::Point,double> soln_circle;
      
      for (int k=0;k+2<distvec.size();k++) {
        cv::Point p1=p_Points[distvec[k+0].second];
        cv::Point p2=p_Points[distvec[k+1].second];
        cv::Point p3=p_Points[distvec[k+2].second];
        soln_circle=circleFromPoints(p1,p2,p3);  //Final palm center,radius
        if (soln_circle.second!=0) {
          break;
        }
      }
      
      palm_centers.push_back(soln_circle);
      if (palm_centers.size()>10) {
        palm_centers.erase(palm_centers.begin());
      }
      
      
      double radius=0;
      for (int k=0;k<palm_centers.size();k++) {
        palm_center += palm_centers[k].first;
        radius += palm_centers[k].second;
      }
      
      palm_center.x /= palm_centers.size();
      palm_center.y /= palm_centers.size();
      
      radius /= palm_centers.size();
      
      circle(drawing,palm_center,5,Scalar(144,144,255),3);
      circle(drawing,palm_center,radius,Scalar(144,144,255),2);
      
      int no_of_fingers=0;
      for (int j=0;j<defects.size();j++) {
        int startidx=defects[j][0]; 
        cv::Point ptStart( contours[i][startidx] );
        int endidx=defects[j][1]; 
        cv::Point ptEnd( contours[i][endidx] );
        int faridx=defects[j][2];
        cv::Point ptFar( contours[i][faridx] );
        //X o--------------------------o Y                                                                   
        double Xdist=sqrt(mydist(palm_center,ptFar));
        double Ydist=sqrt(mydist(palm_center,ptStart));
        double length=sqrt(mydist(ptFar,ptStart));
        
        double retLength=sqrt(mydist(ptEnd,ptFar));
        if(length<=3*radius&&Ydist>=0.4*radius&&length>=10&&retLength>=10&&max(length,retLength)/min(length,retLength)>=0.8) {
          if(min(Xdist,Ydist)/max(Xdist,Ydist)<=0.8) {
            if((Xdist>=0.1*radius&&Xdist<=1.3*radius&&Xdist<Ydist)||(Ydist>=0.1*radius&&Ydist<=1.3*radius&Xdist>Ydist)) {
              line( frame, ptEnd, ptFar, Scalar(0,255,0), 1 ),no_of_fingers++;
            }
          }
        }
      }
      
      if(totalHullSize>100.0) {
        CGEventRef move1 = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved,CGPointMake(palm_center.x, palm_center.y),kCGMouseButtonLeft );
        CGEventPost(kCGHIDEventTap, move1);
        CFRelease(move1);
      }
    }
  }
      }
    }
    sprintf(str, "%f - size of hull", totalHullSize);
    putText(drawing, str, cv::Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
    if(totalHullSize <= 12000 && totalHullSize > 100) {
      
      putText(drawing, "clicked", cv::Point(100,400), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
      CGEventRef click1_down = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseDown,CGPointMake(palm_center.x, palm_center.y),kCGMouseButtonLeft);
      CGEventPost(kCGHIDEventTap, click1_down);
      CFRelease(click1_down);
    } else {
      putText(drawing, "let go", cv::Point(100,200), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
      CGEventRef click1_up = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseUp,CGPointMake(palm_center.x, palm_center.y),kCGMouseButtonLeft);
      CGEventPost(kCGHIDEventTap, click1_up);
      CFRelease(click1_up);
    }
    imshow("Foreground", drawing );
    char c = (char) waitKey ( 30 );
    if( c ==  (char) 27 ) {
      return -1;
    }
  }
  return 0;
}

