//
//  main.cpp
//  test_cityflow
//
//  Created by Ishay Tubi on 02/06/2020.
//

#include <string>
#include <iostream>
#include <engine/engine.h>

#include <iostream>
#include <unistd.h>

#include <opencv2/opencv.hpp>
using namespace cv;

using namespace std;
const int MAT_SIZE = 1024;
typedef std::vector<cv::Point2f> cv_points_vector;



std::string get_working_path()
{
    const int MAXPATHLEN = 500;
    char temp[MAXPATHLEN];
    return ( getcwd(temp, sizeof(temp)) ? std::string( temp ) : std::string("") );
}

typedef cv::Matx<float, 1, 2> Vec2D;

class Normlizer2D{
public:
    cv::Point2f minxy;
    cv::Point2f maxxy;
    cv::Mat image = cv::Mat::zeros(cv::Size(MAT_SIZE, MAT_SIZE), CV_32FC1);

    Normlizer2D(cv::Point2f minxy_, cv::Point2f maxxy_): minxy(minxy_), maxxy(maxxy_){};
    points_vector operator ()(const points_vector& points){
        return  points;
    }
  
    auto forward(cv_points_vector &v){
        Mat m  = cv::Mat(cv::Size(int(v.size()), 1), CV_32FC2, v.data());
        Point2f range3 = maxxy - minxy;
        Point2f range_1 = Point2f((MAT_SIZE-1.0)/range3.x, (MAT_SIZE-1.0)/range3.y);

        m.forEach<Point2f>(
           [range_1, this](Point2f &pixel, const int * position) -> void
           {
               pixel -= minxy;
               pixel.x *= range_1.x;
               pixel.y *= range_1.y;
           }
        );
        cout<<m.at<float>(0,0)<<" "<<to_string(m.at<float>(1,0))<<endl;
        return m;
    }
    
    static int test1(){
        cv_points_vector edges = {{-100.0, -200.0}, {+100.0, +200.0}};
        cv_points_vector edges2 = {{-100.0, -200.0}, {+100.0, +200.0}};
        Normlizer2D normlizer2D(edges[0], edges[1]);
        Mat e = normlizer2D.forward(edges2);
        for(auto &edge:edges2){ cout<<"X:"<<to_string(edge.x)<<" y:"<<to_string(edge.y)<<" ";}
        
        cv_points_vector v{{-100.0, -200}, {0, 100}, {-100, +100}};
        //Mat m = normlizer2D.as_mat(v);
        return 0;
    }
};


int main(int argc, const char * argv[]) {
    Normlizer2D::test1();
    
    // insert code here...
    std::cout << "Hello, World!\n";
    std::string cwd = get_working_path();
    std::string config_path = "/Users/ishay/projects/CityFlow/data/esquare3/config_engine.json";
    if (argc > 1) {
        config_path = argv[1];  // Take pathh from command line arg
    }
    int threadNum = 1;
    int total_vc_count = 0;
    int total_stopping = 0;
    int step = 0 ;

    CityFlow::Engine eng(config_path, threadNum);
    cv::Mat mat(cv::Size(MAT_SIZE, MAT_SIZE), CV_8U);

    for (auto &intersection : eng.roadnet.getIntersections()) {
        if (!intersection.isVirtualIntersection()){
            cout<<intersection.getId()<<" "<<to_string(intersection.getTrafficLight().getPhases().size());
            for (auto &phase:intersection.getTrafficLight().getPhases()){
                cout<<endl<<"\t\t"<<phase;
            }
            cout<<endl;
        }
    }
    
    double avg = 0.0;

    while (step<10000) {
        int vc = eng.getVehicleCount();
        //eng.get_as_image(mat.data);
        cout<<"Sum as mat:"<<to_string(cv::sum(mat)[0])<<endl;
        points_vector_d p = eng.getVehiclesLocation();

        total_vc_count += vc;
        //eng.get_lane_waiting_vehicle_count()
        //eng.get_current_time()
        //eng.get_vehicle_speed()
        auto vehicle_speed = eng.getVehicleSpeed();
        int stopping = 0;
        avg = 0.0;
        
        for (auto kv : vehicle_speed){
            stopping += kv.second == 0.0;
            avg += kv.second;
        }
        
        if (vehicle_speed.size() > 0 ) {
            avg /= vehicle_speed.size();
        }
        //stopping = np.sum([item==0.0 for key, item in vehicle_speed.items()])
        total_stopping += stopping;
        if (step  % 100 == 1){
            eng.bumpPhase();
            float ratio = float(total_stopping) / float(total_vc_count) ;

            cout<<step<<"   "<<ratio<<"      "<<to_string(avg)<<"\r"<<std::flush;
            auto c = eng.getLaneVehicleCount();
            //print(step, total_vc_count, total_stopping, total_stopping/total_vc_count)
            /*steps_list.append(dict(
                                   total_stopping=total_stopping,
                                   total_vc_count=total_vc_count,
                                   step=step
                                   ))
             */
            cout<<to_string(c.size());
        }
        
        int a = eng.getVehicleCount();
        //eng.loadConfig(<#const std::string &configFile#>)
        //eng.setTrafficLightPhase(<#const std::string &id#>, <#int phaseIndex#>)
        eng.nextStep();
        step++;
    }
    cout<<endl;

    float ratio = float(total_stopping) / float(total_vc_count) ;
    
    cout<<"Simulation ended with success after steps:"<<step<<" with ratio :"<< ratio <<" avg speed "<< to_string(avg) <<endl;
    return 0;
}
