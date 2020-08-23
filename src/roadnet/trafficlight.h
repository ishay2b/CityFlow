#ifndef CITYFLOW_TRAFFICLIGHT_H
#define CITYFLOW_TRAFFICLIGHT_H

#include <vector>
#include <string>
#include <ostream>
#include <iostream>
#include <iomanip>

namespace CityFlow {
    class Intersection;

    class RoadLink;

    class RoadNet;

    class TrafficLight;

    class LightPhase {
        friend class RoadNet;
        friend class RoadLink;
        friend class TrafficLight;
        friend std::ostream& operator << (std::ostream& os, LightPhase &a) {
            os << std::fixed << "Phase:" <<std::to_string(a.phase)<<", time: "
            << std::setprecision(2)<<std::setfill(' ')<<std::setw(5)<<std::to_string(int(a.time))<< std::setw(15)<<" roadLinkAvailable:";
            for (const auto b:a.roadLinkAvailable) os<<" "<<to_string(b);
            return os;
            
        }
    private:
        unsigned int phase = 0;
        double time = 0.0;
        std::vector<bool> roadLinkAvailable;
    };

    class TrafficLight {
        friend class RoadNet;
        friend class Archive;
    private:
        Intersection *intersection = nullptr;
        std::vector<LightPhase> phases;
        std::vector<int> roadLinkIndices;
        double remainDuration = 0.0;
        int curPhaseIndex = 0;
        int verbose = 0;
        
    public:
        void init(int initPhaseIndex);

        int getCurrentPhaseIndex();

        LightPhase &getCurrentPhase();

        Intersection &getIntersection();

        std::vector<LightPhase> &getPhases();

        void passTime(double seconds);

        void setPhase(int phaseIndex);
        
        int bumpPhase();
        
        void reset();
        
        void setVerbose(int _verbose);
    };
}

#endif //CITYFLOW_TRAFFICLIGHT_H
