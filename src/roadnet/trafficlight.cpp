#include "roadnet/trafficlight.h"
#include "roadnet/roadnet.h"

namespace CityFlow {

    void TrafficLight::init(int initPhaseIndex) {
        if (intersection->isVirtual)
            return;
        this->curPhaseIndex = initPhaseIndex;
        this->remainDuration = phases[initPhaseIndex].time;
    }

    int TrafficLight::getCurrentPhaseIndex() {
        return this->curPhaseIndex;
    }

    LightPhase &TrafficLight::getCurrentPhase() {
        return this->phases[this->curPhaseIndex];
    }

    Intersection &TrafficLight::getIntersection() {
        return *this->intersection;
    }

    std::vector<LightPhase> &TrafficLight::getPhases() {
        return phases;
    }

    
    void TrafficLight::passTime(double seconds) {
        if(intersection->isVirtual)
            return;
        remainDuration -= seconds;
        if (0 && (int) phases.size()>1){
            //std::cout<<" (int) phases.size()"<<std::endl;
        }
        while (remainDuration <= 0.0) {
            setPhase((curPhaseIndex + 1) % (int) phases.size());
            remainDuration += phases[curPhaseIndex].time;
        }
    }
    
    void TrafficLight::bumpPhase() {
        /* Bump to next phase regardless of time */
        if(intersection->isVirtual)
            return;
        setPhase((curPhaseIndex + 1) % (int) phases.size());
        remainDuration = phases[curPhaseIndex].time;
    }

    void TrafficLight::setPhase(int phaseIndex) {
        curPhaseIndex = phaseIndex;
        std::cout<<" Phase changed to:"<<phases[curPhaseIndex]<<std::endl;
    }

    void TrafficLight::reset() {
        init(0);
    }

    void TrafficLight::setVerbose(int _verbose){
        verbose = _verbose;
    }
}
