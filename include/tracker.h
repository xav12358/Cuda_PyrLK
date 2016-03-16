#ifndef TRACKER_H
#define TRACKER_H

typedef enum{
    WaitFirstFrame,
    WaitSecondFrame,
    Process,
    Lost
}StateTracker;


class tracker
{
    StateTracker eState;
public:
    tracker();
};

#endif // TRACKER_H
