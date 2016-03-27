#include <include/pthreadcpp.h>


////////////////////////////////////////////////////
/// \brief PThread::PThread
///
PThread::PThread()
{
    this->finished = false;
    this->running = false;
    this->handle = new pthread_t;
}


////////////////////////////////////////////////////
/// \brief PThread::~PThread
///
PThread::~PThread()
{
    //delete handle;
}


////////////////////////////////////////////////////
/// \brief PThread::IsRunning
/// \return
///
bool PThread::IsRunning() {
    return this->running;
}

////////////////////////////////////////////////////
/// \brief PThread::IsFinished
/// \return
///
bool PThread::IsFinished() {
    return this->finished;
}


////////////////////////////////////////////////////
/// \brief PThread::join
/// \return
///
void* PThread::Join() {
   void* ret;
   pthread_join(*handle,&ret);
   return ret;
}
