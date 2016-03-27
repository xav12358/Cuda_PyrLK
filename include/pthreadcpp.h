/*
#########################################################################
#
#  This file is part of trustyRC.
#
#  trustyRC, fully modular IRC robot 
#  Copyright (C) 2006-2008 Nicoleau Fabien 
#
#  trustyRC is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  trustyRC is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with trustyRC.  If not, see <http://www.gnu.org/licenses/>.
#
#########################################################################
*/

/** @file pthread.h
 * @brief PThread header file
 */

#ifndef PTHREAD_H
#define PTHREAD_H

#include <pthread.h>


class PThread
{

public:
    PThread();
    ~PThread();

    /// pthread handle
    pthread_t* handle;
    /// running status
    bool running;
    /// finished status
    bool finished;
    /// threaded function
    /// Check if the thread is running
    bool IsRunning();
    /// Check if the thread is finished
    bool IsFinished();
    /// Join thread
    void* Join();



public:


    //static void * ThrdProcess(void *targ) = 0 ; // la fonction appelée par pthread_create


};


#endif
