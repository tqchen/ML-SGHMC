/*
 *  Copyright 2009-2010 APEX Data & Knowledge Management Lab, Shanghai Jiao Tong University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*!
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 * 
 * Acknowledgement: The MAC support part of this code is provided by Artemy Kolchinsky
 */

#ifndef _APEX_THREAD_H_
#define _APEX_THREAD_H_

// this header include the necessary resource for multi-threading
// we try to make it portable and limit the functions necessary

#ifdef _MSC_VER

#include "apex_utils.h"
#include <windows.h>
#include <process.h>

// thread implementation in WINDOWS, not tested for correctless
namespace apex_thread{

    class Semaphore{
    private:
        HANDLE sem;
    public :
        inline void init( int init_val ){
            sem = CreateSemaphore( NULL, init_val, 10, NULL );
            apex_utils::assert_true( sem != NULL, "create Semaphore error");
        }
        inline void destroy(){
            CloseHandle( sem );
        }        
        inline void wait(){
            apex_utils::assert_true( WaitForSingleObject( sem, INFINITE ) == WAIT_OBJECT_0, "WaitForSingleObject error" );
        }
        inline void post(){
            apex_utils::assert_true( ReleaseSemaphore( sem, 1, NULL )  != 0, "ReleaseSemaphore error");
        }       
    };
    
    class Thread{
    private:
        HANDLE    thread_handle;                
        unsigned  thread_id;                
    public :
        inline void start( unsigned int __stdcall entry( void* ), void *param ){
            thread_handle = (HANDLE)_beginthreadex( NULL, 0, entry, param, 0, &thread_id );
        }        

        inline int join(){
            WaitForSingleObject( thread_handle, INFINITE );                                 
            return 0;
        }        
    };

	inline void thread_exit( void *status ){
		_endthreadex(0);
    }
};

#define APEX_THREAD_PREFIX unsigned int __stdcall 

#else
// thread interface using g++ 

#include <semaphore.h>
#include <pthread.h>


namespace apex_thread{
    /*!\brief semaphore class */
    class Semaphore{
#ifdef __APPLE__
    private:
        sem_t* semPtr;
        char sema_name[20];
     
    public:
         inline void gen_random_string(char *s, const int len) {
             static const char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" ;
             for (int i = 0; i < len; ++i) {
               s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
             }
             s[len] = 0;
        }
        inline void init( int init_val ){
            sema_name[0]='/'; 
            sema_name[1]='s'; 
            sema_name[2]='e'; 
            sema_name[3]='/'; 
            gen_random_string(&sema_name[4], 16);
            if((semPtr = sem_open(sema_name, O_CREAT, 0644, init_val)) == SEM_FAILED) {
                perror("sem_open");
                exit(1);
            }
            apex_utils::assert_true( semPtr != NULL, "create Semaphore error");
        }
        inline void destroy(){
            if (sem_close(semPtr) == -1) {
                perror("sem_close");
                exit(EXIT_FAILURE);
            }
            if (sem_unlink(sema_name) == -1) {
                perror("sem_unlink");
                exit(EXIT_FAILURE);
            }
        }
        inline void wait(){
            sem_wait( semPtr );
        }
        inline void post(){
            sem_post( semPtr );
        }       

#else
    private:
        sem_t sem;
    public:
        inline void init( int init_val ){
            sem_init( &sem, 0, init_val );
        }
        inline void destroy(){
            sem_destroy( &sem );
        }
        inline void wait(){
            sem_wait( &sem );
        }
        inline void post(){
            sem_post( &sem );
        }       
#endif
        
    };
    /*!\brief simple thread class */
    class Thread{
    private:
        pthread_t thread;                
    public :
        inline void start( void * entry( void* ), void *param ){
            pthread_attr_t attr;
            pthread_attr_init( &attr );
            pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
            pthread_create( &thread, &attr, entry, param );
        }        

        inline int join(){
            void *status;
            return pthread_join( thread, &status );
        }
    };    
    
    inline void thread_exit( void *status ){
        pthread_exit( status );
    }
};

#define APEX_THREAD_PREFIX void *

#endif

#endif

