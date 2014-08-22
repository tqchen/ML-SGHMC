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
 */

#ifndef _APEX_BUFFER_LOADER_H_
#define _APEX_BUFFER_LOADER_H_

#include <vector>
#include <cstring>
#include <cstdlib>
#include <climits>
#include "apex_thread.h"
#include "apex_utils.h"

namespace apex_utils{    
    /*!\brief
     * buffered loading iterator that uses multithread
     * this template method will assume the following paramters
     * \tparam Elem elememt type to be buffered
     * \tparam ElemFactory factory type to implement in order to use thread buffer
     */
    template<typename Elem, typename ElemFactory>
    class ThreadBufferIterator{
    public :
        // factory object used to load configures
        ElemFactory factory;
        // size of buffer
        int  buf_size;
    private:
        // index in current buffer
        int buf_index;
    private: 
        // indicate which one is current buffer
        int current_buf;
        // max limit of visit, also marks termination
        int endA, endB;
        // double buffer, one is accessed by loader
        // the other is accessed by consumer 
        // buffer of the data
        std::vector<Elem> bufA, bufB;
    private:
        // initialization end
        bool init_end;
        // singal whether the data is loaded
        bool data_loaded;
        // signal to kill the thread
        bool destroy_signal;
        // thread object
        apex_thread::Thread    loader_thread;
        // signal of the buffer 
        apex_thread::Semaphore loading_end, loading_need;
    private:
        /*!
         * \brief slave thread 
         * this implementation is like producer-consumer style
         */
        inline void run_loader(){
            while( !destroy_signal ){
                // sleep until loading is needed
                loading_need.wait();

                std::vector<Elem> &buf = current_buf ? bufB : bufA;
                int i;
                for( i = 0; i < buf_size ; i ++ ){
                    if( !factory.load_next( buf[i] ) ){
                        int &end = current_buf ? endB : endA;
                        end = i; // marks the termination
                        break;
                    }
                }

                // signal that loading is done
                data_loaded = true;
                loading_end.post();
            }
        }   
        /*!\brief entry point of loader thread */
        inline static APEX_THREAD_PREFIX loader_entry( void *pthread ){
            static_cast< ThreadBufferIterator<Elem,ElemFactory>* >( pthread )->run_loader();
            apex_thread::thread_exit( NULL );
            return NULL;
        }
        /*!\brief start loader thread */
        inline void start_loader(){
            destroy_signal = false; 
            // set param
            current_buf = 1;

            loading_need.init( 1 );
            loading_end .init( 0 );            
            // reset terminate limit
            endA = endB = buf_size;
            loader_thread.start( loader_entry, this );
            // wait until first part of data is loaded
            loading_end.wait();
            // set current buf to right value
            current_buf = 0; 
            // wake loader for next part
            data_loaded = false;
            loading_need.post();
            
            buf_index = 0; 
        }
    private:
        /*!\brief switch double buffer */
        inline void switch_buffer(){
            loading_end.wait();           
            // loader shall be sleep now, critcal zone!
            current_buf = !current_buf;
            // wake up loader
            data_loaded = false;            
            loading_need.post();
        }                
        
    public :        
        /*!\brief constructor */
        ThreadBufferIterator(){
            this->init_end = false;
            this->buf_size = 30;
        }
        ~ThreadBufferIterator(){
            if( init_end ) this->destroy();
        }
        /*!\brief set parameter, will also pass the parameter to factory */
        inline void set_param( const char *name, const char *val ){
            if( !strcmp( name, "buffer_size") ) buf_size  = atoi( val );
            factory.set_param( name, val );
        }

        /*!
         * \brief initalize the buffered iterator
         * \param param a initialize parameter that will pass to factory, ignore it if not necessary
         * \return false if the initlization can't be done, e.g. buffer file hasn't been created 
         */
        inline bool init( int param = 0 ){            
            if( !factory.init( param ) ) return false;
            
            for( int i = 0; i < buf_size; i ++ ){
                bufA.push_back( factory.create() );
                bufB.push_back( factory.create() );
            }                    
            this->init_end = true;    
            this->start_loader();
            return true;
        }

        
        /*!\brief place the iterator before first value */
        inline void before_first( void ){
            // wait till last loader end
            loading_end.wait();
            // critcal zone
            current_buf = 1;
            factory.before_first();
            // reset terminate limit
            endA = endB = buf_size;
            // wake up loader for first part
            loading_need.post();
            // wait til first part is loaded
            loading_end.wait();
            // set current buf to right value
            current_buf = 0; 
            // wake loader for next part
            data_loaded = false;
            loading_need.post();

            // set buffer value
            buf_index = 0;
        }

        /*! \brief destroy the buffer iterator, will deallocate the buffer */
        inline void destroy( void ){
            // wait until the signal is consumed
            this->destroy_signal = true;
            loading_need.post();
            loader_thread.join();
            loading_need.destroy();
            loading_end .destroy();
            
            for( size_t i = 0; i < bufA.size(); i ++ ){
                factory.free_space( bufA[i] );
            }
            for( size_t i = 0; i < bufB.size(); i ++ ){
                factory.free_space( bufB[i] );
            }
            bufA.clear(); bufB.clear(); 
            factory.destroy();    
            this->init_end = false;        
        }

        /*! 
         * \brief get the next element needed in buffer
         * \param elem element to store into
         * \return whether reaches end of data
         */
        inline bool next( Elem &elem ){
            // end of buffer try to switch
            if( buf_index == buf_size ){
                this->switch_buffer();
                buf_index = 0;
            }
            if( buf_index >= ( current_buf ? endA : endB ) ){   
                return false;
            }
            std::vector<Elem> &buf = current_buf ? bufA : bufB;
            elem = buf[ buf_index ];            
            buf_index ++; 

            return true;
        }        
        /*!
         * \brief get the factory object
         */
        inline const ElemFactory &get_factory() const{
            return factory;
        }
    };    
};

#endif

