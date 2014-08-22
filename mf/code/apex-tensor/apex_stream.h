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

#ifndef _APEX_STREAM_H_
#define _APEX_STREAM_H_

#include <cstdio>
/*!
 * \file apex_stream.h
 * \brief general stream interface
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn 
 */
namespace apex_tensor{
    /*! \brief interface of stream I/O, used to serialize tensor data */
    class IStream{
    public:
        /*! 
         * \brief read data from stream
         * \param ptr pointer to memory buffer
         * \param size size of block
         * \return usually is the size of data readed
         */
        virtual size_t read( void *ptr, size_t size ) = 0;        
        /*! 
         * \brief write data to stream
         * \param ptr pointer to memory buffer
         * \param size size of block
         */
        virtual void write( const void *ptr, size_t size ) = 0;
        /*! \brief virtual destructor */
        virtual ~IStream( void ){}
    };
    
};

namespace apex_tensor{
    /*! \brief implementation of file i/o stream */
    class FileStream: public IStream{
    private:
        FILE *fp;
    public:        
        FileStream( FILE *fp ){
            this->fp = fp;
        }
        virtual size_t read( void *ptr, size_t size ){
            return fread( ptr, size, 1, fp );
        }
        virtual void write( const void *ptr, size_t size ){
            fwrite( ptr, size, 1, fp );
        }
        inline void close( void ){
            fclose( fp );
        }
    };
};
#endif

