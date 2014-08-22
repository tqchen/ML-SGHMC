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

#ifndef _APEX_TASK_H_
#define _APEX_TASK_H_

#include "apex_config.h"
#include <vector>
#include <cstring>

namespace apex_utils{
    /* interface of task program */
    class ITask{
    public:
        virtual void set_param( const char *name , const char *val ) = 0;
        virtual void set_task ( const char *task ){}
        virtual void print_task_help( FILE *fo ) const = 0;
        virtual void run_task( void )= 0;
    public:
        virtual ~ITask(){}
    };
    
    inline int run_task( int argc , char *argv[] , ITask *tsk ){
        if( argc < 2 ){
            tsk->print_task_help( stdout );
            return 0;            
        }   
        tsk->set_task( argv[1] );
        
        for( int i = 2 ; i < argc ; i ++ ){
            char name[256],val[256];
            if( sscanf( argv[i] ,"%[^=]=%[^\n]", name , val ) == 2 ){
                tsk->set_param( name , val );
            }
        }
        tsk->run_task();
        return 0;
    } 

    // running task with config in argv[1]
    inline int run_config_task( int argc , char *argv[] , ITask *tsk ){
        if( argc < 2 ){
            tsk->print_task_help( stdout );
            return 0;            
        }

        ConfigIterator itr( argv[1] );
        while( itr.next() ){
            tsk->set_param( itr.name() , itr.val() );
        }
        
        for( int i = 2 ; i < argc ; i ++ ){
            char name[256],val[256];
            if( sscanf( argv[i] ,"%[^=]=%[^\n]", name , val ) == 2 ){
                tsk->set_param( name , val );
            }   
        }
        tsk->run_task();
        return 0;
    } 
};

namespace apex_utils{
    // running task with config in argv[1], and enable section support
    inline int run_section_task( int argc , char *argv[] , ITask *tsk ){
        if( argc < 2 ){
            printf("General Usage:<config.conf> [name=val]... [enable:sec_name]\n");
            tsk->print_task_help( stdout );
            return 0;     
        }
        std::vector< const char* > sec;
        sec.push_back( "global" );
        
        for( int i = 2 ; i < argc ; i ++ ){
            if( !strncmp( "enable:", argv[i], 7 ) ){
                sec.push_back( argv[i] + 7 );
            }
        }
        
        bool sec_enable = true;
        ConfigIterator itr( argv[1] );
        while( itr.next() ){
            if( !strcmp( itr.name(), "section" ) ){
                char sname[ 256 ];
                apex_utils::assert_true( sscanf( itr.val(), "%[^:]:", sname ) == 1, ": is missing" );
                sec_enable = false;
                for( size_t j = 0; j < sec.size(); j ++ ){
                    if( !strcmp( sname, sec[j] ) ){
                        sec_enable = true; break;
                    }
                }
                
            }else{
                if( sec_enable ){
                    tsk->set_param( itr.name() , itr.val() );
                }
            }
        }
        
        for( int i = 2 ; i < argc ; i ++ ){
            char name[256],val[256];
            if( sscanf( argv[i] ,"%[^=]=%[^\n]", name , val ) == 2 ){
                tsk->set_param( name , val );
            }   
        }
        tsk->run_task();
        return 0;
    }     
};
#endif
