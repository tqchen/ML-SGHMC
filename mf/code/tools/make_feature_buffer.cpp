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
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "apex_svd_data.h"
#include "apex-utils/apex_utils.h"

using namespace apex_svd;

int main( int argc, char *argv[] ){
    if( argc < 3 ){
        printf("Usage:make_feature_buffer <input> <output> [options...]\n"\
               "options: -batch_size batch_size, -scale_score scale_score\n"\
               "example: make_feature_buffer input output -batch_size 100 -scale_score 1\n"\
               "\tmake a buffer used for svd-feature\n"\
               "\tbatch_size is the mini-batch size for the data entry, must be set smaller than total number of entrys\n"\
               "\tscale_score will divide the score by scale_score, we suggest to scale the score to 0-1 if it's too big\n");
        return 0; 
    }
    int batch_size = 1000;
    IDataIterator<SVDFeatureCSR::Elem> *loader = create_csr_iterator( input_type::TEXT_FEATURE );
    loader->set_param( "scale_score", "1.0" );

    time_t start = time( NULL );
    for( int i = 3; i < argc; i ++ ){
        if( !strcmp( argv[i], "-batch_size") ){
            batch_size = atoi( argv[++i] ); continue;
        }
        if( !strcmp( argv[i], "-scale_score") ){
            loader->set_param( "scale_score", argv[++i] ); continue;
        }
    }
    
    loader->set_param( "data_in", argv[1] );
    loader->init();
    printf("start creating buffer...\n");
    create_binary_buffer( argv[2], loader, batch_size );
    printf("all generation end, %lu sec used\n", (unsigned long)(time(NULL) - start) );
    delete loader;
    return 0;
}
