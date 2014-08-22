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
#include <climits>

#include "apex_svd.h"
#include "apex-utils/apex_task.h"
#include "apex-utils/apex_utils.h"
#include "apex-utils/apex_config.h"
#include "apex-tensor/apex_random.h"

namespace apex_svd{
    class SVDTrainTask : public apex_utils::ITask{
    private:
        // type of model 
        SVDTypeParam mtype;
        ISVDTrainer   *svd_trainer;
    private:
        int input_type;
        IDataIterator<SVDFeatureCSR::Elem> *itr_csr;
        IDataIterator<SVDPlusBlock>        *itr_plus;
    private:
        // using hadoop mode
        int hadoop_mode;
        // initialize end
        int init_end;
        // name of job
        char name_job[ 256 ];
        // name of configure file
        char name_config[ 256 ];
        // 0 = new layer, 1 = continue last layer's training
        int task;
        // continue from model folder
        int continue_training;
        // enforce to change to new extend type
        int enforce_extend;
        // enforce to change to new active type
        int enforce_active;
        // whether to be silent 
        int silent;        
        // input model name 
        char name_model_in[ 256 ];
        // start counter of model
        int start_counter;                
        // folder name of output  
        char name_model_out_folder[ 256 ];
    private:
        float print_ratio;
        int   do_save_model;
        int   num_round, train_repeat, max_round;
    private:
        apex_utils::ConfigSaver cfg;
    private:
        inline void reset_default(){
            init_end = 0;
            enforce_extend = 0;
            enforce_active = 0;
            this->svd_trainer = NULL;
            this->input_type  = input_type::BINARY_BUFFER;
            this->itr_csr     = NULL;
            this->itr_plus    = NULL;
            strcpy( name_config, "config.conf" );
            strcpy( name_job, "" );
            strcpy( name_model_out_folder, "models" );
            print_ratio = 0.05f;
            train_repeat = 1;
            do_save_model = 1;
            num_round = 10;            
            hadoop_mode = task = silent = start_counter = 0; 
            max_round = INT_MAX;
            continue_training = 0;            
        }
    public:
        SVDTrainTask(){
            this->reset_default();
        }
        virtual ~SVDTrainTask(){           
            if( init_end ){
                delete svd_trainer;
                if( itr_csr != NULL ) delete itr_csr;
                if( itr_plus!= NULL ) delete itr_plus;
            }
        }
    private:
        inline void set_param_inner( const char *name, const char *val ){
            if( !strcmp( name,"task"   ))             task    = atoi( val ); 
            if( !strcmp( name,"seed"   ))             apex_random::seed( atoi( val ) ); 
            if( !strcmp( name,"continue"))            continue_training = atoi( val ); 
            if( !strcmp( name,"max_round"))           max_round = atoi( val ); 
            if( !strcmp( name,"start_counter" ))      start_counter = atoi( val );
            if( !strcmp( name,"model_in" ))           strcpy( name_model_in, val ); 
            if( !strcmp( name,"model_out_folder" ))   strcpy( name_model_out_folder, val ); 
            if( !strcmp( name,"num_round"  ))         num_round    = atoi( val ); 
            if( !strcmp( name,"train_repeat"  ))      train_repeat = atoi( val );
            if( !strcmp( name, "silent") )            silent     = atoi( val );
            if( !strcmp( name, "save_model") )        do_save_model = atoi( val );
            if( !strcmp( name, "hadoop_mode") )       hadoop_mode= atoi( val );
            if( !strcmp( name, "enforce_extend") )    enforce_extend = atoi( val );
            if( !strcmp( name, "enforce_active") )    enforce_active = atoi( val );
            if( !strcmp( name, "job") )               strcpy( name_job, val ); 
            if( !strcmp( name, "print_ratio") )       print_ratio= (float)atof( val );            
            if( !strcmp( name, "input_type"  ))       input_type = atoi( val ); 
            mtype.set_param( name, val );
        }
        
        inline void configure( void ){
            apex_utils::ConfigIterator itr( name_config );
            while( itr.next() ){
                cfg.push_back( itr.name(), itr.val() );
            }            

            cfg.before_first();
            while( cfg.next() ){
                set_param_inner( cfg.name(), cfg.val() );
            }
            // decide format type
            mtype.decide_format( input_type == 2 ? svd_type::USER_GROUP_FORMAT : svd_type::AUTO_DETECT );
        }

        // configure iterator
        inline void configure_iterator( void ){
            if( mtype.format_type == svd_type::USER_GROUP_FORMAT ){
                this->itr_plus = create_plus_iterator( input_type );
            }else{
                this->itr_csr = create_csr_iterator( input_type );
            }

            cfg.before_first();
            while( cfg.next() ){
                if( itr_csr != NULL  ) itr_csr->set_param( cfg.name(), cfg.val() );
                if( itr_plus != NULL ) itr_plus->set_param( cfg.name(), cfg.val() );
            }                
            if( itr_csr != NULL ) itr_csr->init();
            if( itr_plus!= NULL ) itr_plus->init();
        }

        inline void configure_trainer( void ){
            cfg.before_first();
            while( cfg.next() ){
                svd_trainer->set_param( cfg.name(), cfg.val() );
            }
        }
        
        // load in latest model from model_folder
        inline int sync_latest_model( void ){
            FILE *fi = NULL, *last = NULL;           
            char name[256];
            int s_counter = start_counter;
            do{
                if( last != NULL ) fclose( last );
                last = fi;
                sprintf(name,"%s/%04d.model" , name_model_out_folder, s_counter++ );
                fi = fopen64( name, "rb");                
            }while( fi != NULL ); 

            if( last != NULL ){
				SVDTypeParam mold = mtype;
                apex_utils::assert_true( fread( &mtype, sizeof(SVDTypeParam), 1, last ) > 0, "loading model" );
                if( enforce_extend != 0 ) mtype.extend_type = mold.extend_type;  
                if( enforce_active != 0 ) mtype.active_type = mold.active_type;  
                svd_trainer = create_svd_trainer( mtype );
                svd_trainer->load_model( last );
                start_counter = s_counter - 1;
                fclose( last );
                return 1;
            }else{
                return 0;
            }
        }
        inline void load_model( void ){
            FILE *fi = apex_utils::fopen_check( name_model_in, "rb" );
			SVDTypeParam mold = mtype;
			apex_utils::assert_true( fread( &mtype, sizeof(SVDTypeParam), 1, fi ) > 0, "loading model" );
			if( enforce_extend != 0 ) mtype.extend_type = mold.extend_type;  
			if( enforce_active != 0 ) mtype.active_type = mold.active_type;  
            svd_trainer = create_svd_trainer( mtype );
            svd_trainer->load_model( fi );
            fclose( fi );
        }
        
        inline void save_model( void ){
            char name[256];
            sprintf(name,"%s/%04d.model" , name_model_out_folder, start_counter ++ );
            if( do_save_model == 0 ) return;
            
            if( do_save_model < 0 ){
                if( start_counter - 1 == 0 ) return;
                if( (start_counter - 1) % (-do_save_model) != 0 ) return;
            }

            FILE *fo  = apex_utils::fopen_check( name, "wb" );            
            fwrite( &mtype, sizeof(SVDTypeParam), 1, fo );
            svd_trainer->save_model( fo );
            fclose( fo );
        }            
        
        
        inline void init( void ){            
            // configure the parameters
            this->configure();     
            // configure trainer
            if( continue_training != 0 && sync_latest_model() != 0 ){
                this->configure_trainer();
            }else{                                   
                continue_training = 0; 
                switch( task ){
                case 0: 
                    svd_trainer = create_svd_trainer( mtype );
                    this->configure_trainer();
                    svd_trainer->init_model();
                    break;
                case 1: 
                    this->load_model(); 
                    this->configure_trainer();
                    break;
                default: apex_utils::error("unknown task");
                }
            }
            this->configure_iterator();
            svd_trainer->init_trainer();
            this->init_end = 1;           
        }     

        template<typename DataType>
        inline void update( int r, unsigned long elapsed, time_t start, IDataIterator<DataType> *itr ){
            size_t total_num = itr->get_data_size();

            // exceptional case when iterator didn't provide data count
            if( total_num == 0 ) total_num = 1; 

            size_t print_step = static_cast<size_t>( floorf(total_num * print_ratio ));
            if( print_step <= 0 ) print_step = 1;
            size_t sample_counter = 0;
            DataType dt;
            for( int j = 0; j < train_repeat; j ++ ){ 
                svd_trainer->set_round( r );
                while( itr->next( dt ) ){
                    svd_trainer->update( dt );
                    if( sample_counter  % print_step == 0 ){
                        if( hadoop_mode ){
                            fprintf( stderr,"reporter:status:Round %3d[%05.1lf%%] update phase, %s\n", r,
                                     (double)sample_counter / total_num * 100.0, name_job );
                        }
                        if( !silent ){
                            elapsed = (unsigned long)(time(NULL) - start); 
                            printf("\r                                                                     \r");
                            printf("round %8d:[%2d/%05.1lf%%] %lu sec elapsed", 
                                   r , j, (double)sample_counter / total_num * 100.0, elapsed );
                            fflush( stdout );
                        }
                    }
                    sample_counter ++;
                }
                svd_trainer->finish_round();
                itr->before_first();                    
            }
        }

    public:
        virtual void set_param( const char *name , const char *val ){
            cfg.push_back_high( name, val );
        }
        virtual void set_task ( const char *task ){
            strcpy( name_config, task );
        }
        virtual void print_task_help( FILE *fo ) const {
            printf("Usage:<config> [xxx=xx]\n");
        }
        virtual void run_task( void ){
            this->init();
            if( !silent ){
                printf("initializing end, start updating\n");
            }            
            if( continue_training == 0 ){
                this->save_model();
            }

            time_t start    = time( NULL );
            unsigned long elapsed = 0;            
            int cc = max_round; 
            while( start_counter <= num_round && cc -- ) {
                if( itr_csr != NULL )
                    this->update( start_counter-1, elapsed, start, itr_csr );
                if( itr_plus != NULL )
                    this->update( start_counter-1, elapsed, start, itr_plus );

                elapsed = (unsigned long)(time(NULL) - start); 
                this->save_model();
            }

            if( !silent ){
                printf("\nupdating end, %lu sec in all\n", elapsed );
            }                        
        }        
    };
};

int main( int argc, char *argv[] ){
    apex_random::seed( 10 );
    apex_svd::SVDTrainTask tsk;
    return apex_utils::run_task( argc, argv, &tsk );
}

