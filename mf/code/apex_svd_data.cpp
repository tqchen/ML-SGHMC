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

#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include "apex_svd_data.h"
#include "apex-utils/apex_utils.h"

namespace apex_svd{
    // basic loader for three column format
    class SVDBasicLoader : public IDataIterator<SVDFeatureCSR::Elem>{
    private:
        FILE *fi;
        float scale_score;
        char name_data[ 256 ];
        unsigned index[ 2 ];
        float    value[ 2 ];
    public:
        SVDBasicLoader(){
            fi = NULL;
            scale_score = 1.0f;
            strcpy( name_data, "NULL");
            value[ 0 ] = value[ 1 ] = 1.0f;
        }
        virtual ~SVDBasicLoader(){
            if( fi != NULL ) fclose( fi ); 
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "scale_score" ) ) scale_score = (float)atof( val );
            if( !strcmp( name, "data_in" ) )     strcpy( name_data, val );
        }
        virtual void init( void ){
            fi = apex_utils::fopen_check( name_data, "r" );
        }
        virtual bool next( SVDFeatureCSR::Elem &e ){
            if( fscanf( fi,"%d%d%f%*[^\n]\n", &index[0], &index[1], &e.label ) != 3 ) return false;
            e.label /= scale_score;
            e.num_global = 0; e.num_ufactor = 1; e.num_ifactor = 1;
            e.set_space( &index[0], &value[0] );
            return true;
        }
        virtual void before_first( void ){
            fseek( fi, 0, SEEK_SET );
        }
    };
};

namespace apex_svd{
    class SVDFeatureCSRLoader : public IDataIterator<SVDFeatureCSR::Elem>{
    private:
        FILE *fi;
        float scale_score;
        char name_data[ 256 ];
        std::vector<unsigned> index;
        std::vector<float>    value;
    public:
        SVDFeatureCSRLoader(){
            fi = NULL;
            scale_score = 1.0f;
            strcpy( name_data, "NULL");
        }
        virtual ~SVDFeatureCSRLoader(){
            if( fi != NULL ) fclose( fi ); 
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "scale_score" ) ) scale_score = (float)atof( val );
            if( !strcmp( name, "data_in" ) )     strcpy( name_data, val );
        }
        virtual void init( void ){
            fi = apex_utils::fopen_check( name_data, "r" );
        }
        virtual bool next( SVDFeatureCSR::Elem &e ){
            if( fscanf( fi,"%f%d%d%d", &e.label, &e.num_global, &e.num_ufactor, &e.num_ifactor ) != 4 ) return false;
            e.label /= scale_score;
            int n = e.total_num();
            index.resize( static_cast<size_t>(n) );
            value.resize( static_cast<size_t>(n) );
            e.set_space( &index[0], &value[0] );

            for( int i = 0; i < n; i ++ ){
                if( fscanf( fi, "%d:%f", &e.index_global[i], &e.value_global[i] ) !=2 ) {
                    fprintf( stderr, "error loading line=%d\n", i );
                    apex_utils::error("error"); 
                }
            }   
            return true;
        }
        virtual void before_first( void ){
            fseek( fi, 0, SEEK_SET );
        }
    };
};

namespace apex_svd{    
    struct SVDFeatureCSRFactory{
    private:
        struct Param{
            int num_batch;
            int batch_size;
            int max_batch_num;
        };
    private:
        FILE *fi;
        int   silent;
        char  name_buf[ 256 ];
        char  name_train[ 256 ];
    public:
        SVDFeatureCSRLoader loader;
    public:
        static inline void create_buffer( const char *name_buf, IDataIterator<SVDFeatureCSR::Elem> *loader, int batch_size ){
            Param param;
            std::vector<SVDFeatureCSR::Elem> data;

            FILE *fo = apex_utils::fopen_check( name_buf, "wb" );
            fwrite( &param, sizeof(Param) , 1, fo );
            param.batch_size    = batch_size;   
            param.max_batch_num = 0;
            param.num_batch     = 0;
            
            SVDFeatureCSR::Elem e;           
            loader->before_first();
            while( true ){
                int num_val = 0;
                int num_row = 0;
                for( int i = 0; i < param.batch_size && loader->next(e); i ++ ){
                    data.push_back( e.clone() );                    
                    num_val += e.total_num();
                    num_row ++ ;
                } 
                if( num_row == 0 ) break;

                fwrite( &num_row, sizeof(int), 1, fo );
                fwrite( &num_val, sizeof(int), 1, fo );                
                if( num_val > param.max_batch_num ) param.max_batch_num = num_val;

                // row_ptr
                int row_ptr = 0;
                fwrite( &row_ptr, sizeof(int), 1, fo );
                for( int j = 0; j < num_row; j ++ ){
                    row_ptr += data[j].num_global;
                    fwrite( &row_ptr, sizeof(int), 1, fo );  
                    row_ptr += data[j].num_ufactor;
                    fwrite( &row_ptr, sizeof(int), 1, fo );  
                    row_ptr += data[j].num_ifactor;
                    fwrite( &row_ptr, sizeof(int), 1, fo );  
                }
                // row label
                for( int j = 0; j < num_row; j ++ ){
                    fwrite( &data[j].label, sizeof(float), 1, fo );  
                }
                // feat index
                for( int j = 0; j < num_row; j ++ ){
                    fwrite( data[j].index_global , sizeof(int), data[j].num_global , fo );  
                    fwrite( data[j].index_ufactor, sizeof(int), data[j].num_ufactor, fo );  
                    fwrite( data[j].index_ifactor, sizeof(int), data[j].num_ifactor, fo );  
                }
                // feat value
                for( int j = 0; j < num_row; j ++ ){
                    fwrite( data[j].value_global , sizeof(int), data[j].num_global , fo );  
                    fwrite( data[j].value_ufactor, sizeof(int), data[j].num_ufactor, fo );  
                    fwrite( data[j].value_ifactor, sizeof(int), data[j].num_ifactor, fo );  
                }
                param.num_batch ++;
                                
                for( size_t i = 0; i < data.size(); i ++ )
                    data[i].free_space();
                data.clear();
            }
                        
            fseek( fo, 0, SEEK_SET );
            fwrite( &param, sizeof(Param) , 1, fo );
            
            fclose( fo );            
        }
    private:
        inline void create_buffer(){
            if( silent == 0 ) printf("start creating new buffer \'%s\' ...\n", name_buf );
                       
            loader.init();
            create_buffer( name_buf, &loader, param.batch_size );
            
            if( silent == 0 ) printf("Buffer created in %s\n", name_buf );
        }
    private:
        int index;
    public:
        Param param;        
    public:
        SVDFeatureCSRFactory(){
            strcpy( name_buf, "svdfeature_buf" );
            silent = 0;
        }

        inline void set_param( const char *name, const char *val ){
            if( !strcmp( name, "buffer_feature" ) )   strcpy( name_buf, val );
            if( !strcmp( name, "data_in" ) )          strcpy( name_train, val );
            if( !strcmp( name, "feature_batch" ) )    param.batch_size = atoi( val );
            if( !strcmp( name, "silent" ) )           silent = atoi( val );
            loader.set_param( name, val );
        }
        
        inline int get_data_size() const{
            return param.num_batch;
        }

        inline bool init( int st ){ 
            fi = fopen64( name_buf, "rb" );            
            if( fi == NULL ){
                printf("can't open buffer %s, try to create from data_in=%s\n", name_buf, name_train );
                this->create_buffer();
                fi = apex_utils::fopen_check( name_buf, "rb" );
            }
            apex_utils::assert_true( fread( &param, sizeof(Param), 1, fi ) > 0,"Buffer Factory");
            if( silent == 0 ) printf("SVDFeatureCSRFactory: num_batch=%d\n", param.num_batch );
            this->index = 0;
            return true;
        }
        
        inline bool load_next( SVDFeatureCSR &val ){        
            if( index < param.num_batch ) {
                val.load_from_file( fi ); 
                ++ index;
                return true;
            }else{
                return false;
            }
        }            

        inline SVDFeatureCSR create(){
            SVDFeatureCSR a;
            a.num_row = param.batch_size;
            a.num_val = param.max_batch_num;
            a.alloc_space();
            return a;
        }

        inline void free_space( SVDFeatureCSR &a ){        
            a.free_space();
        }                

        inline void destroy(){
            fclose( fi );
        }    
        
        inline void before_first(){
            this->index = 0; 
            fseek( fi, sizeof(Param), SEEK_SET );
        }        
    };
};
   
#include "apex-utils/apex_buffer_loader.h"
namespace apex_svd{
    // thread iterator for CSR factory
    template<typename FactoryType>
    class SVDCSRThreadIterator: public IDataIterator<SVDFeatureCSR::Elem>{
    private:
        int idx; 
        SVDFeatureCSR dt;
        apex_utils::ThreadBufferIterator<SVDFeatureCSR,FactoryType> itr;
    public :
        SVDCSRThreadIterator(){
            idx = -1;
            itr.set_param( "buffer_size", "2" );
        }
        virtual ~SVDCSRThreadIterator(){
            itr.destroy();
        }
        virtual void set_param( const char *name, const char *val ){
            itr.set_param( name, val );
        }
        virtual void init( void ){
            itr.init();
        }
        virtual size_t get_data_size( void ){
            return static_cast<size_t>( itr.get_factory().get_data_size() ) * itr.factory.param.batch_size;
        }
        virtual void before_first( void ){
            idx = -1; itr.before_first();
        }
        virtual bool next( SVDFeatureCSR::Elem &elem ){
            if( idx == -1 || idx == dt.num_row ){
                if( !itr.next( dt ) ) return false;
                idx = 0;
            }
            elem = dt[ idx ++ ];
            return true;
        }
    };
};

// input structure for SVD++ style implict explict feedback
namespace apex_svd{
    // loader that load SVDPlus block from text data
    class SVDPlusBlockLoader: public IDataIterator<SVDPlusBlock>{
    private:
        // maximum number of lines that can be fit into a block         
        int block_max_line;
        // number of remaining lines to be loaded
        int nline_remain;
    private:
        FILE *fi, *finfo;
        char name_data[ 256 ];
        char name_info[ 256 ];
    private:
        std::vector<unsigned> index_ufeedback;
        std::vector<float>    value_ufeedback; 
    private:        
        std::vector<float> row_label;
        std::vector<int>   row_ptr;        
        std::vector<unsigned>   feat_index;
        std::vector<float>      feat_value;        
    private:
        struct Elem{
            unsigned index;
            float    value;
            inline bool operator<( const Elem &b ) const{
                return index < b.index;
            }
        };
        inline void load( std::vector<Elem> &vec, int n ){
            Elem e;
            vec.clear();
            while( n -- ){
                apex_utils::assert_true( fscanf( fi, "%d:%f", &e.index, &e.value ) == 2, "invalid feature format" );  
                vec.push_back( e );
            }
            sort( vec.begin(), vec.end() );
        }
        inline void add( const std::vector<Elem> &vec ){
            for( size_t i = 0; i < vec.size(); i ++ ){
                feat_index.push_back( vec[i].index );
                feat_value.push_back( vec[i].value );
            } 
        }
    private:
        // do buffer creation with only fi, a bit duplicate code for correctness
        float olabel;
        std::vector<Elem> og, ou, oi;
        inline bool next_onlyfi( SVDPlusBlock &e ){
            if( ou.size() == 0 ){
                int ng, nu, ni;
                if( fscanf( fi, "%f%d%d%d", 
                            &olabel, &ng, &nu, &ni ) != 4 ) return false; 
                this->load( og, ng ); 
                this->load( ou, nu );
                this->load( oi, ni );
            }            
            index_ufeedback.clear(); 
            value_ufeedback.clear();
            e.num_ufeedback = 0;
            e.extend_tag = svdpp_tag::DEFAULT;
            e.index_ufeedback = NULL;
            e.value_ufeedback = NULL;                                    

            unsigned uid = ou[0].index;

            if( this->nline_remain != 0 ){
                int start  = ( block_max_line + 1 ) / 2;
                int nrow   = static_cast<int>( row_label.size() ) - start;
                int rstart = row_ptr[ start * 3 ];
                
                memmove( &row_label[0] , &row_label [ start ] , sizeof(float) * nrow );
                memmove( &feat_index[0], &feat_index[ rstart ], sizeof(unsigned) * ( row_ptr.back() - rstart ) ); 
                memmove( &feat_value[0], &feat_value[ rstart ], sizeof(float) * ( row_ptr.back() - rstart ) ); 
                for( size_t i = start * 3; i < row_ptr.size(); i ++ ){
                    row_ptr[ i - start * 3 ] = row_ptr[ i ] - rstart; 
                }                
                row_label.resize( nrow );
                row_ptr.resize( nrow * 3 + 1 );
                feat_index.resize( row_ptr.back() );
                feat_value.resize( row_ptr.back() );               
            }else{
                row_label.resize( 0 );
                row_ptr.resize( 0 );
                row_ptr.push_back( 0 );
                feat_index.clear();
                feat_value.clear();
            }

            {
                row_label.push_back( olabel / scale_score );
                row_ptr.push_back( row_ptr.back() + static_cast<int>( og.size() ) );
                row_ptr.push_back( row_ptr.back() + static_cast<int>( ou.size() ) );
                row_ptr.push_back( row_ptr.back() + static_cast<int>( oi.size() ) );
                this->add( og ); this->add( ou ); this->add( oi );
                og.clear(); ou.clear(); oi.clear();
            }
            this->nline_remain = 0;                
            int ng, nu, ni;
            while( fscanf( fi, "%f%d%d%d", 
                           &olabel, &ng, &nu, &ni ) == 4 ){
                std::vector<Elem> vg, vu, vi;
                this->load( vg, ng ); 
                this->load( vu, nu ); 
                this->load( vi, ni ); 
                apex_utils::assert_true( vu.size() != 0, "need at least one user feature in feature file" );
                if( vu[0].index != uid ){
                    og = vg; ou = vu; oi = vi; break;
                } 
                if( row_label.size() >= static_cast<size_t>( block_max_line ) ){
                    this->nline_remain = 1;
                    og = vg; ou = vu; oi = vi; break;
                }
                row_label.push_back( olabel / scale_score );
                row_ptr.push_back( row_ptr.back() + ng );
                row_ptr.push_back( row_ptr.back() + nu );
                row_ptr.push_back( row_ptr.back() + ni );
                this->add( vg ); this->add( vu ); this->add( vi );
            }
            if( this->nline_remain != 0 ){
                e.data.num_row = ( block_max_line + 1 ) / 2 ;
            }else{
                e.data.num_row = static_cast<int>( row_label.size() );
            }
            e.data.num_val   = row_ptr[ e.data.num_row * 3 ];
            e.data.row_ptr   = &row_ptr[0];
            e.data.row_label = &row_label[0];
            e.data.feat_index= &feat_index[0];
            e.data.feat_value= &feat_value[0];
            return true;
        }
    public:
        float scale_score;
        SVDPlusBlockLoader(){
            scale_score = 1.0f;
            strcpy( name_info, "NULL" );
            fi = NULL; finfo = NULL;
            this->block_max_line = 10000; 
        }
        virtual ~SVDPlusBlockLoader(){
            if( fi != NULL ) this->close();
        }
        virtual void init( void ){            
            this->open( name_data, name_info );
            this->nline_remain = 0;
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "scale_score" ) ) scale_score = (float)atof( val );
            if( !strcmp( name, "data_in" ) )     strcpy( name_data, val );
            if( !strcmp( name, "feedback_in" ) ) strcpy( name_info, val );
            if( !strcmp( name, "block_max_line" ) ) block_max_line = atoi( val );
        }
        // load data into e, caller isn't responsible for space free
        // Loader will keep the space until next call of this function 
        virtual bool next( SVDPlusBlock &e ){
            if( finfo == NULL ) return next_onlyfi( e );

            e.extend_tag = svdpp_tag::MIDDLE_TAG;
            // load from feedback file
            if( nline_remain == 0 ){
                int num_ufeedback;                        
                if( fscanf( finfo, "%d%d", &nline_remain, &num_ufeedback ) != 2 ) return false;
                index_ufeedback.resize( static_cast<size_t>( num_ufeedback ) );
                value_ufeedback.resize( static_cast<size_t>( num_ufeedback ) );
                for( int i = 0; i < num_ufeedback; i ++ ){
                    apex_utils::assert_true( fscanf( finfo, "%d:%f", &index_ufeedback[i], &value_ufeedback[i] ) == 2, 
                                             "can't load implict feedback file" );
                }
                e.extend_tag = e.extend_tag & svdpp_tag::START_TAG;
            }
            
            // check lines to be loaded
            int num_line = nline_remain;
            if( nline_remain > block_max_line ){
                // smart arrangement to make data in similar size in each block
                int pc = ( nline_remain + block_max_line - 1 ) / block_max_line;
                num_line = ( nline_remain + pc - 1 ) / pc;
            }else{
                e.extend_tag = e.extend_tag & svdpp_tag::END_TAG;
            }
            this->nline_remain -= num_line;
            // set data 
            if( e.extend_tag == svdpp_tag::MIDDLE_TAG ){        
                e.num_ufeedback = 0;
                e.index_ufeedback = NULL;
                e.value_ufeedback = NULL;
            }else{
                e.num_ufeedback = static_cast<int>( index_ufeedback.size() );
                if( index_ufeedback.size() != 0 ){
                    e.index_ufeedback = &index_ufeedback[0];
                    e.value_ufeedback = &value_ufeedback[0];
                }
            }

            int num_elem = 0;
            row_label.resize( static_cast<size_t>( num_line ) );
            row_ptr.resize  ( static_cast<size_t>( num_line*3 + 1 ) );
            row_ptr[ 0 ] = 0;
            feat_index.clear();
            feat_value.clear();
            for( int i = 0; i < num_line; i ++ ){
                int ng, nu, ni;
                apex_utils::assert_true( fscanf( fi, "%f%d%d%d", &row_label[i], &ng, &nu, &ni ) == 4, "invalid feature format" );  
                row_label[ i ] /= scale_score;
                row_ptr[ i*3 + 1 ] = (num_elem += ng);                
                row_ptr[ i*3 + 2 ] = (num_elem += nu);
                row_ptr[ i*3 + 3 ] = (num_elem += ni);
                std::vector<Elem> vg, vu, vi;
                this->load( vg, ng ); this->add( vg );
                this->load( vu, nu ); this->add( vu );
                this->load( vi, ni ); this->add( vi );
            }            
            e.data.num_row   = num_line;
            e.data.num_val   = num_elem;
            e.data.row_ptr   = &row_ptr[0];
            e.data.row_label = &row_label[0];
            e.data.feat_index= &feat_index[0];
            e.data.feat_value= &feat_value[0];
            
            return true;
        }
        virtual void before_first(){
            fseek( fi, 0, SEEK_SET );
            if( finfo != NULL ) fseek( finfo, 0, SEEK_SET );
            og.clear(); ou.clear(); oi.clear();
            this->nline_remain = 0;
        }
        inline void close(){
            if( fi != NULL ) fclose( fi );
            if( finfo != NULL ) fclose( finfo );
            this->fi = NULL;
            this->finfo = NULL;
        }
        inline void open( const char *fname, const char *fname_info ){
            fi = apex_utils::fopen_check( fname, "r" );
            if( strcmp( fname_info, "NULL" ) ){
                finfo = apex_utils::fopen_check( fname_info, "r" );
            }else{
                finfo = NULL;
            }
        }
    };
    
    class SVDPlusBlockFactory{
    private:
        struct Param{
            int num_batch;
            int max_num_ufeedback;
            int max_num_row;
            int max_num_val;
        };
    private:
        FILE *fi;
        int   silent;
        char  name_buf[ 256 ];
    public:
        char  name_train[ 256 ], name_info[ 256 ];        
    public:
        SVDPlusBlockLoader loader;
    public:
        static inline void create_buffer( const char *name_buf, IDataIterator<SVDPlusBlock> *loader ){
            Param param;

            FILE *fo = apex_utils::fopen_check( name_buf, "wb" );
            fwrite( &param, sizeof(Param) , 1, fo );
            param.num_batch         = 0;   
            param.max_num_ufeedback = 0;
            param.max_num_row       = 0;
            param.max_num_val       = 0;
            SVDPlusBlock e;
            loader->before_first();
            while( loader->next(e) ){
                e.save_to_file( fo );
                if( e.num_ufeedback > param.max_num_ufeedback ) param.max_num_ufeedback = e.num_ufeedback;
                if( e.data.num_row  > param.max_num_row ) param.max_num_row = e.data.num_row;
                if( e.data.num_val  > param.max_num_val ) param.max_num_val = e.data.num_val;
                param.num_batch ++;
            }
                        
            fseek( fo, 0, SEEK_SET );
            fwrite( &param, sizeof(Param) , 1, fo );            
            fclose( fo );            
        }
    private:
        inline void create_buffer(){
            if( silent == 0 ) printf("start creating new buffer \'%s\' ...\n", name_buf );
                       
            loader.open( name_train, name_info );
            create_buffer( name_buf, &loader );
            loader.close();
            
            if( silent == 0 ) printf("Buffer created in %s\n", name_buf );
        }
    private:
        int index;
    public:
        Param param;        
    public:
        SVDPlusBlockFactory(){
            strcpy( name_buf, "svdplusfeature_buf" );
            silent = 0;
        }

        inline void set_param( const char *name, const char *val ){
            if( !strcmp( name, "buffer_feature" ) )   strcpy( name_buf  , val );
            if( !strcmp( name, "data_in"        ) )   strcpy( name_train, val );
            if( !strcmp( name, "feedback_in"    ) )   strcpy( name_info , val );
            if( !strcmp( name, "silent" ) )           silent     = atoi( val );
            loader.set_param( name, val );
        }
        
        inline int get_data_size() const{
            return param.num_batch;
        }

        inline bool init( int st ){ 
            fi = fopen64( name_buf, "rb" );            
            if( fi == NULL ){
                printf("can't open buffer %s, try to create from data_in=%s,feedback_in=%s\n", name_buf, name_train, name_info );
                this->create_buffer();
                fi = apex_utils::fopen_check( name_buf, "rb" );
            }
            apex_utils::assert_true( fread( &param, sizeof(Param), 1, fi ) > 0,"Buffer Factory");
            if( silent == 0 ) printf("SVDPlusBlock: num_batch=%d\n", param.num_batch );
            this->index = 0; 
            return true;
        }
        
        inline bool load_next( SVDPlusBlock &val ){        
            if( index < param.num_batch ) {
                ++index; val.load_from_file( fi );
                return true;
            }else{
                return false;
            }
        }            

        inline SVDPlusBlock create(){
            SVDPlusBlock a;
            a.num_ufeedback= param.max_num_ufeedback;
            a.data.num_row = param.max_num_row;
            a.data.num_val = param.max_num_val;
            a.alloc_space();
            return a;
        }

        inline void free_space( SVDPlusBlock &a ){        
            a.free_space();
        }                

        inline void destroy(){
            fclose( fi );
        }    
        
        inline void before_first(){
            this->index = 0; 
            fseek( fi, sizeof(Param), SEEK_SET );
        }        
    };
};

namespace apex_svd{
    // apdater for iterator
    template<typename ItrType, typename DType>
    class IteratorAdapter: public IDataIterator<DType>{
    public :
        ItrType itr;
        IteratorAdapter(){
            itr.set_param( "buffer_size", "100" );
        }
        virtual ~IteratorAdapter(){
            itr.destroy();
        }
        virtual void set_param( const char *name, const char *val ){
            itr.set_param( name, val );
        }
        virtual void init( void ){
            itr.init();
        }
        virtual size_t get_data_size( void ){
            return static_cast<size_t>( itr.get_factory().get_data_size() );
        }
        virtual void before_first( void ){
            itr.before_first();
        }
        virtual bool next( DType &elem ){
            return itr.next( elem );
        }
    };    
};

namespace apex_svd{
    // SVDFeatureCSR buffering factory
    struct SVDFeatureCSRBuffer{
    public:
        // data provider
        IDataIterator<SVDFeatureCSR::Elem> *itr_data;
        SVDFeatureCSRBuffer(){ itr_data = NULL; }
        inline void set_param( const char *name, const char *val ){
            if( itr_data != NULL ) itr_data->set_param( name, val );
        }
        
        inline size_t get_data_size() const{
            return itr_data->get_data_size();
        }

        inline bool init( int st ){ 
            itr_data->init(); 
            return true;
        }
        
        inline bool load_next( SVDFeatureCSR::Elem &val ){        
            SVDFeatureCSR::Elem e;
            if( itr_data->next(e) ){
                if( val.index_global != NULL ) val.free_space();
                val = e.clone();
                return true;
            }else{
                return false;
            }            
        }            

        inline SVDFeatureCSR::Elem create(){
            SVDFeatureCSR::Elem e;
            e.index_global = NULL;
            return e;
        }

        inline void free_space( SVDFeatureCSR::Elem &val ){        
            if( val.index_global != NULL ) val.free_space();
        }                

        inline void destroy(){
            delete itr_data;
        }    
        
        inline void before_first(){
            itr_data->before_first();
        }        
    };

    // SVDPlusBlock buffering factory
    struct SVDPlusBlockBuffer{
    public:
        // data provider
        IDataIterator<SVDPlusBlock> *itr_data;
        SVDPlusBlockBuffer(){ itr_data = NULL; }
        inline void set_param( const char *name, const char *val ){
            if( itr_data != NULL ) itr_data->set_param( name, val );
        }
        
        inline size_t get_data_size() const{
            return itr_data->get_data_size();
        }

        inline bool init( int st ){ 
            itr_data->init(); 
            return true;
        }
        
        inline bool load_next( SVDPlusBlock &val ){        
            SVDPlusBlock e;
            if( itr_data->next(e) ){
                if( val.index_ufeedback != NULL ) val.free_space();
                val = e.clone();
                return true;
            }else{
                return false;
            }            
        }            

        inline SVDPlusBlock create(){
            SVDPlusBlock e;
            e.index_ufeedback = NULL;
            return e;
        }

        inline void free_space( SVDPlusBlock &val ){        
            if( val.index_ufeedback != NULL ) val.free_space();
        }                

        inline void destroy(){
            delete itr_data;
        }    
        
        inline void before_first(){
            itr_data->before_first();
        }        
    };   
};

namespace apex_svd{
    // buffer iterator, buffering all data in memory
    template<typename DType>
    class BufferIterator: public IDataIterator<DType>{
    private:
        size_t dptr;
        std::vector<DType> buffer;
        IDataIterator<DType> *itr_data;
    public :
        BufferIterator( IDataIterator<DType> *base ){
            itr_data = base;
        }
        virtual ~BufferIterator(){
            if( itr_data != NULL ){
                delete itr_data;
            }
            for( size_t i = 0; i < buffer.size(); i ++ ){
                buffer[i].free_space();
            }
        }
        virtual void set_param( const char *name, const char *val ){
            apex_utils::assert_true( itr_data!=NULL );
            itr_data->set_param( name, val );
        }
        virtual void init( void ){
            apex_utils::assert_true( itr_data!=NULL );
            itr_data->init();
            itr_data->before_first();
            DType e;
            while( itr_data->next( e ) ){
                buffer.push_back( e.clone() );
            }
            delete itr_data;
            this->itr_data = NULL;
            this->dptr = 0;
        }
        virtual size_t get_data_size( void ){
            return buffer.size();
        }
        virtual void before_first( void ){
            this->dptr = 0;
        }
        virtual bool next( DType &elem ){
            if( dptr < buffer.size() ) {
                elem = buffer[ dptr ++ ];
                return true;
            }else{
                return false;
            }
        }
    };    
};

// We can add customized iterator in the following data
#include <ctime>
#include <algorithm>
#include "apex-tensor/apex_random.h"

namespace apex_svd{
    // pairwise sample generator
    // generate pairwise sample for each user, is frugal and useful
    class PairwiseRankGenerator : public IDataIterator<SVDPlusBlock>{
    private:
        int   sample_num;
        int   sample_max;
        int   rank_sample_method;
        int   rank_sample_pointwise;
        float rank_sample_gap;        
        int   seed_sampler_bytime;
        float pos_sample_lowerb, neg_sample_upperb;
        IDataIterator<SVDPlusBlock> *itr_data;
    private:
        std::vector<float>    row_label;
        std::vector<int>      row_ptr;
        std::vector<unsigned> findex;
        std::vector<float>    fvalue;
        std::vector<SVDFeatureCSR::Elem> pos, neg;
        inline unsigned merge( unsigned *index1, unsigned *index2,
                               float    *value1, float    *value2,
                               unsigned num1   , unsigned num2 ){
            unsigned num = 0, i = 0, j = 0;
            while( i < num1 && j < num2 ){
                if( index1[i] < index2[j] ){
                    findex.push_back( index1[i] );
                    fvalue.push_back( value1[i] );
                    ++i; ++ num;
                    continue;
                }
                if( index2[j] < index1[i] ){
                    findex.push_back( index2[j] );
                    fvalue.push_back( -value2[j] );
                    ++j; ++ num;
                    continue;
                }
                findex.push_back( index1[i] );
                fvalue.push_back( value1[i] - value2[j] );
                ++ i; ++ j; ++ num;
            }
            while( i < num1 ){
                findex.push_back( index1[i] );
                fvalue.push_back( value1[i] );
                ++i; ++ num;
            }
            while( j < num2 ){
                findex.push_back( index2[j] );
                fvalue.push_back( -value2[j] );
                ++j; ++ num;
            }
            return num;
        }

        inline void genpair_pointwise( SVDFeatureCSR::Elem p, float label ){
            for( int i = 0; i < p.num_global; i ++ ){
                findex.push_back( p.index_global[i] );
                fvalue.push_back( p.value_global[i] );
            }
            row_ptr.push_back( row_ptr.back() + p.num_global );
            int nufactor = 0;
            for( int i = 0; i < p.num_ufactor; i ++ ){
                if( p.value_ufactor[i] > 1e-6f || p.value_ufactor[i] < -1e-6f ){
                    findex.push_back( p.index_ufactor[i] );
                    fvalue.push_back( p.value_ufactor[i] );
                    nufactor ++;
                }
                
            }
            row_ptr.push_back( row_ptr.back() + nufactor );
            for( int i = 0; i < p.num_ifactor; i ++ ){
                findex.push_back( p.index_ifactor[i] );
                fvalue.push_back( p.value_ifactor[i] );
            }
            row_ptr.push_back( row_ptr.back() + p.num_ifactor );
            row_label.push_back( label ); 
        }

        inline void genpair( SVDFeatureCSR::Elem p, SVDFeatureCSR::Elem n ){
            if( rank_sample_pointwise != 0 ){
                this->genpair_pointwise( p, 1.0f );
                this->genpair_pointwise( n, 0.0f );
                return;
            }
            row_ptr.push_back( row_ptr.back() + 
                               merge( p.index_global, n.index_global,
                                      p.value_global, n.value_global,
                                      p.num_global  , n.num_global ) );
            int nufactor = 0;
            for( int i = 0; i < p.num_ufactor; i ++ ){
                if( p.value_ufactor[i] > 1e-6f || p.value_ufactor[i] < -1e-6f ){
                    findex.push_back( p.index_ufactor[i] );
                    fvalue.push_back( p.value_ufactor[i] );
                    nufactor ++;
                }
                
            }
            row_ptr.push_back( row_ptr.back() + nufactor );
            row_ptr.push_back( row_ptr.back() + 
                               merge( p.index_ifactor, n.index_ifactor,
                                      p.value_ifactor, n.value_ifactor,
                                      p.num_ifactor  , n.num_ifactor ) );            
            if( rank_sample_method / 10 == 0 ){
                row_label.push_back( 1.0f );
            }else{
                row_label.push_back( p.label - n.label ); 
            }
        }
    private:
        inline static bool cmp_rate( const SVDFeatureCSR::Elem &a, const SVDFeatureCSR::Elem &b ){
            return a.label < b.label;
        }
        inline void sample_cmp( SVDPlusBlock &e ){
            pos.resize( 0 ); neg.resize( 0 );
            for( int i = 0; i < e.data.num_row; i ++ ){
                SVDFeatureCSR::Elem el = e.data[i];
                pos.push_back( el ); neg.push_back( el );
            }
            apex_random::shuffle( neg );
            std::sort( pos.begin(), pos.end(), cmp_rate );
            for( size_t i = 0; i < neg.size(); i ++ ){
                SVDFeatureCSR::Elem el = neg[i];
                el.label -= rank_sample_gap;
                size_t left  = std::lower_bound( pos.begin(), pos.end(), el, cmp_rate ) - pos.begin();
                el.label += rank_sample_gap * 2;
                size_t right = std::lower_bound( pos.begin(), pos.end(), el, cmp_rate ) - pos.begin();
                uint32_t rng = static_cast<uint32_t>( left + pos.size() - right );
                if( rng > 0 ){
                    size_t idx = apex_random::next_uint32( rng );
                    if( idx < left ){
                        genpair( neg[i], pos[ idx ] );
                    }else{
                        genpair( pos[ right+idx-left ], neg[i] );
                    }
                }                 
            }
        } 
        // sampling using postive vs negative sample method
        inline void sample_posneg( SVDPlusBlock &e ){
            // generate positive and negative samples
            pos.resize( 0 ); neg.resize( 0 );
            for( int i = 0; i < e.data.num_row; i ++ ){
                SVDFeatureCSR::Elem el = e.data[i];
                if( el.label - pos_sample_lowerb > -1e-6f ) pos.push_back( el );  
                if( el.label - neg_sample_upperb <  1e-6f ) neg.push_back( el );  
            } 
            if( pos.size() > 0 && neg.size() > 0 ){
                apex_random::shuffle( neg );
                apex_random::shuffle( pos );
                size_t snum = neg.size();
                if( sample_num > 0 ) {
                    snum = (size_t) sample_num;
                }
                if( snum > (unsigned)sample_max ) snum = sample_max;
                for( size_t i = 0; i < snum; i ++ )
                    genpair( pos[ i % pos.size() ] , neg[ i % neg.size() ] );
            }
        }
    public :
        PairwiseRankGenerator( IDataIterator<SVDPlusBlock>  *itr ){
            this->itr_data = itr;
            this->sample_num = -1; 
            this->sample_max = INT_MAX;
            this->rank_sample_method = 0;
            this->rank_sample_gap = 0.0001f;
            this->seed_sampler_bytime = 0;
            this->pos_sample_lowerb = 0.8f;
            this->neg_sample_upperb = 1e-6f;
            this->rank_sample_pointwise = 0;
        }
        ~PairwiseRankGenerator( void ){
            delete itr_data;
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "pos_sample_lowerb" )) pos_sample_lowerb = (float)atof( val );
            if( !strcmp( name, "neg_sample_upperb" )) neg_sample_upperb = (float)atof( val );
            if( !strcmp( name, "rank_sample_num" ))   sample_num = atoi( val );
            if( !strcmp( name, "rank_sample_max" ))   sample_max = atoi( val );
            if( !strcmp( name, "seed_sampler_bytime" )) seed_sampler_bytime = atoi( val );
            if( !strcmp( name, "rank_sample_method" ))  rank_sample_method = atoi( val );
            if( !strcmp( name, "rank_sample_gap" )) rank_sample_gap = (float)atof( val );
            if( !strcmp( name, "rank_sample_pointwise" ))  rank_sample_pointwise = atoi( val );
            itr_data->set_param( name, val );
        }
        virtual void init( void ){
            itr_data->init();
            if( this->seed_sampler_bytime != 0 ){
                apex_random::seed( static_cast<uint32_t>(time(NULL)) );
            }
            apex_utils::assert_true( rank_sample_gap > 0.0f, "must set rank_sample_gap to a value bigger than 0" ); 
        }
        virtual void before_first( void ){
            itr_data->before_first();
        }
        virtual bool next( SVDPlusBlock &e ){
            if( !itr_data->next( e ) ) return false;            
            row_label.resize( 0 ); 
            row_ptr.resize( 0 ); row_ptr.push_back( 0 );
            findex.resize( 0 ) ; fvalue.resize( 0 ); 
            switch( this->rank_sample_method ){
            case 0: this->sample_posneg( e ); break;
            case 1: this->sample_cmp( e );    break;
            default:apex_utils::error("unkown rank sample method\n");
            }            
            e.data.num_row   = static_cast<int>( row_label.size() );
            e.data.num_val   = static_cast<int>( findex.size() ); 
            e.data.row_ptr   = &row_ptr[0];
            if( e.data.num_row != 0 ){
                e.data.row_label = &row_label[0];
                e.data.feat_index= &findex[0];
                e.data.feat_value= &fvalue[0];
            }
            return true;
        }       
        virtual size_t get_data_size( void ){ 
            return itr_data->get_data_size();
        }        
    };
};

namespace apex_svd{
    // iterator that attaches addtional iterator to input
    class AttachBlockIterator: public IDataIterator<SVDPlusBlock>{
    private:
        IDataIterator<SVDPlusBlock> *itr_primary;
        IDataIterator<SVDPlusBlock> *itr_attached;
        int attach_skip, attach_insert;
        int skip_counter, insert_counter;
    public:
        AttachBlockIterator( IDataIterator<SVDPlusBlock> *ip, IDataIterator<SVDPlusBlock> *ia ){
            attach_insert = attach_skip = 1;
            this->itr_primary = ip;
            this->itr_attached= ia;
        }
        virtual ~AttachBlockIterator(){
            delete this->itr_primary;
            delete this->itr_attached;
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( "attach_skip", name   ) ) attach_skip = atoi( val );
            if( !strcmp( "attach_insert", name ) ) attach_insert = atoi( val );
            itr_primary->set_param( name, val );
            if( !strncmp( name, "attach:", 7 ) ){
                char sname[ 256 ];
                sscanf( name, "attach:%s", sname );
                itr_attached->set_param( sname, val );
            }
        }
        virtual void init( void ){
            itr_primary->init();
            itr_attached->init();
            insert_counter = 0;
            skip_counter   = attach_skip;
        } 
        virtual void before_first( void ){
            insert_counter = 0;
            skip_counter   = attach_skip;
            itr_primary->before_first();
        }
        virtual bool next( SVDPlusBlock &e ){
            e.extra_info  = 0;
            if( insert_counter != 0 ){
                if( !itr_attached->next( e ) ){
                    itr_attached->before_first();
                    itr_attached->next( e );
                }
                if( e.extend_tag == svdpp_tag::END_TAG || e.extend_tag == svdpp_tag::DEFAULT ){
                    insert_counter --;
                    if( insert_counter == 0 ) skip_counter = attach_skip;
                }
                // indicate e is a attach buffer
                e.extra_info  = 1;
                return true;
            }
            if( skip_counter != 0 ){
                if( !itr_primary->next( e ) ) return false;
                if( e.extend_tag == svdpp_tag::END_TAG || e.extend_tag == svdpp_tag::DEFAULT ){
                    skip_counter --;
                    if( skip_counter == 0 ) insert_counter = attach_insert;
                }
                return true;
            }
            apex_utils::error("this point can't be reached");
            return false;
        }        
        virtual size_t get_data_size( void ){ 
            return ( itr_primary->get_data_size() * ( attach_skip + attach_insert) + attach_skip-1 ) / attach_skip;
        }
    };
};

namespace apex_svd{
    // iterator that filters features
    class FilterBlockIterator: public IDataIterator<SVDPlusBlock>{
    private:
        struct Entry{
            unsigned start, end;
        };
        IDataIterator<SVDPlusBlock> *itr_base;
        // different parts of filter
        std::vector<Entry> ft_ufeedback, ft_global;
    private:
        inline void filter( unsigned *findex, 
                            float *fvalue, 
                            int len,
                            const std::vector<Entry> &ft ){
            for( int i = 0; i < len; i ++ ){
                const unsigned id =  findex[i];
                for( size_t j = 0; j < ft.size(); j ++ ){
                    if( id >= ft[j].start && id < ft[j].end ) fvalue[i] = 0.0f; 
                } 
            }
        }
    public:
        FilterBlockIterator( IDataIterator<SVDPlusBlock> *it ){
            this->itr_base = it;
        }
        virtual ~FilterBlockIterator(){
            delete this->itr_base;
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( "filter_ufeedback", name ) ){
                Entry e;
                apex_utils::assert_true( sscanf( val, "%u-%u", &e.start, &e.end ) == 2, "filter ufeedback" );
                ft_ufeedback.push_back( e );
            }
            if( !strcmp( "filter_global", name ) ){
                Entry e;
                apex_utils::assert_true( sscanf( val, "%u-%u", &e.start, &e.end ) == 2, "filter global" );
                ft_global.push_back( e );
            }
            itr_base->set_param( name, val );
        }
        virtual void init( void ){
            itr_base->init();
        } 
        virtual void before_first( void ){
            itr_base->before_first();
        }
        virtual bool next( SVDPlusBlock &e ){
            if( !itr_base->next( e ) ) return false;
            this->filter( e.index_ufeedback, e.value_ufeedback, e.num_ufeedback, ft_ufeedback );
            for( int i = 0; i < e.data.num_row; i ++ ){
                SVDFeatureCSR::Elem em = e.data[i];
                this->filter( em.index_global, em.value_global, em.num_global, ft_global );
            }
            return true;
        }        
        virtual size_t get_data_size( void ){ 
            return itr_base->get_data_size();
        }
    };
};

namespace apex_svd{
    // SVDFeatureCSR Page buffering factory
    struct SVDFeatureCSRPageFactory{
    private:
        bool next_ok; 
        SVDFeatureCSR::Elem e;
    public:
        // data provider
        IDataIterator<SVDFeatureCSR::Elem> *itr_data;
        SVDFeatureCSRPageFactory( void ){ itr_data = NULL; }
        inline void set_param( const char *name, const char *val ){
            if( itr_data != NULL ) itr_data->set_param( name, val );
        }
        
        inline size_t get_data_size() const{
            return itr_data->get_data_size();
        }

        inline bool init( int st ){ 
            itr_data->init(); 
            next_ok = itr_data->next( e );
            return true;
        }
        
        inline bool load_next( SVDFeatureCSRPage &val ){
            if( !next_ok ) return false;
            val.clear();
            while( next_ok && val.push_back( e ) ){
                next_ok = itr_data->next( e );
            }
            return true;
        }            

        inline SVDFeatureCSRPage create(){
            SVDFeatureCSRPage e;
            e.alloc_space();
            return e;
        }

        inline void free_space( SVDFeatureCSRPage &e ){        
            e.free_space();
        }                

        inline void destroy(){
            delete itr_data;
        }    
        
        inline void before_first(){
            itr_data->before_first();
            next_ok = itr_data->next( e );
        }
    };

    // SVDFeatureCSR Page file reading factory
    struct SVDFeatureCSRPageFileFactory{
    private:
        FILE *fi;
        int  nblock, idx;
        char name_buf[ 256 ];
    public:
        // data provider
        SVDFeatureCSRPageFileFactory( void ){
            strcpy( name_buf, "svdfeature_buf" );
            this->fi = NULL;
        }
        inline void set_param( const char *name, const char *val ){
            if( !strcmp( name, "buffer_feature" ) ) strcpy( name_buf  , val );
        }        
        inline size_t get_data_size() const{            
            return 0;
        }
        inline bool init( int st ){ 
            fi = apex_utils::fopen_check( name_buf, "rb" );
            fseek( fi, 0, SEEK_END );
            size_t sz = ftell( fi );
            apex_utils::assert_true( sz % (SVDFeatureCSRPage::psize*sizeof(int)) == 0, "file must have exact blocks" );
            nblock = sz / (SVDFeatureCSRPage::psize*sizeof(int));
            this->before_first();
            return true;
        }        
        inline bool load_next( SVDFeatureCSRPage &val ){
            if( idx >= nblock ) return false;
            val.load_from_file( fi );
            idx ++;
            return true;
        }            
        inline SVDFeatureCSRPage create(){
            SVDFeatureCSRPage e;
            e.alloc_space();
            return e;
        }
        inline void free_space( SVDFeatureCSRPage &e ){        
            e.free_space();
        }                
        inline void destroy(){
            fclose( fi );
        }            
        inline void before_first(){
            idx = 0;
            fseek( fi, 0, SEEK_SET );
        }
    };

    template<typename FactoryType>
    class SVDCSRPageThreadIterator: public IDataIterator<SVDFeatureCSR::Elem>{
    private:
        int idx; 
        SVDFeatureCSRPage dt;
    public :
        apex_utils::ThreadBufferIterator<SVDFeatureCSRPage,FactoryType> itr;
        SVDCSRPageThreadIterator(){
            idx = -1;
            itr.set_param( "buffer_size", "2" );
        }
        virtual ~SVDCSRPageThreadIterator(){
            itr.destroy();
        }
        virtual void set_param( const char *name, const char *val ){
            itr.set_param( name, val );
        }
        virtual void init( void ){
            itr.init();
        }
        virtual size_t get_data_size( void ){
            return static_cast<size_t>( itr.get_factory().get_data_size() );
        }
        virtual void before_first( void ){
            idx = -1; itr.before_first();
        }
        virtual bool next( SVDFeatureCSR::Elem &elem ){
            if( idx == -1 || idx == dt.num_row() ){
                if( !itr.next( dt ) ) return false;
                idx = 0;
            }
            elem = dt[ idx ++ ];
            return true;
        }
    };        
};


namespace apex_svd{
    IDataIterator<SVDFeatureCSR::Elem> *create_csr_iterator( int dtype ){
        switch( dtype ){
        case input_type::BINARY_BUFFER: return new SVDCSRThreadIterator<SVDFeatureCSRFactory>();
        case input_type::TEXT_FEATURE : return create_slavethread_iter( new SVDFeatureCSRLoader() );
        case input_type::TEXT_BASIC   : return create_slavethread_iter( new SVDBasicLoader() );
        case input_type::BINARY_PAGE  : return new SVDCSRPageThreadIterator<SVDFeatureCSRPageFileFactory>();
        default: apex_utils::error("unknown iterator type"); return NULL;
        }
    }
    IDataIterator<SVDPlusBlock> *create_plus_iterator( int dtype ){
        // create filter block iterator, for filtering parts of input
        if( dtype >= 200 && dtype < 300 ){
            return new FilterBlockIterator( create_plus_iterator( dtype % 100 ) );
        }
        // create attach block iterator, for composite input
        if( dtype >= 100 && dtype < 200 ){
            int dleft = ( dtype / 10 ) % 10;
            int dright= dtype % 10;
            return new AttachBlockIterator( create_plus_iterator( dleft ),
                                            create_plus_iterator( dright ) );
        }
        if( dtype >= 300 && dtype < 400 ){
            return new BufferIterator<SVDPlusBlock>( create_plus_iterator( dtype % 10 ) );
        }
        switch( dtype ){
        case input_type::BINARY_BUFFER: return 
                new IteratorAdapter< apex_utils::ThreadBufferIterator<SVDPlusBlock,SVDPlusBlockFactory>, 
                                     SVDPlusBlock>();
        case input_type::TEXT_FEATURE : return create_slavethread_iter( new SVDPlusBlockLoader() );
        case input_type::BINARY_BUFFER_RANK:
        case input_type::TEXT_FEATURE_RANK : 
            return new PairwiseRankGenerator( create_plus_iterator( dtype&1 ) );
        default: apex_utils::error("unknown iterator type"); return NULL;
        }
    }
};

namespace apex_svd{
    IDataIterator<SVDFeatureCSR::Elem> *create_slavethread_iter( IDataIterator<SVDFeatureCSR::Elem> *slave_iter ){
        //typedef IteratorAdapter< apex_utils::ThreadBufferIterator<SVDFeatureCSR::Elem,SVDFeatureCSRBuffer>, 
        //SVDFeatureCSR::Elem> itr_master;
        typedef SVDCSRPageThreadIterator<SVDFeatureCSRPageFactory> itr_master;
        itr_master *itr = new itr_master();
        ((itr->itr).factory).itr_data = slave_iter;
        return itr;
    }

    IDataIterator<SVDPlusBlock> *create_slavethread_iter( IDataIterator<SVDPlusBlock> *slave_iter ){
        typedef IteratorAdapter< apex_utils::ThreadBufferIterator<SVDPlusBlock,SVDPlusBlockBuffer>, 
                                 SVDPlusBlock> itr_master;
        itr_master *itr = new itr_master();
        ((itr->itr).factory).itr_data = slave_iter;
        return itr;
    }       
};

namespace apex_svd{
    void create_binary_buffer( const char *name_buf, IDataIterator<SVDFeatureCSR::Elem> *data_iter, int batch_size ){
        SVDFeatureCSRFactory::create_buffer( name_buf, data_iter, batch_size );
    } 
    void create_binary_buffer( const char *name_buf, IDataIterator<SVDPlusBlock> *data_iter ){
        SVDPlusBlockFactory::create_buffer( name_buf, data_iter );
    } 
};

namespace apex_svd{
    IDataIterator<SVDFeatureCSR::Elem> *create_csr_iterator( int dtype, const char *fname ){
        IDataIterator<SVDFeatureCSR::Elem> *itr;
        switch( dtype ){
        case input_type::BINARY_BUFFER: 
        case input_type::TEXT_FEATURE :
        case input_type::TEXT_BASIC   : 
        case input_type::BINARY_PAGE  : itr = create_csr_iterator( dtype ); break;
        default: apex_utils::error("unknown iterator type");
        }
        itr->set_param( "silent", "1" );
        switch( dtype ){
        case input_type::BINARY_PAGE  :
        case input_type::BINARY_BUFFER: itr->set_param( "buffer_feature", fname ); break;
        case input_type::TEXT_BASIC   :
        case input_type::TEXT_FEATURE : itr->set_param( "data_in", fname ); break;
        }
        itr->init();
        return itr;
    }
};
