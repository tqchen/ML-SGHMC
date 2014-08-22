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
 * \file apex_svd_data.h
 * \brief this file contains the input data structure for SVDFeature and interface for data loading
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */
#ifndef _APEX_SVD_DATA_H_
#define _APEX_SVD_DATA_H_
#include <vector>
#include <cstring>
#include "apex-utils/apex_utils.h"

namespace apex_svd{
    /*! 
     * \brief storage block of random order input, 
     *  in CSR format for sparse matrix, multiple lines are stored together in a block
     * \sa SVDFeatureCSR::Elem
     */
    struct SVDFeatureCSR{
        /*! 
         * \brief storage block of random order input, a single line of feature
         */
        struct Elem{
            /*! \brief result label, rate or {0,1} for classification */
            float  label;
            /*! \brief number of nonzero global feature */
            int num_global;
            /*! \brief number of nonzero user feature */
            int num_ufactor;
            /*! \brief number of nonzero item feature */
            int num_ifactor;
            /*! \brief array of global feature index */
            unsigned *index_global;
            /*! \brief array of user feature index */
            unsigned *index_ufactor;
            /*! \brief array of item feature index */
            unsigned *index_ifactor;
            /*! \brief array of global feature value */
            float *value_global;
            /*! \brief array of user feature value */
            float *value_ufactor;
            /*! \brief array of item feature value */
            float *value_ifactor;
            /*! 
             * \brief get total number of nonzero features
             * \return total number of nonzero features
             */
            inline int total_num( void ) const{
                return num_global + num_ufactor + num_ifactor;
            }
            /*! 
             * \brief set the data space of feature
             * \param index pointer to the provided index space
             * \param value pointer to the provided value space
             */
            inline void set_space( unsigned *index, float *value ){
                index_global = index;
                value_global = value;
                index_ufactor= index_global  + num_global;
                index_ifactor= index_ufactor + num_ufactor;
                value_ufactor= value_global  + num_global;
                value_ifactor= value_ufactor + num_ufactor;
            }
            /*! 
             * \brief allocate space for the given setting:num_global,num_user,num_item must be set first
             */
            inline void alloc_space( void ){
                this->set_space( new unsigned[ total_num() ], new float[ total_num() ] );
            }
            /*! 
             * \brief free the space for index and value
             */
            inline void free_space( void ){
                delete [] index_global;
                delete [] value_global;
            }
            /*! 
             * \brief clone another copy of data
             * \return the cloned element
             */
            inline Elem clone( void ) const{
                Elem val = *this;
                val.alloc_space();
                memcpy( val.index_global , index_global, sizeof(unsigned)*num_global );
                memcpy( val.value_global , value_global, sizeof(float)*num_global );
                memcpy( val.index_ufactor, index_ufactor, sizeof(unsigned)*num_ufactor );
                memcpy( val.value_ufactor, value_ufactor, sizeof(float)*num_ufactor );
                memcpy( val.index_ifactor, index_ifactor, sizeof(unsigned)*num_ifactor );
                memcpy( val.value_ifactor, value_ifactor, sizeof(float)*num_ifactor );
                return val;
            }
        };        
        // note: features is stored in order, global, ufactor, ifactor 
        /*! \brief number of rows in the sparse matrix */
        int num_row;
        /*! \brief number of nonzero entries in sparse matrix */
        int num_val;
        /*! \brief label of each row */
        float *row_label;
        /*! 
         * \brief points to the beginning and ends of each row in the data space, size(row_ptr)=3*num_row+1,
         *   because we need to points the beginning and ends of global feature, user feature and item feature
         */
        int   *row_ptr;        
        /*! \brief pointer to the index space */
        unsigned   *feat_index;
        /*! \brief pointer to the value space */
        float      *feat_value;
        /*! 
         * \brief get r'th row in the matrix 
         * \param r row number 
         * \return r'th row
         */
        inline Elem operator[]( int r ){
            Elem e;
            e.label = row_label[ r ];
            e.num_global   = row_ptr[ r * 3 + 1 ] - row_ptr[ r * 3 + 0 ];
            e.num_ufactor  = row_ptr[ r * 3 + 2 ] - row_ptr[ r * 3 + 1 ];
            e.num_ifactor  = row_ptr[ r * 3 + 3 ] - row_ptr[ r * 3 + 2 ];            
            e.index_global  = feat_index + row_ptr[ r * 3 + 0 ];
            e.index_ufactor = feat_index + row_ptr[ r * 3 + 1 ];
            e.index_ifactor = feat_index + row_ptr[ r * 3 + 2 ];            
            e.value_global  = feat_value + row_ptr[ r * 3 + 0 ]; 
            e.value_ufactor = feat_value + row_ptr[ r * 3 + 1 ]; 
            e.value_ifactor = feat_value + row_ptr[ r * 3 + 2 ];
            return e;
        }
        /*! 
         * \brief get r'th row in the matrix 
         * \param r row number 
         * \return r'th row
         */
        inline const Elem operator[]( int r ) const{
            Elem e;
            e.label = row_label[ r ];
            e.num_global   = row_ptr[ r * 3 + 1 ] - row_ptr[ r * 3 + 0 ];
            e.num_ufactor  = row_ptr[ r * 3 + 2 ] - row_ptr[ r * 3 + 1 ];
            e.num_ifactor  = row_ptr[ r * 3 + 3 ] - row_ptr[ r * 3 + 2 ];            
            e.index_global  = feat_index + row_ptr[ r * 3 + 0 ];
            e.index_ufactor = feat_index + row_ptr[ r * 3 + 1 ];
            e.index_ifactor = feat_index + row_ptr[ r * 3 + 2 ];            
            e.value_global  = feat_value + row_ptr[ r * 3 + 0 ]; 
            e.value_ufactor = feat_value + row_ptr[ r * 3 + 1 ]; 
            e.value_ifactor = feat_value + row_ptr[ r * 3 + 2 ];
            return e; 
        }
        /*! 
         * \brief get a submatrix, with same number of column, but a subrange of rows
         * \param r_start starting row number
         * \param num_row number of rows of the result submatrix
         * \return the submatrix
         */
        inline SVDFeatureCSR slice_rows( int r_start, int num_row ) const{
            SVDFeatureCSR sp;
            sp.num_row = num_row;
            sp.num_val = row_ptr[ r_start*3 + num_row*3 ] - row_ptr[ r_start*3 ];
            sp.row_label = this->row_label + r_start;
            sp.row_ptr   = this->row_ptr + r_start * 3;
            sp.feat_index = this->feat_index;
            sp.feat_value = this->feat_value;
            return sp;
        }
        /*! 
         * \brief allocate space for given parameters: num_row, num_val must be set before allocation
         */
        inline void alloc_space( void ){
            row_ptr    = new int[ num_row*3 + 1 ];
            row_label  = new float[ num_row ];
            feat_index = new unsigned[ num_val ];
            feat_value = new float   [ num_val ];
        }
        /*! 
         * \brief free space of the CSR matrix
         */        
        inline void free_space( void ){
            delete [] row_ptr;
            delete [] row_label;
            delete [] feat_index;
            delete [] feat_value;
        }
        /*! 
         * \brief save the data to binary file
         * \param fo pointer to file
         */
        inline void save_to_file( FILE *fo ) const{
            const int start = row_ptr[ 0 ];
            std::vector<int> tmp_ptr;
            for( int i = 0; i < num_row*3 + 1; i ++ )
                tmp_ptr.push_back( row_ptr[ i ] - start );
            
            fwrite( this, sizeof(int), 2, fo );            
            fwrite( &tmp_ptr[0], sizeof(int), num_row*3 + 1, fo );
            if( num_row > 0 ){
                fwrite( row_label, sizeof(float), num_row, fo );
            }
            if( num_val > 0 ){
                fwrite( feat_index + start, sizeof(unsigned)  , num_val, fo );
                fwrite( feat_value + start, sizeof(float), num_val, fo );
            }
        }
        /*! 
         * \brief load data from file
         * \param fi pointer to file
         */        
        inline void load_from_file( FILE *fi ) {
            apex_utils::assert_true( fread( this, sizeof(int), 2, fi ) > 0, "CSR load from file");
            apex_utils::assert_true( fread( row_ptr, sizeof(int), num_row*3+1, fi ) > 0, "CSR load from file" );
            if( num_row > 0 ){
                apex_utils::assert_true( fread( row_label, sizeof(float), num_row, fi ) > 0, "CSR load from file" );
            }
            if( num_val > 0 ){
                apex_utils::assert_true( fread( feat_index, sizeof(unsigned), num_val, fi ) > 0 , "CSR load from file");
                apex_utils::assert_true( fread( feat_value, sizeof(float), num_val, fi ) > 0, "CSR load from file");
            }           
        }
    };
};

namespace apex_svd{
    /*!
     * \brief fix size page data structure for storage page for CSR data
     *  can be used to store SVDFeatureCSR::Elem in memory
     */
    class SVDFeatureCSRPage{
    public:
        /*!\brief constant for pagesize */
        static const int psize = 1 << 20;
    private:
        /*!\brief pointer to the space */
        int *dptr;        
    public:
        /*!\brief constructor */
        SVDFeatureCSRPage( void ){
            dptr = NULL;
            apex_utils::assert_true( sizeof(float) == sizeof(unsigned)
                                     && sizeof(unsigned) == sizeof(int), 
                                     "sizeof float must meets sizeof int for the page to work " );
        }
        /*!
         * \brief load data from binary file
         * \param fi input file
         */
        inline void load_from_file( FILE *fi ){
            apex_utils::assert_true( fread( dptr, sizeof(int), psize, fi ) > 0, "load CSR page" );
        }
        /*!
         * \brief save data to binary file
         * \param fo onput file
         */
        inline void save_to_file( FILE *fo ){
            fwrite( dptr, sizeof(int), psize, fo );
        }
        /*!\brief allocate space for the data */
        inline void alloc_space( void ){
            if( dptr == NULL ){
                dptr = new int[ psize ];
                // num_row
                dptr[ 0 ] = dptr[ 1 ] = 0;
            }
        }
        /*!\brief free space for the data */
        inline void free_space( void ){
            if( dptr != NULL ) delete []dptr;
        }
        /*!
         * \brief add data to the page, if full, return false
         * \return whether the data is succesfully inserted
         */
        inline bool push_back( const SVDFeatureCSR::Elem &e ){
            const int nrow = dptr[ 0 ];
            const int space_head = (nrow << 2) + 1;
            const int nval = dptr[ space_head ];
            const int n = e.total_num();
            const int space_cost = space_head + 5 + ((n+nval)<<1);
            if( space_cost > psize ) return false;
            {// set information in head
                int *p = dptr + space_head;
                *((float*)(p+1)) = e.label;
                p[ 2 ] = p[ 0 ] + e.num_global; 
                p[ 3 ] = p[ 2 ] + e.num_ufactor;
                p[ 4 ] = p[ 3 ] + e.num_ifactor;
                dptr[ 0 ] ++;
            }
            {// set data in tail
                unsigned   *pidx = (unsigned*)(dptr + psize - ((nval + n)<<1));
                float      *pval = (float*)(pidx + n);
                for( int i = 0; i < e.num_global; i ++, pidx ++, pval ++ ){
                    pidx[ 0 ] = e.index_global[ i ];
                    pval[ 0 ] = e.value_global[ i ];
                } 
                for( int i = 0; i < e.num_ufactor; i ++, pidx ++, pval ++ ){
                    pidx[ 0 ] = e.index_ufactor[ i ];
                    pval[ 0 ] = e.value_ufactor[ i ];
                } 
                for( int i = 0; i < e.num_ifactor; i ++, pidx ++, pval ++ ){
                    pidx[ 0 ] = e.index_ifactor[ i ];
                    pval[ 0 ] = e.value_ifactor[ i ];
                } 
            }
            return true;
        }
        /*!\brief set number of row to 0 */
        inline void clear( void ){
            dptr[ 0 ] = dptr[ 1 ] = 0;
        }
        /*!
         * \brief get number of row in the data 
         * \return number of row
         */
        inline int num_row( void ) const{
            return dptr[ 0 ];
        }
        /*! 
         * \brief get r'th row in the matrix 
         * \param r row number 
         * \return r'th row
         */
        inline const SVDFeatureCSR::Elem operator[]( int r ) const{
            SVDFeatureCSR::Elem e;
            int *p = dptr + (r << 2) + 1; 
            e.label = *((float*)(p+1));                
            e.num_global   = p[ 2 ] - p[ 0 ];
            e.num_ufactor  = p[ 3 ] - p[ 2 ];
            e.num_ifactor  = p[ 4 ] - p[ 3 ];
            unsigned *pidx = (unsigned*)(dptr + psize - (p[4]<<1));
            float    *pval = (float*)( pidx + p[ 4 ] - p[ 0 ] );
            e.set_space( pidx, pval );
            return e; 
        }
    };

    /*! 
     * \brief namespace for extension tag in SVDPlusBlock,
     *  used to store information when we split data of a user into several consecutive
     *  blocks when a user has too much data
     *  that's too large to be fit into one block
     */    
    namespace svdpp_tag{
        /*! 
         * \brief indicate this block is a default block, all the user's features are in this block, 
         * when no large-scale data is encountered, default should be fine
         */
        const int DEFAULT    = 0;
        /*! 
         * \brief the data is split into several blocks, indicate current block is the first one
         */
        const int START_TAG  = 1;
        /*! 
         * \brief the data is split into several blocks, indicate current block is the last one
         */
        const int END_TAG    = 2;
        /*! 
         * \brief the data is split into several blocks, indicate current block is neither the first or last one
         */
        const int MIDDLE_TAG = 3;
    };
    /*! 
     * \brief storage block of user grouped input
     *  used for efficient SVD++ training and rank-model
     */
    struct SVDPlusBlock{
        /*! \brief number of user feedback information */
        int num_ufeedback;
        /*! 
         * \brief extension tag of the block
         * \sa svdpp_tag
         */
        int extend_tag;
        /*! 
         * \brief extra information hold by the block
         * \sa svdpp_tag
         */        
        int extra_info;
        /*! \brief feature index for user feedback information */
        unsigned *index_ufeedback;
        /*! \brief feature value for user feedback information */
        float    *value_ufeedback;
        /*! \brief features in the user group,these feature share the same implict/explict feedback information */
        SVDFeatureCSR data;
        /*! \brief constructor */
        SVDPlusBlock(){ extend_tag = svdpp_tag::DEFAULT; extra_info = 0; }
        /*! 
         * \brief allocate space with given parameters: num_ufeedback, data.num_val, data.num_elem
         *  will allocate index_ufeedback, value_ufeedback, data.row_ptr, data.row_label, data.feat_index, data.feat_value
         */
        inline void alloc_space( void ){
            index_ufeedback = new unsigned[ num_ufeedback ];
            value_ufeedback = new float   [ num_ufeedback ];
            data.alloc_space();
        }
        /*! 
         * \brief free space of the storge
         * \sa alloc_space
         */
        inline void free_space( void ){
            data.free_space();
            delete [] index_ufeedback;
            delete [] value_ufeedback;
        }
        /*! 
         * \brief save current storage to file
         * \param fo pointer to file
         */        
        inline void save_to_file( FILE *fo ) const{
            // store extend_tag if it's not default, use negative num_ufeedback to mark
            if( extend_tag != svdpp_tag::DEFAULT ){
                int nu = num_ufeedback | ( 1 << 31 );
                fwrite( &nu , sizeof(int), 1, fo );
                fwrite( &extend_tag , sizeof(int), 1, fo );
            }else{
                fwrite( &num_ufeedback , sizeof(int), 1, fo );
            }
            fwrite( index_ufeedback, sizeof(unsigned), num_ufeedback, fo );
            fwrite( value_ufeedback, sizeof(float)   , num_ufeedback, fo );
            data.save_to_file( fo );
        }
        /*! 
         * \brief load data from file, sufficient space needs to be allocated before loading
         * \param fi pointer to file
         */        
        inline void load_from_file( FILE *fi ) {
            apex_utils::assert_true( fread( &num_ufeedback , sizeof(int), 1, fi ) > 0, "can't load SVD++ block" );
            // use negative num_ufeedback to mark non-default extend_tag, load if necessary
            if( num_ufeedback < 0 ){
                num_ufeedback = num_ufeedback & (~(1<<31));
                apex_utils::assert_true( fread( &extend_tag , sizeof(int), 1, fi ) > 0, "can't load SVD++ block" );
            }else{
                this->extend_tag = svdpp_tag::DEFAULT;
            }
            if( num_ufeedback > 0 ){ 
                apex_utils::assert_true( fread( index_ufeedback, sizeof(unsigned), num_ufeedback, fi ) > 0, "can't load SVD++ block" );
                apex_utils::assert_true( fread( value_ufeedback, sizeof(float)   , num_ufeedback, fi ) > 0, "can't load SVD++ block" );
            }
            data.load_from_file( fi );
        }
        /*! 
         * \brief clone another copy of the storage block
         * \return a cloned copy of the storage block
         */        
        inline SVDPlusBlock clone( void ) const{
            SVDPlusBlock val = *this;
            val.alloc_space();
            memcpy( val.index_ufeedback, index_ufeedback, sizeof(unsigned)*num_ufeedback );
            memcpy( val.value_ufeedback, value_ufeedback, sizeof(float)*num_ufeedback );
            memcpy( val.data.row_ptr   , data.row_ptr  , sizeof(int)*(data.num_row*3+1) );
            memcpy( val.data.row_label , data.row_label, sizeof(float)*data.num_row );
            memcpy( val.data.feat_index, data.feat_index, sizeof(unsigned)*data.num_val );
            memcpy( val.data.feat_value, data.feat_value, sizeof(float)*data.num_val );
            return val;
        }
    };    
    /*! 
     * \brief interface for data iterator
     * \tparam DType data type of the iterator
     */
    template<typename DType>
    class IDataIterator{
    public:
        /*! 
         * \brief set parameters, 
         *  used to pass parameter to iterator before initialization
         * \param name name of the parameter
         * \param val  value of the parameter
         */
        virtual void set_param( const char *name, const char *val ) = 0;
        /*! 
         * \brief initialize the iterator before use
         */
        virtual void init( void ) = 0; 
        /*! 
         * \brief set iterator from begining
         */
        virtual void before_first( void ) = 0;
        /*! 
         * \brief get next element
         * \param e the output element
         * \return whether the iterator reaches at end 
         */
        virtual bool next( DType &e ) = 0;        
        /*! 
         * \brief get number of elements in data set
         * used for progress estimation, optinal
         * \return the size of data in the iterator
         */
        virtual size_t get_data_size( void ){ return 0; }
        /*! \brief destructor */
        virtual ~IDataIterator( void ){}
    }; 
};

namespace apex_svd{ 
    /*! 
     * \brief namespace defining the input iterator types, we can add more when doing extension
     */    
    namespace input_type{
        /*! \brief binary file type, commonly used in training */
        const int BINARY_BUFFER = 0;
        /*! \brief text file type, specified by manual */
        const int TEXT_FEATURE  = 1;
        /*! \brief buffer file type, rank-pair generation by toolkit */
        const int BINARY_BUFFER_RANK = 2;
        /*! \brief text file type  , rank-pair generation by toolkit */
        const int TEXT_FEATURE_RANK  = 3;
        /*! \brief text file type  , basic three column format for SVD */
        const int TEXT_BASIC  = 4;
        /*! \brief binary page type, stored as consecutive SVDFeatureCSRPage */
        const int BINARY_PAGE = 5;
    };
    /*! 
     * \brief create a iterator for random order input
     * \return a iterator uninitialized
     */    
    IDataIterator<SVDFeatureCSR::Elem> *create_csr_iterator ( int dtype );
    /*! 
     * \brief create a iterator for random order input
     * \param dtype type of iterator
     * \fname name of file to open
     * \return iterator created and initialized
     */        
    IDataIterator<SVDFeatureCSR::Elem> *create_csr_iterator( int dtype, const char *fname );
    /*! 
     * \brief create a iterator for user grouped input, used in efficient SVD++ training and rank
     */    
    IDataIterator<SVDPlusBlock>  *create_plus_iterator( int dtype );
    /*! 
     * \brief create a iterator that use a slavethread to load the input, 
     *     this function is used to create pipeline style-code, user can provide
     *     a simple iterator and pass it to the function to get a slavethread iterator
     * \param slave_iter the iterator that provides the input
     * \return a iterator that use a slave thread to load data from slave_iter
     */
    IDataIterator<SVDFeatureCSR::Elem> *create_slavethread_iter( IDataIterator<SVDFeatureCSR::Elem> *slave_iter );
    /*! 
     * \brief create a iterator that use a slavethread to load the input, 
     *     this function is used to create pipeline style-code, user can provide
     *     a simple iterator and pass it to the function to get a slavethread iterator
     * \param slave_iter the iterator that provides the input
     * \return a iterator that use a slave thread to load data from slave_iter
     */
    IDataIterator<SVDPlusBlock>  *create_slavethread_iter( IDataIterator<SVDPlusBlock> *slave_iter );    
    /*! 
     * \brief create random order format binary buffer file with the data provided by data_iter
     * \param name_buf name of the binary buffer file, 
     * \param data_iter data iterator that provide the data input
     * \param batch_size how many elements to be stored in a block, 
     *    not a too important parameter, simply use the default value
     */
    void create_binary_buffer( const char *name_buf, IDataIterator<SVDFeatureCSR::Elem> *data_iter, int batch_size = 1000 );
    /*! 
     * \brief create user grouped format binary buffer file with the data provided by data_iter
     * \param name_buf name of the binary buffer file
     * \param data_iter data iterator that provide the data input
     */
    void create_binary_buffer( const char *name_buf, IDataIterator<SVDPlusBlock> *data_iter );
};
#endif
