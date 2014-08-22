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

#ifndef _APEX_TENSOR_CPU_H_
#define _APEX_TENSOR_CPU_H_

/*!
 * \file apex_tensor_cpu.h
 * \brief CPU part implementation of tensor
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */

#include <cstdlib>
#include "apex_tensor.h"
#include "apex_exp_template.h"
/*! \brief namespace of tensor imlementation */
namespace apex_tensor{
    using namespace apex_exp_template::operators;
};
namespace apex_tensor{
    /*! 
     * \brief pointer in CPU 
     * \tparam TValue storage type of pointer
     */
    template<typename TValue>
    class CPtr{
    private:
        /*! \brief real pointer */
        TValue *ptr;
    public:
        CPtr(){}
        /*! \brief constructor */
        CPtr( TValue *p ):ptr( p ){} 
        /*! \brief convert to real pointer */
        inline operator TValue*(){ return ptr; }
        /*! \brief convert to const real pointer */
        inline operator const TValue*() const{ return ptr; }
    };

    /*! \brief 1D tensor in CPU */
    class CTensor1D: public apex_exp_template::ContainerExp<CTensor1D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor1D(){}
        /*! \brief constructor */
        CTensor1D( int x_max ){ 
            set_param( x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int x_max ){
            this->x_max = x_max;
        }
        /*! \brief number of elements in tensor */
        inline int num_elem()const{
            return x_max;
        }
        /*! \brief fill by TENSOR_FLOAT */
        inline CTensor1D & operator=( TENSOR_FLOAT s );
        /*! \brief overload operator= to enable expression template */
        template<typename T>
        inline CTensor1D & operator=( const apex_exp_template::CompositeExp<T> &exp ){
            return __assign( exp.__name_const() );
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor1D &exp );

        /*! \brief operator[] */
        inline TENSOR_FLOAT& operator[]( size_t idx ){
            return elem[idx];
        }        
        /*! \brief operator[] */
        inline const TENSOR_FLOAT& operator[]( size_t idx )const{
            return elem[idx];
        }    
        /*! \brief return a sub area start from (x_start) with range(x_max) */
        inline CTensor1D sub_area( int x_start, int x_max );
        /*! \brief return a sub area start from (x_start) with range(x_max) */
        inline const CTensor1D sub_area( int x_start, int x_max ) const;
    };

    /*! \brief 2D tensor in CPU */
    class CTensor2D: public apex_exp_template::ContainerExp<CTensor2D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch_x;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor2D(){}
        /*! \brief constructor */
        CTensor2D( int y_max, int x_max ){ 
            set_param( y_max, x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int y_max,int x_max ){
            this->y_max = y_max;
            this->x_max = x_max;
        }
        /*! \brief number of elements in tensor */
        inline int num_elem()const{
            return y_max*x_max;
        }
        /*! \brief fill by TENSOR_FLOAT */
        inline CTensor2D & operator=( TENSOR_FLOAT s );
        /*! \brief overload operator= to enable expression template */
        template<typename T>
        inline CTensor2D & operator=( const apex_exp_template::CompositeExp<T> &exp ){
            return __assign( exp.__name_const() );
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor2D &exp );
        /*! \brief operator[] */
        inline CTensor1D operator[]( size_t idx );
        /*! \brief operator[] */
        inline const CTensor1D operator[]( size_t idx ) const;
        /*! \brief return a sub area start from (y_start,x_start) with range(y_max, x_max) */
        inline CTensor2D sub_area( int y_start, int x_start, int y_max, int x_max );
        /*! \brief return a sub area start from (y_start,x_start) with range(y_max, x_max) */
        inline const CTensor2D sub_area( int y_start, int x_start, int y_max, int x_max ) const;
    };

    /*! \brief 3D tensor in CPU */
    class CTensor3D: public apex_exp_template::ContainerExp<CTensor3D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of element in z dimension */
        int z_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch_x;
        /*! \brief number of bytes allocated in xy dimension */
        unsigned int pitch_xy;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor3D(){}
        /*! \brief constructor */
        CTensor3D( int z_max, int y_max, int x_max ){ 
            set_param( z_max, y_max, x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int z_max, int y_max,int x_max, bool reset_pitch = false ){
            this->z_max = z_max;
            this->y_max = y_max;
            this->x_max = x_max;
            if( reset_pitch ) this->pitch_xy = this->pitch_x * y_max; 
        }
        /*! \brief number of elements in tensor */
        inline int num_elem()const{
            return z_max*y_max*x_max;
        }
        inline CTensor3D & operator=( TENSOR_FLOAT s );
        /*! \brief overload operator= to enable expression template */
        template<typename T>
        inline CTensor3D & operator=( const apex_exp_template::CompositeExp<T> &exp ){
            return __assign( exp.__name_const() );
        }       
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor3D &exp );
        /*! \brief operator[] */
        inline CTensor2D operator[]( size_t idx );
        /*! \brief operator[] */
        inline const CTensor2D operator[]( size_t idx ) const;       
        /*! \brief return a sliced area start from z_start, with range z_max */
        inline CTensor3D slice_z( int z_start, int z_max );
        /*! \brief return a sliced area start from z_start, with range z_max */
        inline const CTensor3D slice_z( int z_start, int z_max ) const;
        /*! \brief return a sub area start from (y_start,x_start) with range(y_max, x_max) */
        inline CTensor3D sub_area( int y_start, int x_start, int y_max, int x_max );
        /*! \brief return a sub area start from (y_start,x_start) with range(y_max, x_max) */
        inline const CTensor3D sub_area( int y_start, int x_start, int y_max, int x_max ) const;
    };
    /*! \brief 4D tensor in CPU */
    class CTensor4D: public apex_exp_template::ContainerExp<CTensor4D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of element in z dimension */
        int z_max;
        /*! \brief number of element in h dimension */
        int h_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch_x;
        /*! \brief number of bytes allocated in xy dimension */
        unsigned int pitch_xy;
        /*! \brief number of bytes allocated in xyz dimension */
        unsigned int pitch_xyz;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor4D(){}
        /*! \brief constructor */
        CTensor4D( int h_max, int z_max, int y_max, int x_max ){ 
            set_param( h_max, z_max, y_max, x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int h_max, int z_max, int y_max, int x_max, bool reset_pitch=false ){
            this->h_max = h_max;
            this->z_max = z_max;
            this->y_max = y_max;
            this->x_max = x_max;
            if( reset_pitch ){
                this->pitch_xy  = this->z_max * this->pitch_x;
                this->pitch_xyz = this->h_max * this->pitch_xy;
            }
        }
        /*! \brief number of elements in tensor */
        inline int num_elem()const{
            return h_max*z_max*y_max*x_max;
        }
        inline CTensor4D & operator=( TENSOR_FLOAT s );
        /*! \brief overload operator= to enable expression template */
        template<typename T>
        inline CTensor4D & operator=( const apex_exp_template::CompositeExp<T> &exp ){
            return __assign( exp.__name_const() );
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor4D &exp );
        /*! \brief operator[] */
        inline CTensor3D operator[]( size_t idx );
        /*! \brief operator[] */
        inline const CTensor3D operator[]( size_t idx ) const;
        /*! \brief step slice z by step len, maximum number z_max */
        inline CTensor4D slice_z( int z_start, int z_max, int z_step = 1 );
        /*! \brief step slice z by step len, maximum number z_max */
        inline const CTensor4D slice_z( int z_start, int z_max, int z_step = 1 ) const;
        /*! \brief step slice h by step len, maximum number z_max */
        inline CTensor4D slice_h( int h_start, int h_max, int h_step = 1 );
        /*! \brief step slice h by step len, maximum number z_max */
        inline const CTensor4D slice_h( int h_start, int h_max, int h_step = 1 ) const;
    };
};
#endif

