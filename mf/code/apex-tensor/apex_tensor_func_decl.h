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
 * \file apex_tensor_func_decl.h
 * \brief declaration of common functions of CPU and GPU
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 * \sa apex_tensor_func_decl_common.h
 * this file will be expanded twice to expand the declaration of CPU and GPU function
 */

#ifdef _XD
#undef _XD
#endif
#ifdef _Tensor1D 
    #error "_Tensor1D must not be defined"
#endif
#ifdef _Tensor2D 
    #error "_Tensor2D must not be defined"
#endif
#ifdef _Tensor2D 
    #error "_Tensor2D must not be defined"
#endif
#ifdef _Tensor3D 
    #error "_Tensor3D must not be defined"
#endif
#ifdef _Tensor4D 
    #error "_Tensor4D must not be defined"
#endif
#ifdef _INLINE
    #error "_INLINE must not be defined"
#endif

/*! \brief  macro to specify real data type */
#define _Tensor1D apex_tensor::CTensor1D
/*! \brief  macro to specify real data type */
#define _Tensor2D apex_tensor::CTensor2D
/*!\brief  macro to specify real data type */
#define _Tensor3D apex_tensor::CTensor3D
/*! \brief  macro to specify real data type */
#define _Tensor4D apex_tensor::CTensor4D
/*! \brief  macro to specify whether the function is inlined */
#define _INLINE   inline

namespace apex_tensor{
    /*! \brief functions that only be implemented in cpu */
    namespace cpu_only{
        /*! \brief dot product of a and b */
        inline TENSOR_FLOAT dot( const CTensor1D &a, const CTensor1D &b );
    };
};

namespace apex_tensor{
    namespace tensor{    
        /*! 
         * \brief complement to BLAS ger function dst = dot( lhs.T, rhs )
         */        
        _INLINE void dot_lt( _Tensor2D &dst, const _Tensor1D &lhs, const _Tensor1D &rhs );

        /*! 
         * \brief BLAS ger function: dst += alpha*dot( lhs.T, rhs )
         */        
        _INLINE void ger( _Tensor2D &dst, const _Tensor1D &lhs, const _Tensor1D &rhs, TENSOR_FLOAT alpha );

        /*! 
         * \brief BLAS gxemv function: dst = dst*beta + dot( lhs, rhs[.T] )*alpha
         * \tparam transposeRight whether to transpose rhs
         */        
        template<bool transposeRight>
        _INLINE void gemv( _Tensor1D &dst, const _Tensor1D &lhs, const _Tensor2D &rhs, TENSOR_FLOAT alpha, TENSOR_FLOAT beta );

        /*! 
         * \brief BLAS gemm function: dst = dst*beta + dot( lhs[.T], rhs[.T] )*alpha
         * \tparam transposeLeft  whether to transpose lhs
         * \tparam transposeRight whether to transpose rhs
         */        
        template<bool transposeLeft,bool transposeRight>
        _INLINE void gemm( _Tensor2D &dst, const _Tensor2D &lhs, const _Tensor2D &rhs, TENSOR_FLOAT alpha, TENSOR_FLOAT beta );
        
        /*! 
         * \brief dst[i] += src
         */
        _INLINE void sadd( _Tensor2D &dst, const _Tensor1D &src );

        /*! 
         * \brief dst[i] = src
         */
        _INLINE void sfill( _Tensor2D &dst, const _Tensor1D &src );

        /*! 
         * \brief dst[y][x] /= src[y]
         */
        _INLINE void sdiv_y( _Tensor2D &dst, const _Tensor1D &src );

        /*! 
         * \brief dst[y][x] *= src[y]
         */
        _INLINE void smul_y( _Tensor2D &dst, const _Tensor1D &src );

        
        /*! 
         * \brief sum over rows
         * \tparam ST storage type apex_exp_template::enums::StoreMethod
         */
        template<typename ST>
        _INLINE void sum_row( _Tensor1D &dst, const _Tensor2D &src );

        /*! 
         * \brief sum over last two dimension 
         * \tparam ST storage type apex_exp_template::enums::StoreMethod
         */
        template<typename ST>
        _INLINE void sum_2D( _Tensor1D &dst, const _Tensor3D &src );

        /*! 
         * \brief sum over last two dimension 
         * \tparam ST storage type apex_exp_template::enums::StoreMethod
         */
        template<typename ST>
        _INLINE void sum_2D( _Tensor2D &dst, const _Tensor4D &src );
        
        /*! \brief normalize by softmax in dimension y */
        _INLINE void norm_softmax_y( _Tensor2D &mean, const _Tensor2D &energy );

        /*! \brief sample by softmax in dimension y */
        _INLINE void sample_softmax_y( _Tensor2D &state, const _Tensor2D &mean );
    };
};

/*------------------------------------------------^-^----------------------*
 * function declaration end here
 *-------------------------------------------------------------------------*/
/**
 * implement common functions 
 */
namespace apex_tensor{
    inline void _Tensor1D::copy_param( const CTensor1D &ts ){
        this->set_param( ts.x_max );
    }
    inline void _Tensor2D::copy_param( const CTensor2D &ts ){
        this->set_param( ts.y_max, ts.x_max );
    }
    inline void _Tensor3D::copy_param( const CTensor3D &ts ){
        this->set_param( ts.z_max, ts.y_max, ts.x_max );
    }
    inline void _Tensor4D::copy_param( const CTensor4D &ts ){
        this->set_param( ts.h_max, ts.z_max, ts.y_max, ts.x_max );
    }

    inline CTensor1D CTensor2D::operator[]( size_t idx ){
        CTensor1D ts( x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + idx*this->pitch_x);
        return ts;
    }
    inline const CTensor1D CTensor2D::operator[]( size_t idx ) const{
        CTensor1D ts( x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + idx*this->pitch_x);
        return ts;
    }
    inline CTensor2D CTensor3D::operator[]( size_t idx ){
        CTensor2D ts( y_max, x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + idx*this->pitch_xy);
        ts.pitch_x = this->pitch_x;
        return ts;
    }
    inline const CTensor2D CTensor3D::operator[]( size_t idx ) const{
        CTensor2D ts( y_max, x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + idx*this->pitch_xy);
        ts.pitch_x = this->pitch_x;
        return ts;
    }
    inline CTensor3D CTensor4D::operator[]( size_t idx ){
        CTensor3D ts( z_max, y_max, x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + idx*this->pitch_xyz);
        ts.pitch_x = this->pitch_x;
        ts.pitch_xy= this->pitch_xy;
        return ts;
    }
    inline const CTensor3D CTensor4D::operator[]( size_t idx ) const{
        CTensor3D ts( z_max, y_max, x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + idx*this->pitch_xyz);
        ts.pitch_x = this->pitch_x;
        ts.pitch_xy= this->pitch_xy;
        return ts;
    } 

    inline CTensor1D CTensor1D::sub_area( int x_start, int x_max ){
        CTensor1D ts( x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + x_start*sizeof(TENSOR_FLOAT));
        return ts;
    }

    inline const CTensor1D CTensor1D::sub_area( int x_start, int x_max ) const{
        CTensor1D ts( x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + x_start*sizeof(TENSOR_FLOAT));
        return ts;
    }

    inline CTensor2D CTensor2D::sub_area( int y_start, int x_start, int y_max, int x_max ){
        CTensor2D ts( y_max, x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + y_start*this->pitch_x + x_start*sizeof(TENSOR_FLOAT));
        ts.pitch_x  = this->pitch_x;
        return ts;
    }
    inline const CTensor2D CTensor2D::sub_area( int y_start, int x_start, int y_max, int x_max ) const{
        CTensor2D ts( y_max, x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + y_start*this->pitch_x+ x_start*sizeof(TENSOR_FLOAT));
        ts.pitch_x  = this->pitch_x;    
        return ts;
    }

    inline CTensor3D CTensor3D::slice_z( int z_start, int z_max ){
        CTensor3D ts( z_max, this->y_max, this->x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + z_start*this->pitch_xy);
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy;
        return ts;
    }

    inline const CTensor3D CTensor3D::slice_z( int z_start, int z_max ) const{
        CTensor3D ts( z_max, this->y_max, this->x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + z_start*this->pitch_xy);
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy;
        return ts;
    }

    inline CTensor3D CTensor3D::sub_area( int y_start, int x_start, int y_max, int x_max ){
        CTensor3D ts( z_max, y_max, x_max );
        TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + y_start*this->pitch_x+ x_start*sizeof(TENSOR_FLOAT));
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy;
        return ts;
    }
    inline const CTensor3D CTensor3D::sub_area( int y_start, int x_start, int y_max, int x_max ) const{
        CTensor3D ts( z_max, y_max, x_max );
        const TENSOR_FLOAT *ptr = this->elem; 
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + y_start*this->pitch_x+ x_start*sizeof(TENSOR_FLOAT));
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy;
        return ts;
    }

    inline CTensor4D CTensor4D::slice_z( int z_start, int z_max, int z_step ){
        CTensor4D ts( this->h_max, z_max, this->y_max, this->x_max );
        TENSOR_FLOAT *ptr= this->elem;
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + z_start*this->pitch_xy);
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy * z_step;
        ts.pitch_xyz= this->pitch_xyz;
        return ts;
    }

    inline const CTensor4D CTensor4D::slice_z( int z_start, int z_max, int z_step ) const{
        CTensor4D ts( this->h_max, z_max, this->y_max, this->x_max );
        const TENSOR_FLOAT *ptr= this->elem;
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + z_start*this->pitch_xy);
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy * z_step;
        ts.pitch_xyz= this->pitch_xyz;
        return ts;
    } 

    inline CTensor4D CTensor4D::slice_h( int h_start, int h_max, int h_step  ){
        CTensor4D ts( h_max, this->z_max, this->y_max, this->x_max );
        TENSOR_FLOAT *ptr= this->elem;
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + h_start*this->pitch_xyz);
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy;
        ts.pitch_xyz= this->pitch_xyz * h_step;
        return ts;
    }

    inline const CTensor4D CTensor4D::slice_h( int h_start, int h_max, int h_step ) const{
        CTensor4D ts( h_max, this->z_max, this->y_max, this->x_max );
        const TENSOR_FLOAT *ptr= this->elem;
        ts.elem = (TENSOR_FLOAT*)( ((char*)ptr) + h_start*this->pitch_xyz);
        ts.pitch_x  = this->pitch_x;
        ts.pitch_xy = this->pitch_xy;
        ts.pitch_xyz= this->pitch_xyz * h_step;
        return ts;
    } 
};

/** 
 * the following codes remaps expression template solver to functions define above
 */
namespace apex_exp_template{
    namespace solver_impl{        
        /*! \brief specialization of SumRowSolver */
        template<typename ST>
        struct SumRowSolver<ST,_Tensor1D,_Tensor2D>{
            /*! \brief implement dst = sum_row( src ) */
            static inline void eval( _Tensor1D &dst, const _Tensor2D &src  ){
                apex_tensor::tensor::sum_row<ST>( dst, src ); 
            }
        };
        /*! \brief specialization of Sum2DSolver */
        template<typename ST>
        struct Sum2DSolver<ST,_Tensor1D,_Tensor3D>{
            /*! \brief implement dst = sum_2D( src ) */
            static inline void eval( _Tensor1D &dst, const _Tensor3D &src  ){
                apex_tensor::tensor::sum_2D<ST>( dst, src ); 
            }
        };
        /*! \brief specialization of Sum2DSolver */
        template<typename ST>
        struct Sum2DSolver<ST,_Tensor2D,_Tensor4D>{
            /*! \brief implement dst = sum_2D( src ) */
            static inline void eval( _Tensor2D &dst, const _Tensor4D &src  ){
                apex_tensor::tensor::sum_2D<ST>( dst, src ); 
            }
        };
        
        /*! \brief specialization of DotSolver */
        template<>
        struct DotSolver<enums::SaveTo,_Tensor2D,_Tensor1D,_Tensor1D,true,false>{
            /*! \brief implement dst = dot( lhs.T, rhs ) */
            static inline void eval( _Tensor2D &dst, const _Tensor1D &lhs, const _Tensor1D &rhs ){
                apex_tensor::tensor::dot_lt( dst, lhs, rhs );
            }
        };        
        template<>
        struct DotSolver<enums::AddTo,_Tensor2D,_Tensor1D,_Tensor1D,true,false>{
            /*! \brief implement dst += dot( lhs.T, rhs ) */
            static inline void eval( _Tensor2D &dst, const _Tensor1D &lhs, const _Tensor1D &rhs ){
                apex_tensor::tensor::ger( dst, lhs, rhs, 1.0f );
            }
        };        
        template<>
        struct DotSolver<enums::SubTo,_Tensor2D,_Tensor1D,_Tensor1D,true,false>{
            /*! \brief implement dst -= dot( lhs.T, rhs ) */
            static inline void eval( _Tensor2D &dst, const _Tensor1D &lhs, const _Tensor1D &rhs ){
                apex_tensor::tensor::ger( dst, lhs, rhs, 1.0f );
            }
        };
        
        template<>
        struct ScaleDotSolver<enums::AddTo,_Tensor2D,_Tensor1D,_Tensor1D,true,false>{
            /*! \brief implement dst -= dot( lhs.T, rhs ) */
            static inline void eval( _Tensor2D &dst, const _Tensor1D &lhs, const _Tensor1D &rhs, double scale ){
                apex_tensor::tensor::ger( dst, lhs, rhs, (apex_tensor::TENSOR_FLOAT)scale );
            }
        };
        
        template<bool transposeRight>
        struct DotSolver<enums::SaveTo,_Tensor1D,_Tensor1D,_Tensor2D,false,transposeRight>{
            /*! \brief implement dst = dot( lhs, rhs[.T] ) */
            static inline void eval( _Tensor1D &dst, const _Tensor1D &lhs, const _Tensor2D &rhs ){
                apex_tensor::tensor::gemv<transposeRight>( dst, lhs, rhs, 1.0f, 0.0f );
            }
        };        
        template<bool transposeRight>
        struct DotSolver<enums::AddTo,_Tensor1D,_Tensor1D,_Tensor2D,false,transposeRight>{
            /*! \brief implement dst += dot( lhs, rhs[.T] ) */
            static inline void eval( _Tensor1D &dst, const _Tensor1D &lhs, const _Tensor2D &rhs ){
                apex_tensor::tensor::gemv<transposeRight>( dst, lhs, rhs, 1.0f, 1.0f );
            }
        };        
        template<bool transposeRight>
        struct DotSolver<enums::SubTo,_Tensor1D,_Tensor1D,_Tensor2D,false,transposeRight>{
            /*! \brief implement dst -= dot( lhs, rhs[.T] ) */
            static inline void eval( _Tensor1D &dst, const _Tensor1D &lhs, const _Tensor2D &rhs ){
                apex_tensor::tensor::gemv<transposeRight>( dst, lhs, rhs, -1.0f, 1.0f );
            }
        };        

        template<bool transposeLeft, bool transposeRight>
        struct DotSolver<enums::SaveTo,_Tensor2D,_Tensor2D,_Tensor2D,transposeLeft,transposeRight>{
            /*! \brief implement dst = dot( lhs[.T], rhs[.T] ) */
            static inline void eval( _Tensor2D &dst, const _Tensor2D &lhs, const _Tensor2D &rhs ){
                apex_tensor::tensor::gemm<transposeLeft,transposeRight>( dst, lhs, rhs, 1.0f, 0.0f );
            }
        };        
        template<bool transposeLeft, bool transposeRight>
        struct DotSolver<enums::AddTo,_Tensor2D,_Tensor2D,_Tensor2D,transposeLeft,transposeRight>{
            /*! \brief implement dst += dot( lhs[.T], rhs[.T] ) */
            static inline void eval( _Tensor2D &dst, const _Tensor2D &lhs, const _Tensor2D &rhs ){
                apex_tensor::tensor::gemm<transposeLeft,transposeRight>( dst, lhs, rhs, 1.0f, 1.0f );
            }
        };        
        template<bool transposeLeft, bool transposeRight>
        struct DotSolver<enums::SubTo,_Tensor2D,_Tensor2D,_Tensor2D,transposeLeft,transposeRight>{
            /*! \brief implement dst -= dot( lhs[.T], rhs[.T] ) */
            static inline void eval( _Tensor2D &dst, const _Tensor2D &lhs, const _Tensor2D &rhs ){
                apex_tensor::tensor::gemm<transposeLeft,transposeRight>( dst, lhs, rhs, -1.0f, 1.0f );
            }
        };        
    };
};

/*------^-^--------------------------------------------------------*
 *  the following codes repeat expand common functions declaration for 1D-4D
 *-----------------------------------------------------------------*/
/*!
 * \brief all kinds of tensors 
 *
 * if you see a function with argument type _TensorXD, then it means any tensor in {C|G}Tensor{1-4}D
 *
 * example: func( _TensorXD a, _CTensorXD b ); means 
 *  func( apex_tensor::CTensor1D a, apex_tensor::CTensor1D b );
 *  func( apex_tensor::GTensor1D a, apex_tensor::CTensor1D b );
 *  func( apex_tensor::CTensor2D a, apex_tensor::CTensor2D b );
 *  ...
 */
#define _TensorXD _Tensor1D
/*!
 * \brief all kinds of apex_tensor::CTensors 
 *
 * if you see a function with argument type  _CTensorXD, then it means any tensor in apex_tensor::CTensor{1-4}D
 */
#define _CTensorXD apex_tensor::CTensor1D
#include  "apex_tensor_func_decl_common.h"
#undef _TensorXD
#undef _CTensorXD

#define _TensorXD _Tensor2D
#define _CTensorXD apex_tensor::CTensor2D
#include  "apex_tensor_func_decl_common.h"
#undef _TensorXD
#undef _CTensorXD

#define _TensorXD _Tensor3D
#define _CTensorXD apex_tensor::CTensor3D
#include  "apex_tensor_func_decl_common.h"
#undef _TensorXD
#undef _CTensorXD

#define _TensorXD _Tensor4D
#define _CTensorXD apex_tensor::CTensor4D
#include  "apex_tensor_func_decl_common.h"
#undef _TensorXD
#undef _CTensorXD

#undef _Tensor1D 
#undef _Tensor2D 
#undef _Tensor3D 
#undef _Tensor4D 
#undef _INLINE   

