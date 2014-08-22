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

#ifndef _APEX_TENSOR_CPU_INLINE_H_
#define _APEX_TENSOR_CPU_INLINE_H_

#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "apex_random.h"

#if __APEX_TENSOR_USE_SSE__
#include "apex_tensor_sse.h"
#endif

// CPU inline implementation of the functions 
// in CTensor 
// \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
// this file will not be documented 

namespace apex_tensor{
    /*! \brief namespace of all support functions in CPU implementation */
    namespace cpu_template{
        inline void assert_true( bool exp, const char *msg ){
            if( !exp ){
                printf("error:%s\n", msg ); exit( -1 );
            }
        }
        template<typename ST>
        struct Store{
            static inline void store( TENSOR_FLOAT &dst, TENSOR_FLOAT src );
        };
        
        template<>
        struct Store<apex_exp_template::enums::SaveTo>{
            static inline void store( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                dst = src;
            }
        };
        template<>
        struct Store<apex_exp_template::enums::AddTo>{
            static inline void store( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                dst += src;
            }
        };
        template<>
        struct Store<apex_exp_template::enums::SubTo>{
            static inline void store( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                dst -= src;
            }
        };
        template<>
        struct Store<apex_exp_template::enums::MulTo>{
            static inline void store( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                dst *= src;
            }
        };
        template<>
        struct Store<apex_exp_template::enums::DivTo>{
            static inline void store( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                dst /= src;
            }
        };
        // binary map
        template<typename OP>
        struct BinaryMap{
            static inline TENSOR_FLOAT map( TENSOR_FLOAT lhs, TENSOR_FLOAT rhs );
        };
        template<>
        struct BinaryMap<apex_exp_template::enums::Add>{
            static inline TENSOR_FLOAT map( TENSOR_FLOAT lhs, TENSOR_FLOAT rhs ){
                return lhs + rhs;
            }
        };
        template<>
        struct BinaryMap<apex_exp_template::enums::Sub>{
            static inline TENSOR_FLOAT map( TENSOR_FLOAT lhs, TENSOR_FLOAT rhs ){
                return lhs - rhs;
            }
        };
        template<>
        struct BinaryMap<apex_exp_template::enums::Mul>{
            static inline TENSOR_FLOAT map( TENSOR_FLOAT lhs, TENSOR_FLOAT rhs ){
                return lhs * rhs;
            }
        };
        template<>
        struct BinaryMap<apex_exp_template::enums::Div>{
            static inline TENSOR_FLOAT map( TENSOR_FLOAT lhs, TENSOR_FLOAT rhs ){
                return lhs / rhs;
            }
        };            
    };    
};

#if __APEX_TENSOR_USE_BLAS__
#include "cblas.h"
#endif

namespace apex_tensor{
    namespace tensor{
        // alloc space 
        inline void alloc_space( CTensor1D &ts ){
#if __APEX_TENSOR_USE_SSE__
            ts.elem = (TENSOR_FLOAT*)apex_sse2::aligned_malloc( ts.x_max*sizeof(TENSOR_FLOAT) );
#else
            ts.elem = new TENSOR_FLOAT[ ts.x_max ];
#endif
        }
        inline void alloc_space( CTensor2D &ts ){
#if __APEX_TENSOR_USE_SSE__
            size_t pitch;
            ts.elem = (TENSOR_FLOAT*)apex_sse2::aligned_malloc_pitch( pitch, ts.x_max*sizeof(TENSOR_FLOAT), ts.y_max );
            ts.pitch_x = (unsigned) pitch;
#else
            ts.elem = new TENSOR_FLOAT[ ts.y_max*ts.x_max ];
            ts.pitch_x = ts.x_max * sizeof(TENSOR_FLOAT);
#endif
        }
        inline void alloc_space( CTensor3D &ts ){
#if __APEX_TENSOR_USE_SSE__
            size_t pitch;
            ts.elem = (TENSOR_FLOAT*)apex_sse2::aligned_malloc_pitch( pitch, ts.x_max*sizeof(TENSOR_FLOAT), ts.y_max*ts.z_max );
            ts.pitch_x = (unsigned) pitch;            
#else
            ts.elem = new TENSOR_FLOAT[ ts.z_max*ts.y_max*ts.x_max ];
            ts.pitch_x  = ts.x_max * sizeof(TENSOR_FLOAT);
#endif
            ts.pitch_xy = ts.y_max * ts.pitch_x;
        }
        inline void alloc_space( CTensor4D &ts ){
#if __APEX_TENSOR_USE_SSE__
            size_t pitch;
            ts.elem = (TENSOR_FLOAT*)apex_sse2::aligned_malloc_pitch( pitch, ts.x_max*sizeof(TENSOR_FLOAT), ts.y_max*ts.z_max*ts.h_max );
            ts.pitch_x = (unsigned) pitch;            
#else
            ts.elem = new TENSOR_FLOAT[ ts.h_max*ts.z_max*ts.y_max*ts.x_max ];
            ts.pitch_x  = ts.x_max * sizeof(TENSOR_FLOAT);
#endif     
            ts.pitch_xy = ts.y_max * ts.pitch_x;
            ts.pitch_xyz= ts.z_max * ts.pitch_xy;
        }
    };

    namespace tensor{
        inline void dot_lt( CTensor2D &dst, const CTensor1D &lhs, const CTensor1D &rhs ){
            for( int y = 0 ; y < lhs.x_max; y ++ )
                for( int x = 0 ; x < rhs.x_max ; x ++ )
                    dst[y][x] = lhs[y] * rhs[x];
        }
    };

    
    namespace cpu_only{
        inline TENSOR_FLOAT dot( const CTensor1D &a, const CTensor1D &b ){
#if __APEX_TENSOR_USE_SSE__
            return apex_sse2::sdot<TENSOR_FLOAT>( a.elem, b.elem, a.x_max );
#else            
            TENSOR_FLOAT sum = 0.0f;
            for( int i = 0; i < a.x_max; i ++ ){
                sum += a[i] * b[i];
            }
            return sum;
#endif
        }
        
    };
#if __APEX_TENSOR_USE_BLAS__        
    namespace tensor{                      
        inline void ger( CTensor2D &dst, const CTensor1D &lhs, const CTensor1D &rhs, TENSOR_FLOAT alpha ){            
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dger( CblasRowMajor, lhs.x_max, rhs.x_max, alpha , lhs.elem, 1, rhs.elem, 1, dst.elem, dst.pitch_x/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sger( CblasRowMajor, lhs.x_max, rhs.x_max, alpha , lhs.elem, 1, rhs.elem, 1, dst.elem, dst.pitch_x/(sizeof(TENSOR_FLOAT)) );
#endif            
        }
        
        template<bool transposeRight>
        inline void gemv( CTensor1D &dst, const CTensor1D &lhs, const CTensor2D &rhs, TENSOR_FLOAT alpha, TENSOR_FLOAT beta ){            
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, 
                         transposeRight? CblasNoTrans:CblasTrans, 
                         rhs.y_max, rhs.x_max,
                         alpha , 
                         rhs.elem, rhs.pitch_x/(sizeof(TENSOR_FLOAT)), 
                         lhs.elem, 1, 
                         beta, 
                         dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, 
                         transposeRight? CblasNoTrans:CblasTrans, 
                         rhs.y_max, rhs.x_max,
                         alpha , 
                         rhs.elem, rhs.pitch_x/(sizeof(TENSOR_FLOAT)), 
                         lhs.elem, 1, 
                         beta, 
                         dst.elem, 1 );
#endif
        }
        
        template<bool transposeLeft, bool transposeRight>
        inline void gemm( CTensor2D &dst, const CTensor2D &lhs, const CTensor2D &rhs, TENSOR_FLOAT alpha, TENSOR_FLOAT beta ){            
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, 
                         transposeLeft  ? CblasTrans:CblasNoTrans, 
                         transposeRight ? CblasTrans:CblasNoTrans, 
                         transposeLeft  ? lhs.x_max : lhs.y_max,
                         transposeRight ? rhs.y_max : rhs.x_max,
                         transposeLeft  ? lhs.y_max : lhs.x_max,
                         alpha , 
                         lhs.elem, lhs.pitch_x/(sizeof(TENSOR_FLOAT)), 
                         rhs.elem, rhs.pitch_x/(sizeof(TENSOR_FLOAT)), 
                         beta, 
                         dst.elem, dst.pitch_x/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, 
                         transposeLeft  ? CblasTrans:CblasNoTrans, 
                         transposeRight ? CblasTrans:CblasNoTrans, 
                         transposeLeft  ? lhs.x_max : lhs.y_max,
                         transposeRight ? rhs.y_max : rhs.x_max,
                         transposeLeft  ? lhs.y_max : lhs.x_max,
                         alpha , 
                         lhs.elem, lhs.pitch_x/(sizeof(TENSOR_FLOAT)), 
                         rhs.elem, rhs.pitch_x/(sizeof(TENSOR_FLOAT)), 
                         beta, 
                         dst.elem, dst.pitch_x/(sizeof(TENSOR_FLOAT)) );
#endif
        }
    };
#else
    
    namespace tensor{
        inline void ger( CTensor2D &dst, const CTensor1D &lhs, const CTensor1D &rhs, TENSOR_FLOAT alpha ){            
            printf("ger not implemented\n"); exit(-1);
        }
        template<bool transposeRight>
        inline void gemv( CTensor1D &dst, const CTensor1D &lhs, const CTensor2D &rhs, TENSOR_FLOAT alpha, TENSOR_FLOAT beta ){            
            printf("gemv not implemented\n"); exit(-1);
        }
        template<bool transposeLeft, bool transposeRight>
        inline void gemm( CTensor2D &dst, const CTensor2D &lhs, const CTensor2D &rhs, TENSOR_FLOAT alpha, TENSOR_FLOAT beta ){            
            printf("gemm not implemented\n"); exit(-1);
        }
        
        };
#endif

    namespace tensor{
        inline void sadd( CTensor2D &dst, const CTensor1D &src ){
            for( int y = 0; y < dst.y_max; y ++ )
                dst[ y ] += src;
        }

        inline void sfill( CTensor2D &dst, const CTensor1D &src ){
            for( int y = 0; y < dst.y_max; y ++ )
                dst[ y ] = src;
        }

        inline void sdiv_y( CTensor2D &dst, const CTensor1D &src ){
            for( int y = 0; y < dst.y_max; y ++ )
                for( int x = 0; x < dst.x_max; x ++ )
                    dst[ y ][ x ] /= src[ y ];
        }

        inline void smul_y( CTensor2D &dst, const CTensor1D &src ){
            for( int y = 0; y < dst.y_max; y ++ )
                for( int x = 0; x < dst.x_max; x ++ )
                    dst[ y ][ x ] *= src[ y ];
        }

        template<typename ST>
        inline void sum_row( CTensor1D &dst, const CTensor2D &src ){
            for( int x = 0; x < dst.x_max; x ++ ){
                TENSOR_FLOAT sum = 0.0f;
                for( int y = 0; y < src.y_max; y ++ ){
                    sum += src[y][x];
                }
                cpu_template::Store<ST>::store( dst[x] ,sum );                
            }
        }

        // sum over last two dimension
        template<typename ST>
        inline void sum_2D( CTensor1D &dst, const CTensor3D &src ){
            for( int i = 0 ; i < dst.x_max ; i ++ )
                cpu_template::Store<ST>::store( dst[i] ,cpu_only::sum( src[i] ) );
        }
       
        template<typename ST>
        inline void sum_2D( CTensor2D &dst, const CTensor4D &src ){
            for( int i = 0 ; i < dst.y_max ; i ++ )
                for( int j = 0 ; j < dst.x_max ; j ++ )
                    cpu_template::Store<ST>::store( dst[i][j] ,cpu_only::sum( src[i][j] ) );
        }       
     
        inline  void norm_softmax_y( CTensor2D &mean, const CTensor2D &energy ){
            for( int x = 0; x < mean.x_max; x ++ ){
                TENSOR_FLOAT mmax = energy[0][x];
                for( int y = 1; y < mean.y_max; y ++ )
                    if( mmax < energy[y][x] ) mmax = energy[y][x];
                TENSOR_FLOAT sum = 0.0f;
                for( int y = 0; y < mean.y_max; y ++ ){
                    mean[y][x] = (TENSOR_FLOAT)exp( energy[y][x] - mmax );
                    sum += mean[y][x];
                }
                for( int y = 0; y < mean.y_max; y ++ ){
                    mean[y][x] /= sum;
                }
            }
        }

        inline void sample_softmax_y( CTensor2D &state, const CTensor2D &mean ){
            for( int x = 0; x < state.x_max; x ++ ){
                bool hit = false;
                TENSOR_FLOAT rnd = (TENSOR_FLOAT)apex_random::next_double();                
                for( int y = 0; y < state.y_max; y ++ ){
                    rnd -= mean[y][x];
                    if( !hit && rnd < 0 ){
                        state[y][x] = 1.0f; hit = true; 
                    }else{
                        state[y][x] = 0.0f; 
                    }                                
                }
            }
        }
    };
};

/*------^-^--------------------------------------------------------*
 *  the following codes repeat expand the common functions for 1D-4D
 *-----------------------------------------------------------------*/
#ifdef For1CTXDLINE
    #error "macro For1CTXDLINE must not be defined"
#endif
#ifdef For2CTXDLINE
    #error "macro For2CTXDLINE must not be defined"
#endif
#ifdef For3CTXDLINE
    #error "macro For3CTXDLINE must not be defined"
#endif
#ifdef _CTensorXD
    #error "macro _CTensorXD must not be defined"
#endif
#ifdef _XD
    #error "macro _XD must not be defined"
#endif

// expand the common funtions for CTensor1D
#define _XD 1
#define _CTensorXD apex_tensor::CTensor1D    
#define For1CTXDLINE(cst,line,ts)                                       \
    cst TENSOR_FLOAT *line = ts.elem;                                   \
    
#define For2CTXDLINE(cst,line,ts,lineA,tsA)                             \
    cst TENSOR_FLOAT *line = ts.elem;                                   \
    const TENSOR_FLOAT *lineA = tsA.elem;                               \

#define For3CTXDLINE(cst,line,ts,lineA,tsA,lineB,tsB)                   \
    cst TENSOR_FLOAT *line = ts.elem;                                   \
    const TENSOR_FLOAT *lineA = tsA.elem;                               \
    const TENSOR_FLOAT *lineB = tsB.elem;                               \


#include "apex_tensor_cpu_inline_common.h"

#undef For1CTXDLINE
#undef For2CTXDLINE
#undef For3CTXDLINE
#undef _CTensorXD
#undef _XD

// expand the common funtions for CTensor2D
#define _XD 2
#define _CTensorXD apex_tensor::CTensor2D
#define For1CTXDLINE(cst,line,ts)                                       \
    cst TENSOR_FLOAT *line;                                             \
    for( int y = 0; (line = ts[y].elem , y < ts.y_max) ; y ++ )         \
        
#define For2CTXDLINE(cst,line,ts,lineA,tsA)                             \
    cst TENSOR_FLOAT *line;                                             \
    const TENSOR_FLOAT *lineA;                                          \
    for( int y = 0; (line = ts[y].elem,lineA = tsA[y].elem, y < ts.y_max) ; y ++ ) \
        
#define For3CTXDLINE(cst,line,ts,lineA,tsA,lineB,tsB)                   \
    cst TENSOR_FLOAT *line;                                             \
    const TENSOR_FLOAT *lineA;                                          \
    const TENSOR_FLOAT *lineB;                                          \
    for( int y = 0; (line = ts[y].elem,lineA = tsA[y].elem,lineB = tsB[y].elem, y < ts.y_max) ; y ++ ) \

#include "apex_tensor_cpu_inline_common.h"
#undef For1CTXDLINE
#undef For2CTXDLINE
#undef For3CTXDLINE
#undef _CTensorXD
#undef _XD

// expand the common functions for CTensor3D
#define _XD  3
#define _CTensorXD apex_tensor::CTensor3D
#define For1CTXDLINE(cst,line,ts)                                       \
    cst TENSOR_FLOAT *line;                                             \
    for( int z = 0; z < ts.z_max ; z++ )                                \
        for( int y = 0; (line = ts[z][y].elem , y < ts.y_max) ; y++ )   \
            
#define For2CTXDLINE(cst,line,ts,lineA,tsA)                             \
    cst TENSOR_FLOAT *line;                                             \
    const TENSOR_FLOAT *lineA;                                          \
    for( int z = 0; z < ts.z_max ; z++ )                                \
        for( int y = 0; (line = ts[z][y].elem ,lineA = tsA[z][y].elem, y < ts.y_max) ; y++ ) \
            
#define For3CTXDLINE(cst,line,ts,lineA,tsA,lineB,tsB)                   \
    cst TENSOR_FLOAT *line;                                             \
    const TENSOR_FLOAT *lineA;                                          \
    const TENSOR_FLOAT *lineB;                                          \
    for( int z = 0; z < ts.z_max ; z++ )                                \
        for( int y = 0; (line = ts[z][y].elem ,lineA = tsA[z][y].elem, lineB = tsB[z][y].elem, y < ts.y_max) ; y++ ) \
            
#include "apex_tensor_cpu_inline_common.h"
#undef For1CTXDLINE
#undef For2CTXDLINE
#undef For3CTXDLINE
#undef _CTensorXD
#undef _XD

// expand the common functions for CTensor4D
#define _XD 4
#define _CTensorXD apex_tensor::CTensor4D
#define For1CTXDLINE(cst,line,ts)                                       \
    cst TENSOR_FLOAT *line;                                             \
    for( int h = 0; h < ts.h_max ; h++ )                                \
        for( int z = 0; z < ts.z_max ; z++ )                            \
            for( int y = 0; (line = ts[h][z][y].elem , y < ts.y_max) ; y++ ) \
                
#define For2CTXDLINE(cst,line,ts,lineA,tsA)                             \
    cst TENSOR_FLOAT *line;                                             \
    const TENSOR_FLOAT *lineA;                                          \
    for( int h = 0; h < ts.h_max ; h++ )                                \
        for( int z = 0; z < ts.z_max ; z++ )                            \
            for( int y = 0; (line = ts[h][z][y].elem ,lineA = tsA[h][z][y].elem, y < ts.y_max) ; y++ ) \
                
#define For3CTXDLINE(cst,line,ts,lineA,tsA,lineB,tsB)                   \
    cst TENSOR_FLOAT *line;                                             \
    const TENSOR_FLOAT *lineA;                                          \
    const TENSOR_FLOAT *lineB;                                          \
    for( int h = 0; h < ts.h_max ; h++ )                                \
        for( int z = 0; z < ts.z_max ; z++ )                            \
            for( int y = 0; (line = ts[h][z][y].elem ,lineA = tsA[h][z][y].elem, lineB = tsB[h][z][y].elem, y < ts.y_max) ; y++ ) \

#include "apex_tensor_cpu_inline_common.h"
#undef For1CTXDLINE
#undef For2CTXDLINE
#undef For3CTXDLINE
#undef _CTensorXD
#undef _XD

#endif

