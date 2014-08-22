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

// CPU inline implementation of the commone functions 
// in CTensor , this file will be included several times from 1D to 4D
// \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
// this file will not be documented 

#ifdef For1CTXDELEM
    #error "macro For1CTXDELEM must not be defined"
#endif
#ifdef For2CTXDELEM
    #error "macro For2CTXDELEM must not be defined"
#endif
#ifdef For3CTXDELEM
    #error "macro For3CTXDELEM must not be defined"
#endif

#define For1CTXDELEM(cst,line,ts)                  \
    For1CTXDLINE(cst,line,ts)                      \
    for( int i = 0 ; i < ts.x_max; i++)            \
        
#define For2CTXDELEM(cst,line,ts,lineA,tsA)        \
    For2CTXDLINE(cst,line,ts,lineA,tsA)            \
    for( int i = 0 ; i < ts.x_max; i++)            \
        
#define For3CTXDELEM(cst,line,ts,lineA,tsA,lineB,tsB) \
    For3CTXDLINE(cst,line,ts,lineA,tsA,lineB,tsB)     \
    for( int i = 0 ; i < ts.x_max; i++)               \
        
#include <cmath>
#include <cstring>
#include "apex_random.h"

#if __APEX_TENSOR_USE_SSE__
#include "apex_tensor_sse.h"
#endif

namespace apex_tensor{
    namespace cpu_only{
        inline void save_to_stream( const _CTensorXD &ts, IStream &dst ){
            dst.write( &ts, _XD*sizeof(int) );
            For1CTXDLINE( const, tl, ts ){
                dst.write( tl, sizeof(TENSOR_FLOAT)*ts.x_max );
            }
        }
        inline void load_from_stream( _CTensorXD &ts, IStream &src, bool pre_alloc ){
            cpu_template::assert_true( src.read( &ts, _XD*sizeof(int) ) > 0, "tensor::load_from_file" );
            if( !pre_alloc ) apex_tensor::tensor::alloc_space( ts );
            For1CTXDLINE( , tl, ts ){
                if( ts.x_max > 0 )
                    cpu_template::assert_true( src.read( tl, sizeof(TENSOR_FLOAT)*ts.x_max ) > 0, 
                                               "tensor::load_from_file");
            }                
        }        
    };

    namespace cpu_only{
        inline void save_to_file( const _CTensorXD &ts, FILE *dst ){
            fwrite( &ts, _XD*sizeof(int) , 1 , dst );
            For1CTXDLINE( const, tl, ts ){
                fwrite( tl, sizeof(TENSOR_FLOAT), ts.x_max, dst );
            }
        }
        
        inline void load_from_file( _CTensorXD &ts, FILE *src, bool pre_alloc ){
            cpu_template::assert_true( fread( &ts, _XD*sizeof(int) , 1 , src ) > 0, "tensor::load_from_file" );
            if( !pre_alloc ) apex_tensor::tensor::alloc_space( ts );
            For1CTXDLINE( , tl, ts ){
                if( ts.x_max > 0 )
                    cpu_template::assert_true( fread( tl, sizeof(TENSOR_FLOAT) , ts.x_max, src ) > 0, 
                                               "tensor::load_from_file");
            }                
        }
        
        inline TENSOR_FLOAT sum( const _CTensorXD &a ){
            double ans = 0.0;
#if __APEX_TENSOR_USE_SSE__
            For1CTXDLINE( const, al, a ){
                ans += apex_sse2::ssum( al, a.x_max );
            }
#else
            For1CTXDELEM( const, al, a ){
                ans += al[i];
            }
#endif
            return (TENSOR_FLOAT) ans;
        }

        inline TENSOR_FLOAT avg( const _CTensorXD &a ){
            return sum( a ) / a.num_elem();
        }

        inline TENSOR_FLOAT var( const _CTensorXD &a ){
            double s = 0.0, ss = 0.0;
            For1CTXDELEM( const, al, a ){
                s += al[i]; ss += al[i]*al[i];
            }
            int n = a.num_elem();            
            return (TENSOR_FLOAT)(ss/n - (s/n)*(s/n));
        }

        inline TENSOR_FLOAT std_var( const _CTensorXD &a ){
            return (TENSOR_FLOAT)sqrt( var( a ));
        }

        inline TENSOR_FLOAT min_value( const _CTensorXD &a ){
            TENSOR_FLOAT ans = a.elem[0];
            For1CTXDELEM( const, al, a ){
                if( ans > al[i] ) ans = al[i]; 
            }            
            return ans;
        }

        inline TENSOR_FLOAT max_value( const _CTensorXD &a ){
            TENSOR_FLOAT ans = a.elem[0];
            For1CTXDELEM( const, al, a ){
                if( ans < al[i] ) ans = al[i]; 
            }            
            return ans;            
        }
    };
    
    namespace async{
        inline void set_dependecy( _CTensorXD &ts, unsigned int stream_id ){}
    };
    namespace tensor{
        inline void free_space( _CTensorXD &ts ){
            TENSOR_FLOAT *ptr = ts.elem;
#if __APEX_TENSOR_USE_SSE__
            apex_sse2::aligned_free( ptr );
#else
            delete[] ptr;
#endif
        }

        inline void copy( _CTensorXD &dst,  const _CTensorXD &src ){
            For2CTXDLINE( , dl, dst, sl, src ){
                memcpy( dl, sl, sizeof(TENSOR_FLOAT)*dst.x_max );
            }
        }               
        
        inline void fill( _CTensorXD &ts, TENSOR_FLOAT scalar ){
#if __APEX_TENSOR_USE_SSE__
            For1CTXDLINE( , dl, ts ){
                apex_sse2::scalar<apex_exp_template::enums::SaveTo>( dl, scalar, ts.x_max );
            }
#else
            For1CTXDELEM( , dl, ts ){
                dl[i] = scalar;
            }
#endif
        }

        inline void regularize_L1( _CTensorXD &ts,  TENSOR_FLOAT eps ){
            For1CTXDELEM( , dl, ts ){
                if( dl[i] > eps ) dl[i] -= eps;
                else 
                    if ( dl[i] < - eps ) dl[i] += eps;
                    else dl[i] = 0.0f;
            }
        }               

        inline void smaller_then_fill( _CTensorXD &ts,  TENSOR_FLOAT scalar ){
            For1CTXDELEM( , dl, ts ){
                if( dl[i] <= scalar ) dl[i] = scalar;
            }
        }               

        inline void sadd__sign( _CTensorXD &dst, const _CTensorXD &src, TENSOR_FLOAT scalar  ){
            For2CTXDELEM( , dl, dst, sl, src ){
                if( sl[i] > 0 ) dl[i] += scalar;
                else dl[i] -= scalar;
            } 
        }               

        template<typename ST,typename OP>
        inline void scalar_map( _CTensorXD &dst, const _CTensorXD &lhs, TENSOR_FLOAT scalar  ){
#if __APEX_TENSOR_USE_SSE__
            For2CTXDLINE( , dl, dst, ll, lhs ){
                apex_sse2::scalar_map<ST,OP,apex_tensor::TENSOR_FLOAT>( dl, ll, scalar, dst.x_max );
            } 
#else
            For2CTXDELEM( , dl, dst, ll, lhs ){
                cpu_template::Store<ST>::store( dl[i], cpu_template::BinaryMap<OP>::map( ll[i] , scalar ) ); 
            }             
#endif
        }               
        
        template<typename ST,typename OP>
        inline void binary_map( _CTensorXD &dst, const _CTensorXD &lhs, const _CTensorXD &rhs  ){
#if __APEX_TENSOR_USE_SSE__
            For3CTXDLINE( , dl, dst, ll, lhs, rl, rhs ){
                apex_sse2::binary_map<ST,OP,apex_tensor::TENSOR_FLOAT>( dl, ll, rl, dst.x_max );
            }
#else
            For3CTXDELEM( , dl, dst, ll, lhs, rl, rhs ){
                cpu_template::Store<ST>::store( dl[i], cpu_template::BinaryMap<OP>::map( ll[i] , rl[i] ) ); 
            }
#endif
        }               
        
        template<typename ST>
        inline void scale_add( _CTensorXD &dst, const _CTensorXD &lhs, const _CTensorXD &rhs, TENSOR_FLOAT sa, TENSOR_FLOAT sb ){
            For3CTXDELEM( , dl, dst, ll, lhs, rl, rhs ){
                cpu_template::Store<ST>::store( dl[i], ll[i]*sa + rl[i]*sb );
            }
        }        
    };
    
    namespace tensor{
        inline void sigmoid( _CTensorXD &dst, const _CTensorXD &src ){
            For2CTXDELEM( , dl, dst, sl, src ){
                dl[i] = (TENSOR_FLOAT)(1.0/( 1.0 + exp( - sl[i] ) ));
            }
        }

        inline void map_sqrt( _CTensorXD &dst, const _CTensorXD &src ){
            For2CTXDELEM( , dl, dst, sl, src ){
                dl[i] = (TENSOR_FLOAT)sqrt( sl[i] );
            }
        }

        inline void sample_binary( _CTensorXD &dst, const _CTensorXD &src ){
            For2CTXDELEM( , dl, dst, sl, src ){
                dl[i] = (TENSOR_FLOAT)apex_random::sample_binary( sl[i] );
            }
        }
        
        inline void sample_gaussian( _CTensorXD &dst, const _CTensorXD &src, TENSOR_FLOAT sd ){
            For2CTXDELEM( , dl, dst, sl, src ){
                dl[i] = (TENSOR_FLOAT)apex_random::sample_normal( sl[i], sd );
            }
        }
        
        // sample gaussian that should be two times fater
        inline void sample_gaussianF( _CTensorXD &dst, const _CTensorXD &src, TENSOR_FLOAT sd ){
            double rx, ry = 0.0;
            if( sd < 1e-8f ){
                For2CTXDELEM( , dl, dst, sl, src ){
                    dl[ i ] = sl[i];                    
                }
                return;
            }
            
            For2CTXDELEM( , dl, dst, sl, src ){
                if( (i & 1) == 0 ){
                    apex_random::sample_normal2D( rx, ry );
                    dl[ i ] = (TENSOR_FLOAT)( sl[i] + rx * sd );                     
                }else{
                    dl[ i ] = (TENSOR_FLOAT)( sl[i] + ry * sd );
                }
            }
        }
        
        inline void sample_gaussian( _CTensorXD &dst, TENSOR_FLOAT sd ){
            For1CTXDELEM( , dl, dst){
                dl[i] = (TENSOR_FLOAT)apex_random::sample_normal() * sd;
            }
        } 

        inline void sample_uniform( _CTensorXD &dst ){
            For1CTXDELEM( , dl, dst){
                dl[i] = (TENSOR_FLOAT)apex_random::next_double();
            }
        }       
    };   
};

#undef For1CTXDELEM
#undef For2CTXDELEM
#undef For3CTXDELEM
