#ifndef _APEX_TENSOR_SSE_H_
#define _APEX_TENSOR_SSE_H_


#include <cmath>
#include "apex_exp_template.h"
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#ifndef _APEX_GPU_COMPILE_MODE_
#include <emmintrin.h>
#endif

namespace apex_sse2{
    inline void* aligned_malloc( size_t space, bool allow_failure = false ){
#ifdef _MSC_VER
        void* res = _aligned_malloc( ((space + 15) >> 4) << 4, 16 ); 
#else
  #ifdef __APPLE__
        void* res = malloc( ((space+15)>>4) << 4); // already aligned
  #else
        void* res =  memalign( 16, ((space + 15) >> 4) << 4 ); 
  #endif
#endif
        if( !allow_failure && res == NULL ){
            fprintf( stderr, "align_malloc error" );
            exit( -1 );
        }
        return res;
    }

    inline void* aligned_malloc_pitch( size_t &pitch, size_t space, size_t num_line, bool allow_failure = false  ){
        pitch = ((space+15) >> 4) << 4;
#ifdef _MSC_VER
        void * res = _aligned_malloc( pitch*num_line, 16 ); 
#else
  #ifdef __APPLE__
        void *res = malloc(pitch*num_line);
  #else
        void * res =  memalign( 16, pitch*num_line ); 
  #endif
#endif
        if( !allow_failure && res == NULL ){
            fprintf( stderr, "align_malloc_pitch error" );
            exit( -1 );
        }
        return res;
    }

    inline void aligned_free( void *ptr ){
#ifdef _MSC_VER
        _aligned_free( ptr );
#else
        free( ptr );
#endif        
    } 
};

#ifndef _APEX_GPU_COMPILE_MODE_    
namespace apex_sse2{
    template<typename Scalar> struct mm{};
    template<> 
    struct mm<float> {
        typedef __m128 type;        
        enum{ size = 4 };
        inline static int upper_norm( int size ){
            return ((size+3) >> 2) << 2;
        }
        inline static int lower_norm( int size ){
            return (size >> 2) << 2;
        }
        
        inline static type zero( void ){
            return _mm_setzero_ps();
        }
        inline static type set1( const float &src ){
            return _mm_set1_ps( src );
        }
        inline static void store( float *dst, const type &src ){
            return _mm_store_ps( dst, src );
        }
        inline static type load( const float *src ){
            return _mm_load_ps( src );
        }
        inline static type add( const type &lhs, const type &rhs ){
            return _mm_add_ps( lhs, rhs );
        }
        inline static type sub( const type &lhs, const type &rhs ){
            return _mm_sub_ps( lhs, rhs );
        }
        inline static type mul( const type &lhs, const type &rhs ){
            return _mm_mul_ps( lhs, rhs );
        }
        inline static type div( const type &lhs, const type &rhs ){
            return _mm_div_ps( lhs, rhs );
        }
        
        inline static float sum_all( const type &src ){
            type ans  = _mm_add_ps( src, _mm_movehl_ps( src, src ) );
            type rst  = _mm_add_ss( ans, _mm_shuffle_ps( ans, ans, 1 ) );
#if defined(_MSC_VER) && ( _MSC_VER <= 1500 ) && defined(_WIN64)
            return rst.m128_f32[ 0 ];
#else
            float rr = _mm_cvtss_f32( rst ) ;
            return rr;
#endif
        }
    };
    template<> 
    struct mm<double> {
        typedef __m128d type;
        enum{ size = 2 };
        inline static int upper_norm( int size ){
            return ((size+1) >> 1)<<1;
        }
        inline static int lower_norm( int size ){
            return (size >> 1)<<1;
        }
        inline static type zero( void ){
            return _mm_setzero_pd();
        }
        inline static type set1( const double &src ){
            return _mm_set1_pd( src );
        }
        inline static type load( const double *src ){
            return _mm_load_pd( src );
        }        
        inline static void store( double *dst, const type &src ){
            return _mm_store_pd( dst, src );
        }
        inline static type add( const type &lhs, const type &rhs ){
            return _mm_add_pd( lhs, rhs );
        }
        inline static type sub( const type &lhs, const type &rhs ){
            return _mm_sub_pd( lhs, rhs );
        }
        inline static type mul( const type &lhs, const type &rhs ){
            return _mm_mul_pd( lhs, rhs );
        }
        inline static type div( const type &lhs, const type &rhs ){
            return _mm_div_pd( lhs, rhs );
        }

        inline static double sum_all( const type &src ){
            __m128d tmp =  _mm_add_sd( src, _mm_unpackhi_pd( src,src ) ) ;
#if defined(_MSC_VER) && ( _MSC_VER <= 1500 ) && defined(_WIN64)
            return tmp.m128d_f64[0];
#else
            double ans = _mm_cvtsd_f64( tmp );
            return ans;
#endif
        }
    };   
};

namespace apex_sse2{        
    // storage method
    template<typename ST, typename ScalarType, typename VecType>
    struct Store{
        inline static void store( ScalarType *dst, const VecType &src  );
    };
    template<typename ScalarType, typename VecType>
    struct Store<apex_exp_template::enums::SaveTo, ScalarType, VecType>{
        inline static void store( ScalarType *dst, const VecType &src  ){
            mm<ScalarType>::store( dst, src );
        }
    };
    template<typename ScalarType, typename VecType>
    struct Store<apex_exp_template::enums::AddTo, ScalarType, VecType>{
        inline static void store( ScalarType *dst, const VecType &src  ){
            typename mm<ScalarType>::type lhs = mm<ScalarType>::load( dst );
            typename mm<ScalarType>::type ans = mm<ScalarType>::add( lhs, src );
            mm<ScalarType>::store( dst, ans );
        }
    };
    template<typename ScalarType, typename VecType>
    struct Store<apex_exp_template::enums::MulTo, ScalarType, VecType>{
        inline static void store( ScalarType *dst, const VecType &src  ){
            typename mm<ScalarType>::type lhs = mm<ScalarType>::load( dst );
            typename mm<ScalarType>::type ans = mm<ScalarType>::mul( lhs, src );
            mm<ScalarType>::store( dst, ans );
        }
    };
    template<typename ScalarType, typename VecType>
    struct Store<apex_exp_template::enums::SubTo, ScalarType, VecType>{
        inline static void store( ScalarType *dst, const VecType &src  ){
            typename mm<ScalarType>::type lhs = mm<ScalarType>::load( dst );
            typename mm<ScalarType>::type ans = mm<ScalarType>::sub( lhs, src );
            mm<ScalarType>::store( dst, ans );
        }
    };
    template<typename ScalarType, typename VecType>
    struct Store<apex_exp_template::enums::DivTo, ScalarType, VecType>{
        inline static void store( ScalarType *dst, const VecType &src  ){
            typename mm<ScalarType>::type lhs = mm<ScalarType>::load( dst );
            typename mm<ScalarType>::type ans = mm<ScalarType>::div( lhs, src );
            mm<ScalarType>::store( dst, ans );
        }
    };
    // binary op 
    template<typename OP,typename ScalarType, typename VecType>
    struct BinaryOp{ 
        inline static VecType map( const VecType &lhs, const VecType &rhs );
    };

    template<typename ScalarType, typename VecType>
    struct BinaryOp<apex_exp_template::enums::Add,ScalarType, VecType>{ 
        inline static VecType map( const VecType &lhs, const VecType &rhs ){
            return mm<ScalarType>::add( lhs, rhs );
        }
    };

    template<typename ScalarType, typename VecType>
    struct BinaryOp<apex_exp_template::enums::Sub,ScalarType, VecType>{ 
        inline static VecType map( const VecType &lhs, const VecType &rhs ){
            return mm<ScalarType>::sub( lhs, rhs );
        }
    };

    template<typename ScalarType, typename VecType>
    struct BinaryOp<apex_exp_template::enums::Mul,ScalarType, VecType>{ 
        inline static VecType map( const VecType &lhs, const VecType &rhs ){
            return mm<ScalarType>::mul( lhs, rhs );
        }
    };

    template<typename ScalarType, typename VecType>
    struct BinaryOp<apex_exp_template::enums::Div,ScalarType, VecType>{ 
        inline static VecType map( const VecType &lhs, const VecType &rhs ){
            return mm<ScalarType>::div( lhs, rhs );
        }
    };
    
    template <typename ST,typename OP, typename Scalar>
    struct ScalarOptimizer{
        inline bool static scalar_map( Scalar *pdst, const Scalar *psrc, Scalar scalar, int n ){             
            return false; 
        }
    };
    
    template <typename ST, typename Scalar>
    struct ScalarOptimizer<ST,apex_exp_template::enums::Mul, Scalar>{
        inline bool static scalar_map( Scalar *pdst, const Scalar *psrc, Scalar scalar, int n ){ 
            if( fabs( scalar - 1.0f ) > 1e-6 ) return false;
            const int len = mm<Scalar>::upper_norm( n );
            for( int i = 0; i < len; i += mm<Scalar>::size ){
                typename mm<Scalar>::type lhs = mm<Scalar>::load( psrc + i );            
                Store<ST,Scalar, typename mm<Scalar>::type>::store( pdst + i, lhs );
            }  
            return true;
        }
    };

};

namespace apex_sse2{        
    // dst [st] scalar
    // assume alignment, and extra space
    template<typename ST,typename Scalar>
    inline void scalar( Scalar *pdst, Scalar scalar, int n ){
        const int len = mm<Scalar>::upper_norm( n );
        typename mm<Scalar>::type ans = mm<Scalar>::set1( scalar );
        for( int i = 0; i < len; i += mm<Scalar>::size ){
            Store<ST,Scalar, typename mm<Scalar>::type>::store
                ( pdst + i, ans );
        }
    }        

    // dst [st] lhs [op] scalar
    // assume alignment, and  extra space
    template<typename ST,typename OP,typename Scalar>
    inline void scalar_map( Scalar *pdst, const Scalar *psrc, Scalar scalar, int n ){
        if( ScalarOptimizer<ST,OP,Scalar>::scalar_map( pdst, psrc, scalar, n) ) return;
        const int len = mm<Scalar>::upper_norm( n );
        typename mm<Scalar>::type rhs= mm<Scalar>::set1( scalar );
        for( int i = 0; i < len; i += mm<Scalar>::size ){
            typename mm<Scalar>::type lhs = mm<Scalar>::load( psrc + i );            
            Store<ST,Scalar, typename mm<Scalar>::type>::store
                ( pdst + i, 
                  BinaryOp< OP,Scalar, typename mm<Scalar>::type >::map( lhs, rhs ) );
        }
    }        
           
    // dst [st] lhs [op] rhs
    // assume alignment, and extra space
    template<typename ST,typename OP,typename Scalar>
    inline void binary_map( Scalar *pdst, const Scalar *plhs, const Scalar *prhs, int n ){
        const int len = mm<Scalar>::upper_norm( n );
        for( int i = 0; i < len; i += mm<Scalar>::size ){
            typename mm<Scalar>::type lhs = mm<Scalar>::load( plhs + i );            
            typename mm<Scalar>::type rhs = mm<Scalar>::load( prhs + i );            
            Store<ST,Scalar, typename mm<Scalar>::type>::store
                ( pdst + i, 
                  BinaryOp< OP,Scalar, typename mm<Scalar>::type >::map( lhs, rhs ) );
        }
    }        

    // return dot( lhs, rhs )
    template<typename Scalar>
    inline Scalar sdot( const Scalar *plhs, const Scalar *prhs, int n ){
        if( plhs == prhs ){
            const int len = mm<Scalar>::lower_norm( n );
            typename mm<Scalar>::type ans = mm<Scalar>::zero();            
            for( int i = 0; i < len; i += mm<Scalar>::size ){
                typename mm<Scalar>::type lhs = mm<Scalar>::load( plhs + i );            
                ans = mm<Scalar>::add( ans, mm<Scalar>::mul( lhs, lhs ) );
            }                       
            Scalar sum = mm<Scalar>::sum_all( ans );
            for( int i = len; i < n; i ++ ){
                sum += plhs[ i ] * plhs[ i ];
            } 
            return sum;
        }else{
            const int len = mm<Scalar>::lower_norm( n );
            typename mm<Scalar>::type ans = mm<Scalar>::zero();            
            for( int i = 0; i < len; i += mm<Scalar>::size ){
                typename mm<Scalar>::type lhs = mm<Scalar>::load( plhs + i );            
                typename mm<Scalar>::type rhs = mm<Scalar>::load( prhs + i );            
                ans = mm<Scalar>::add( ans, mm<Scalar>::mul( lhs, rhs) );
            }                       
            Scalar sum = mm<Scalar>::sum_all( ans );
            for( int i = len; i < n; i ++ ){
                sum += plhs[ i ] * prhs[ i ];
            } 
            return sum;
        }
     }                

    // return sum( src )
    template<typename Scalar>
    inline Scalar ssum( const Scalar *psrc, int n ){
        const int len = mm<Scalar>::lower_norm( n );
        typename mm<Scalar>::type ans = mm<Scalar>::zero();            
        for( int i = 0; i < len; i += mm<Scalar>::size ){
            typename mm<Scalar>::type src = mm<Scalar>::load( psrc + i );            
            ans = mm<Scalar>::add( ans, src );
        }                       
        Scalar sum = mm<Scalar>::sum_all( ans );
        for( int i = len; i < n; i ++ ){
            sum += psrc[i];
        } 
        return sum;
     }                
};
#else
// dummy implementations to avoid compile error 
namespace apex_sse2{
    template<typename ST,typename Scalar>
    inline void scalar( Scalar *pdst, Scalar scalar, int n ){}
    template<typename ST,typename OP,typename Scalar>
    inline void scalar_map( Scalar *pdst, const Scalar *psrc, Scalar scalar, int n ){}
    template<typename ST,typename OP,typename Scalar>
    inline void binary_map( Scalar *pdst, const Scalar *plhs, const Scalar *prhs, int n ){}
    template<typename Scalar>
    inline Scalar sdot( const Scalar *plhs, const Scalar *prhs, int n ){
        return 0.0f;
    }
    template<typename Scalar>
    inline Scalar ssum( const Scalar *psrc, int n ){
        return 0.0f;
    }
};
#endif

#endif

