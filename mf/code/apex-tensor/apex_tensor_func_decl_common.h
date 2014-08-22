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
 * \file apex_tensor_func_decl_common.h
 * \brief declaration of common functions of all tensors
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 * \sa apex_tensor_func_decl.h  _TensorXD _CTensorXD _GTensorXD
 */

#include <cstdio>
#include "apex_stream.h"

namespace apex_tensor{    
    namespace cpu_only{
        /*! 
         * \brief save data to file 
         * \param ts object to save
         * \param dst destination file
         * \sa _CTensorXD
         */
        inline void save_to_file( const _CTensorXD &ts, FILE *dst );
        /*! 
         * \brief load data from file, this function will automicly allocate the necessary space needed if pre_alloc=false 
         * \param ts object to load
         * \param src source file
         * \param pre_alloc whether the space is pre-allocated
         * \sa _CTensorXD
         *
         * ts must not be allocated yet, this function will allocate the space according to data in file
         */
        inline void load_from_file( _CTensorXD &ts, FILE *src, bool pre_alloc=false );        

        /*! 
         * \brief save data to stream
         * \param ts object to save
         * \param dst destination stream
         * \sa _CTensorXD
         */
        inline void save_to_stream( const _CTensorXD &ts, IStream &dst );
        /*! 
         * \brief load data from stream, this function will automicly allocate the necessary space needed if pre_alloc=false 
         * \param ts object to load
         * \param src source stream
         * \param pre_alloc whether the space is pre-allocated
         * \sa _CTensorXD
         *
         * ts must not be allocated yet, this function will allocate the space according to data in file
         */
        inline void load_from_stream( _CTensorXD &ts, IStream &src, bool pre_alloc=false );        
                
        /*! \brief sum of all elements */
        inline TENSOR_FLOAT sum( const _CTensorXD &a );
        /*! \brief average of all elements */
        inline TENSOR_FLOAT avg( const _CTensorXD &a );
        /*! \brief variance of all elements */
        inline TENSOR_FLOAT var( const _CTensorXD &a );
        /*! \brief standard variance of all elements */
        inline TENSOR_FLOAT std_var( const _CTensorXD &a );
        /*! \brief min_value of all elements */
        inline TENSOR_FLOAT min_value( const _CTensorXD &a );
        /*! \brief max_value of all elements */
        inline TENSOR_FLOAT max_value( const _CTensorXD &a );
    };
};

namespace apex_tensor{
    namespace tensor{
        /*! 
         * \brief allocate space for a {C|G}Tensor{1-4}D object
         * \param ts  object to allocate space
         * \sa _TensorXD 
         */
        _INLINE void alloc_space( _TensorXD &ts );               

        /*! 
         * \brief free space for a {C|G}Tensor{1-4}D object
         * \param ts object to free space
         * \sa _TensorXD 
         */        
        _INLINE void free_space( _TensorXD &ts );               

        /*! 
         * \brief copy CTensor{x}D to {C|G}Tensor{x}D object, x={1-4}
         * \param dst copy destination
         * \param src copy source
         * \sa _TensorXD _CTensorXD
         */        
        _INLINE void copy( _TensorXD &dst,  const _CTensorXD &src );        
                
        /*! 
         * \brief dst = scalar, fill scalar into dst
         * \param dst destination
         * \param scalar scalar parameter to be filled 
         */
        _INLINE void fill( _TensorXD &dst,  TENSOR_FLOAT scalar );               
        
        /*! 
         * \brief perform L1 regularize on dst
         */
        _INLINE void regularize_L1( _TensorXD &dst,  TENSOR_FLOAT eps );           

        /*! 
         * \brief dst = dst < scalar ? :dst, fill scalar into dst if dst is smaller than scalar
         * \param dst destination
         * \param scalar scalar parameter to be filled 
         */
        _INLINE void smaller_then_fill( _TensorXD &dst,  TENSOR_FLOAT scalar );           
        
        /*! 
         * \brief dst += sign(src) * scalar
         * \param dst destination
         * \param src source
         * \param scalar scalar parameter
         */
        _INLINE void sadd__sign( _TensorXD &dst, const _TensorXD &src,  TENSOR_FLOAT scalar );

        /*! 
         * \brief dst {st} lhs {op} scalar
         * \param dst destination
         * \param lhs left operand
         * \param scalar scalar operand
         * \tparam ST storage type apex_exp_template::enums::StoreMethod
         * \tparam OP bianry operator from apex_exp_template::enums::BinaryOperator
         * \sa _TensorXD apex_exp_template::enums
         */
        template<typename ST,typename OP>
        _INLINE void scalar_map( _TensorXD &dst, const _TensorXD &lhs, TENSOR_FLOAT scalar  );               
        
        /*! 
         * \brief dst {st} lhs {op} rhs
         * \param dst destination
         * \param lhs left operand
         * \param rhs left operand
         * \tparam ST storage type apex_exp_template::enums::StoreMethod
         * \tparam OP bianry operator from apex_exp_template::enums::BinaryOperator
         * \sa _TensorXD apex_exp_template::enums
         */
        template<typename ST,typename OP>
        _INLINE void binary_map( _TensorXD &dst, const _TensorXD &lhs, const _TensorXD &rhs  );               

        /*! 
         * \brief dst {st} lhs*sa + rhs*sb
         * \param dst destination
         * \param lhs left operand
         * \param rhs left operand
         * \param sa  left scalar
         * \param sb  right scalar 
         * \tparam ST storage type apex_exp_template::enums::StoreMethod
         * \sa _TensorXD apex_exp_template::enums
         */
        template<typename ST>
        _INLINE void scale_add( _TensorXD &dst, const _TensorXD &lhs, const _TensorXD &rhs, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                       
    };         
    
    namespace tensor{
        /*! 
         * \brief mean = sigmoid( energy )
         * \param mean destination
         * \param energy energy to be mapped
         * \sa _TensorXD 
         */
        _INLINE void sigmoid( _TensorXD &mean, const _TensorXD &energy );

        /*! 
         * \brief dst = sqrt( src )
         * \param dst destination
         * \param src src to be mapped
         * \sa _TensorXD 
         */
        _INLINE void map_sqrt( _TensorXD &dst, const _TensorXD &src );

        /*! 
         * \brief sample binary distribution from prob
         * \param state binary state sampled
         * \param prob probability
         * \sa _TensorXD 
         */
        _INLINE void sample_binary( _TensorXD &state, const _TensorXD &prob );
        
        /*! 
         * \brief sample state = normal( mean, sd )
         * \param state gaussian state sampled
         * \param mean mean value of gaussian
         * \param sd standard deviation of gaussian
         * \sa _TensorXD 
         */
        _INLINE void sample_gaussian( _TensorXD &state, const _TensorXD &mean, TENSOR_FLOAT sd );
        
        /*! 
         * \brief sample state = normal( 0, sd )
         * \param state gaussian state sampled
         * \param sd standard deviation of gaussian
         * \sa _TensorXD 
         */
        _INLINE void sample_gaussian( _TensorXD &state, TENSOR_FLOAT sd ); 

        /*! 
         * \brief sample state = uniform[ 0, 1 )
         * \param state  uniform state sampled         
         * \sa _TensorXD 
         */
        _INLINE void sample_uniform( _TensorXD &state ); 
    };
};

/*------------------------------------------------^-^----------------------*
 * function declaration end here
 *-------------------------------------------------------------------------*/

/**
 * implement common functions 
 */
namespace apex_tensor{
    inline _TensorXD & _TensorXD::operator=( TENSOR_FLOAT s ){
        apex_tensor::tensor::fill( *this, s );
        return *this;        
    }
};
/** 
 * the following codes remaps expression template solver to functions define above
 */
namespace apex_exp_template{
    namespace solver_impl{        
        /*! \brief specilize implementation of ScalarMapSolver for _TensorXD */
        template<typename ST,typename OP>
        struct ScalarMapSolver<ST,OP,_TensorXD,double>{
            /*! \brief redirect dst [st] src [op] scalar */
            static inline void eval( _TensorXD &dst, const _TensorXD &src, double scalar ){
                apex_tensor::tensor::scalar_map<ST,OP>( dst, src, (apex_tensor::TENSOR_FLOAT)scalar );
            }
        };

        /*! \brief specilize implementation of BinaryMapSolver for _TensorXD */
        template<typename ST,typename OP>
        struct BinaryMapSolver<ST,OP,_TensorXD,_TensorXD,_TensorXD>{
            /*! \brief redirect dst [st] lhs [op] rhs */
            static inline void eval( _TensorXD &dst, const _TensorXD &lhs, const _TensorXD &rhs ){
                apex_tensor::tensor::binary_map<ST,OP>( dst, lhs, rhs );
            }
        };

        /*! \brief specilize implementation of ScaleAddSolver for _TensorXD */
        template<typename ST>
        struct ScaleAddSolver<ST,_TensorXD,double>{
            /*! \brief redirect dst [st] lhs*sa + rhs*sb */
            static inline void eval( _TensorXD &dst, const _TensorXD &lhs, const _TensorXD &rhs, double sa, double sb ){
                apex_tensor::tensor::scale_add<ST>( dst, lhs, rhs, (apex_tensor::TENSOR_FLOAT)sa, (apex_tensor::TENSOR_FLOAT)sb );
            }
        };

        /*! \brief specilize impLementation of CloneSolver for _TensorXD */
        template<typename Src>
        struct CloneSolver<_TensorXD,Src>{
            /*! \brief implement dst = clone( src ) */
            static inline void eval( _TensorXD &dst, const Src &src ){
                dst.copy_param( src );
                apex_tensor::tensor::alloc_space( dst );
                apex_tensor::tensor::copy( dst, src );
            }
        };

        /*! \brief specilize implementation of AllocLikeSolver for _TensorXD */
        template<typename Src>
        struct AllocLikeSolver<_TensorXD,Src>{
            /*! \brief implement dst = alloc_like( src ) */
            static inline void eval( _TensorXD &dst, const Src &src ){
                dst.copy_param( src );
                apex_tensor::tensor::alloc_space( dst );
            }
        };
    };
};
