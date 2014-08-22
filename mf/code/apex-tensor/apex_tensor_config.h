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

#ifndef _APEX_TENSOR_CONFIG_H_
#define _APEX_TENSOR_CONFIG_H_
/*!
 * \file apex_tensor_config.h
 * \brief this file is the configure file of apex tensor library
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */

#ifndef __APEX_TENSOR_DOUBLE_PRECISION__
/*! \brief the default precision of tensor */
#define __APEX_TENSOR_DOUBLE_PRECISION__ 0
#endif

#ifndef __APEX_TENSOR_USE_SSE__
/*! \brief whether to use SSE to speed up vector computation */
#define __APEX_TENSOR_USE_SSE__   1
#endif 

#ifndef __APEX_TENSOR_USE_BLAS__
/*! \brief whether to use BLAS to speed up matrix computation */
#define __APEX_TENSOR_USE_BLAS__   0
#endif 

// accuracy of tensor float
namespace apex_tensor{
#if __APEX_TENSOR_DOUBLE_PRECISION__
    /*! \brief storage type of apex-tensor */
    typedef double TENSOR_FLOAT;
#else
    /*! \brief storage type of apex-tensor */
    typedef float TENSOR_FLOAT;
#endif
};

#endif
