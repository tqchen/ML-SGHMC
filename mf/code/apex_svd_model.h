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
 * \file apex_svd_model.h
 * \brief this file provide the default data structure of model and training parameters,
 *  note that we don't restrict to this form of model and training parameter,
 *  any extension can be done so long as ISVDTrainer interface is obeyed.
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */
#ifndef _APEX_SVD_MODEL_H_
#define _APEX_SVD_MODEL_H_

#include <cstdio>
#include <cstdlib>
#include "apex-utils/apex_utils.h"
#include "apex-tensor/apex_tensor.h"

/*! \brief namespace for matrix data structures and operations */
namespace apex_tensor{
    // basic documentation for tensor structure
    /*!       
     * \class apex_tensor::CTensor1D
     * \brief data structure for 1D vector, can be understanded as float array like float arr[x_max]
     */
    /*!       
     * \class apex_tensor::CTensor2D
     * \brief data structure for 2D matrix, can be understanded as 2D array like float arr[y_max][x_max]
     */
};

namespace apex_svd{
    /*! 
     * \brief namespace for solver type, with corresponding input format 
     */    
    namespace svd_type{
        /*! \brief basic solver using random ordered input */
        const int RANDOM_ORDER_FORMAT = 0;
        /*! \brief solver using user grouped input */
        const int USER_GROUP_FORMAT = 1;
        /*! \brief try to auto detect format type according to extend_type */
        const int AUTO_DETECT = 2;
    };
    /*! 
     * \brief namespace for active function and loss function type, as well as common calculation functions
     */    
    namespace active_type{
        /*! \brief identity map, square loss */
        const int LINEAR             = 0;
        /*! \brief sigmoid function, square loss */
        const int SIGMOID_L2         = 1;
        /*! \brief sigmoid function, log-likelihood loss */
        const int SIGMOID_LIKELIHOOD = 2;
        /*! 
         * \brief sigmoid function, log-likelihood loss. this option is reserved for rank 
         *   during prediction, the model will ouput the result 
         *   before sigmoid transformation, which is sufficient for rank
         */
        const int SIGMOID_RANK       = 3;
        /*! \brief smoothed hinge loss, used in MMMF */
        const int HINGE_SMOOTH       = 5;
        /*! \brief truncated L2 loss */
        const int HINGE_L2           = 6;
        /*! \brief sigmoid but return 1/4 as sgrad */
        const int SIGMOID_QSGRAD     = 7;

        /*! \brief z^2 */
        inline float sqr( float z ){
            return z * z;
        }        
        /*! 
         * \brief calculate the gradient of smoothed hinge loss for given prediction when label=1 
         * \param z the prediction
         * \return the gradient of smoothed hinge loss when label=1
         */
        inline float smooth_hinge_grad( float z ){
            if( z > 1.0f ) return  0.0f;
            if( z < 0.0f ) return  1.0f;
            return 1.0f - z;
        }
        /*! 
         * \brief calculate the smoothed hinge loss
         * \param z the prediction
         * \return the smoothed hinge loss when label=1
         */
        inline float smooth_hinge_loss( float z ){
            if( z > 1.0f ) return  0.0f;
            if( z < 0.0f ) return  0.5f - z;
            return 0.5f * sqr( 1.0f - z );            
        }
        /*! 
         * \brief map the input by the activation function
         *    can be used by the solver for calculation
         * \param sum the input data
         * \param type type of loss and activation function
         * \return the output mapped from input by the activation function
         */
        inline float map_active( float sum, int type ){
            switch( type ){
            case LINEAR: return sum;
            case SIGMOID_L2:
            case SIGMOID_LIKELIHOOD      : return 1.0f/(1.0f+expf(-sum ));
            case SIGMOID_RANK : return sum;
            case HINGE_SMOOTH : return sum;
            case HINGE_L2     : return sum;
            case SIGMOID_QSGRAD: return sum;
            default:apex_utils::error("unkown active type"); return 0.0f;
            }                
        } 
        /*! 
         * \brief calculate the gradiant of obj to maximize given the prediction and true label
         *    can be used by the solver during training calculation
         * \param r true label of the data
         * \param pred predicted value 
         * \param type type of activation function and loss
         * \return gradient value 
         */        
        inline float cal_grad( float r, float pred, int type ){
            switch( type ){
            case LINEAR: return r - pred;
            case SIGMOID_L2: return ( r - pred ) * pred * ( 1 - pred ); 
            case SIGMOID_LIKELIHOOD: return r - pred;
            case SIGMOID_QSGRAD    : 
            case SIGMOID_RANK      : return r - 1.0f/(1.0f+expf(-pred));
            case HINGE_SMOOTH : 
                {
                    if( r > 0.5f ) return smooth_hinge_grad( pred-0.5f );
                    else return - smooth_hinge_grad( 0.5f - pred );
                }
            case HINGE_L2:
                {
                    if( r > 0.5f ){
                        if( pred > 1.0f ) return 0.0f;
                        else return r - pred;
                    }else{
                        if( pred < 0.0f ) return 0.0f;
                        else return r - pred;
                    }
                }
            default:apex_utils::error("unkown active type"); return 0.0f;
            }                
        }
        /*! 
         * \brief calculate the loss function 
         * \param r true label of the data
         * \param pred predicted value 
         * \param type type of activation function and loss
         * \return loss function value
         */        
        inline float calc_loss( float r, float pred, int type  ){
            switch( type ){
            case active_type::LINEAR: 
            case active_type::SIGMOID_L2: return 0.5 * sqr( r- pred );
            case active_type::SIGMOID_QSGRAD:
            case active_type::SIGMOID_RANK: pred = 1.0f / (1.0f+expf( - pred ));
            case active_type::SIGMOID_LIKELIHOOD: return - r * logf( pred ) - (1.0f-r)*logf( pred );
            case active_type::HINGE_SMOOTH : {
                pred -= 0.5f;
                if( r > 0.5f ){
                    return active_type::smooth_hinge_loss( pred );
                }else{                    
                    return - active_type::smooth_hinge_loss( -pred );
                }
            }
            case active_type::HINGE_L2:{
                if( r > 0.5f ){
                    if( pred > 1.0f ) pred = 1.0f;
                    return 0.5 * sqr(1.0f-pred);
                }else{
                    if( pred < 0.0f ) pred = 0.0f;
                    return 0.5 * sqr(pred);
                }                
            }                
            default:apex_utils::error("unkown active type"); return 0.0f;
            }
        } 
        /*! 
         * \brief calculate the second order gradiant of obj to maximize given the prediction and true label
         *    can be used by the solver during training calculation,
         *    return 1.0 for hinge loss
         * \param r true label of the data
         * \param pred predicted value 
         * \param type type of activation function and loss
         * \return second order gradient value 
         */
        inline float cal_sgrad( float r, float pred, int type ){
            switch( type ){
            case LINEAR: return - 1.0f;
            case SIGMOID_L2: return - 0.25f;
            case SIGMOID_LIKELIHOOD: return - pred * ( 1.0f - pred );
            case SIGMOID_RANK      : {
                pred = 1.0f/(1.0f+expf(-pred));
                return - pred * ( 1.0f - pred );
            }
            case HINGE_SMOOTH  :
            case HINGE_L2      : return - 1.0f;
            case SIGMOID_QSGRAD: return - 0.25;
            default:apex_utils::error("unkown second order gradient for active type"); return 0.0f;
            }                
        }
        /*! 
         * \brief reverse transform by activation function
         * \param base_score orignal base_score set by output level
         * \param type type of activation function and loss
         * \return the transformed base_score
         */        
        inline float calc_base_score( float base_score, int type ){
            // scale base score by inverse of activation function
            switch( type ){                
            case LINEAR: 
            case HINGE_L2:
            case HINGE_SMOOTH: return base_score;
            case SIGMOID_L2: 
            case SIGMOID_LIKELIHOOD: 
            case SIGMOID_RANK:
            case SIGMOID_QSGRAD:
                {
                    apex_utils::assert_true( base_score > 0.0f && base_score < 1.0f, "sigmoid range constrain" );
                    return - logf( 1.0f / base_score - 1.0f );
                    break;
                }
            default: apex_utils::error("unkown active type"); return 0.0f;
            }
        }
    };
    /*! 
     * \brief type of the solver, used to decide which solver to use
     */
    struct SVDTypeParam{
        /*! 
         * \brief type of input to use
         * \sa svd_type
         */            
        uint8_t format_type;
        /*! 
         * \brief type of type of activation function and loss
         * \sa active_type
         */            
        uint8_t active_type;
        /*! \brief reserved parameter, can use to to specify new solver */
        uint8_t extend_type;
        /*! \brief reserved parameter */
        uint8_t variant_type;
        /*! \brief constructor, set default values */
        SVDTypeParam( void ){
            format_type = svd_type::AUTO_DETECT;
            active_type = extend_type = variant_type = 0; 
        }
        /*! 
         * \brief set parameters
         * \param name name of the parameter
         * \param val  value of the parameter
         */
        inline void set_param( const char *name, const char *val ){
            // compatible with previous versions
            if( !strcmp("model_type", name ) )    format_type  = (uint8_t)atoi( val );
            if( !strcmp("format_type", name ) )   format_type  = (uint8_t)atoi( val );
            if( !strcmp("active_type", name ) )   active_type  = (uint8_t)atoi( val );
            if( !strcmp("extend_type", name ) )   extend_type  = (uint8_t)atoi( val );
            if( !strcmp("variant_type", name ) )  variant_type = (uint8_t)atoi( val );
        }
        /*! 
         * \brief decide which format to use, 
         * \param fmt suggested format by outside
         */        
        inline void decide_format( int fmt = svd_type::AUTO_DETECT ){
            if( format_type != svd_type::AUTO_DETECT ) return;
            format_type = fmt;
            if( format_type != svd_type::AUTO_DETECT ) return;
            // use simple rule to suggest according to extend type 
            // most extended solvers need user grouped format
            format_type = ( extend_type == 0 ? svd_type::RANDOM_ORDER_FORMAT : svd_type::USER_GROUP_FORMAT );
        }
    };   
    /*! 
     * \brief default training parameters for SVDFeature
     */
    struct SVDTrainParam{        
        /*! \brief learning rate */
        float learning_rate;
        /*! 
         * \brief whether to decay learning rate during training, default=0 
         * \sa decay_rate
         */
        int decay_learning_rate;
        /*! 
         * \brief learning rate decay rate, only valid when decay_learning_rate=1
         *   for every round, learning_rate will be timed by decay_rate
         */
        float decay_rate;       
        /*! 
         * \brief min learning rate allowed during decay
         */        
        float min_learning_rate;
        /*! 
         * \brief max learning rate allowed during decay
         */        
        float max_learning_rate;
        /*! \brief weight decay of user, for regularization */
        float wd_user;
        /*! \brief weight decay of item, for regularization */
        float wd_item;
        /*! \brief weight decay of user bias, for regularization */
        float wd_user_bias;
        /*! \brief weight decay of item bias, for regularization */
        float wd_item_bias;               
        /*! \brief regularization method for latent factors */
        int reg_method;
        /*! \brief weight decay for global bias, for regularization */
        float wd_global;
        /*! \brief regularization method for global bias */
        int   reg_global;
        /*! \brief the first num_regfree_global global parameters are free from regularization, default=0 */
        unsigned  num_regfree_global;
        //-- parameters for implict/explict feedback, learning_rate = learning_rate * scale_lr_ufeedback
        /*! \brief learning_rate = learning_rate * scale_lr_ufeedback for user feedback parameters */
        float scale_lr_ufeedback;
        /*! \brief weight decay that comes in per user */      
        float wd_ufeedback_user;
        /*! \brief weight decay for user feedback factors, for regularization */
        float wd_ufeedback;
        /*! \brief weight decay for user feedback bias, for regularization */
        float wd_ufeedback_bias;       
        /*! \brief constructor, set the default values */
        SVDTrainParam( void ){
            learning_rate = 0.01f;
            reg_method = 0;           
            wd_user = wd_item = 0.0f;           
            wd_user_bias = wd_item_bias = 0.0f;

            num_regfree_global = 0; reg_global = 0; wd_global = 0.0f;             
            decay_learning_rate = 0; decay_rate = 1.0f; min_learning_rate = 0.0f;

            scale_lr_ufeedback = 1.0; wd_ufeedback = wd_ufeedback_user = wd_ufeedback_bias = 0; 
            max_learning_rate = 1.0f;
        }
        /*! 
         * \brief set parameters
         * \param name name of the parameter
         * \param val  value of the parameter
         */
        inline void set_param( const char *name, const char *val ){
            if( !strcmp("learning_rate", name ) )    learning_rate = (float)atof( val );
            if( !strcmp("wd_user", name ) )          wd_user = (float)atof( val );
            if( !strcmp("wd_item", name ) )          wd_item = (float)atof( val );
            if( !strcmp("wd_uiset", name ) )         wd_user = wd_item = (float)atof( val );
            if( !strcmp("wd_user_bias", name ) )          wd_user_bias = (float)atof( val );
            if( !strcmp("wd_item_bias", name ) )          wd_item_bias = (float)atof( val );
            if( !strcmp("wd_uiset_bias", name ) )         wd_user_bias = wd_item_bias = (float)atof( val );
            if( !strcmp("wd_global", name ) )             wd_global = (float)atof( val );
            if( !strcmp("reg_method", name ) )            reg_method  = atoi( val );
            if( !strcmp("reg_global", name ) )            reg_global  = atoi( val );
            if( !strcmp("num_regfree_global", name ) )    num_regfree_global  = (unsigned)atoi( val );
            if( !strcmp("decay_learning_rate" , name ) )  decay_learning_rate = atoi( val );
            if( !strcmp("min_learning_rate" , name ) )    min_learning_rate = (float)atof( val );
            if( !strcmp("max_learning_rate" , name ) )    max_learning_rate = (float)atof( val );
            if( !strcmp("decay_rate" , name ) )           decay_rate = (float)atof( val );
            if( !strcmp("scale_lr_ufeedback" , name ) )   scale_lr_ufeedback = (float)atof( val );
            if( !strcmp("wd_ufeedback" , name ) )         wd_ufeedback = (float)atof( val );
            if( !strcmp("wd_ufeedback_bias" , name ) )    wd_ufeedback_bias = (float)atof( val );
        }        
    };
    /*! 
     * \brief default model parameters for SVDFeature
     */
    struct SVDModelParam{
        /*! \brief number of user */
        int num_user;
        /*! \brief number of item */
        int num_item;
        /*! \brief number of factor */
        int num_factor;
        /*! \brief number of global feature */
        int num_global;
        /*! \brief initialize std variance for user factor */
        float u_init_sigma;
        /*! \brief initialize std variance for item factor */
        float i_init_sigma;        
        /*! \brief global mean of prediction */
        float base_score;
        /*! \brief not include user bias into model, used in rank-setting */        
        int no_user_bias;
        /*! \brief number of user feedback info */
        int num_ufeedback;
        /*! \brief initialize std variance for user feedback parameters */
        float ufeedback_init_sigma;
        /*! 
         * \brief number of first num_randinit_ufactor user factors will be randomly initialized,
         *    the remaining factors are left as 0.
         *    this paramter won't take effect if equals 0,i.e all the factors will be randomly initialized
         *    this is not a very important parameters,
         *    can be used to only random initialize basic user factor,
         *    while leave other user factors( such as time dependent user factor ) started as 0
         */
        int num_randinit_ufactor;
        /*! 
         * \brief number of first num_randinit_ifactor item factors will be randomly initialized,
         *    the remaining factors are left as 0.
         *    this paramter won't take effect if equals 0,i.e all the factors will be randomly initialized
         *    this is not a very important parameters
         *  \sa num_randinit_ufactor
         */
        int num_randinit_ifactor;
        /*! 
         * \brief whether user feature and item feature use common index space 
         *    this option can be set to 1 in learning to match(metric learning) setting. 
         *    For example: s(i,j) = dot( p_i, p_j ) = dot( p_j, p_i )
         */
        int common_latent_space;
        
        /*! \brief whether to only allow nonnegative user factors */
        int user_nonnegative;
        
        /*! 
         * \brief whether user feature and user feedback feature use common index space 
         */
        int common_feedback_space;
        
        /*! \brief extension flag indicating whether there is other model file following up */
        int extend_flag;
        
        
        /*! \brief whether to only allow nonnegative item factors */
        int item_nonnegative;

        /*! \brief reserved fields */
        int reserved[247];
        /*! \brief constructor, set the default values */
        SVDModelParam( void ){
            num_user = num_item = num_global = num_factor = 0;
            u_init_sigma = i_init_sigma = 0.01f; 
            no_user_bias = 0;
            base_score = 0.5f; 
            num_ufeedback = 0; 
            ufeedback_init_sigma = 0.0f; 
            num_randinit_ufactor = num_randinit_ifactor = 0;
            common_latent_space = 0;
            user_nonnegative = 0;
            item_nonnegative = 0;
            common_feedback_space = 0;
            extend_flag = 0;
            memset( reserved, 0, sizeof(reserved) );
        }
        /*! 
         * \brief set parameters
         * \param name name of the parameter
         * \param val  value of the parameter
         */
        inline void set_param( const char *name, const char *val ){
            if( !strcmp("num_user"  , name ) )    num_user     = atoi( val );       
            if( !strcmp("num_item"  , name ) )    num_item     = atoi( val );
            if( !strcmp("num_uiset"  , name ) )   num_user = num_item = atoi( val );
            if( !strcmp("num_global", name ) )    num_global   = atoi( val );       
            if( !strcmp("num_factor" , name ) )   num_factor   = atoi( val );       
            if( !strcmp("u_init_sigma", name ) )  u_init_sigma = (float)atof( val );
            if( !strcmp("i_init_sigma", name ) )  i_init_sigma = (float)atof( val );
            if( !strcmp("ui_init_sigma"  , name ) ) u_init_sigma = i_init_sigma = (float)atof( val );
            if( !strcmp("base_score", name ) )    base_score   = (float)atof( val );
            if( !strcmp("no_user_bias", name ) )  no_user_bias = atoi( val );
            if( !strcmp("num_ufeedback", name ) ) num_ufeedback  = atoi( val );
            if( !strcmp("num_randinit_ufactor", name ) )  num_randinit_ufactor = atoi( val );
            if( !strcmp("num_randinit_ifactor", name ) )  num_randinit_ifactor = atoi( val );
            if( !strcmp("num_randinit_uifactor", name ) ) num_randinit_ifactor = num_randinit_ufactor = atoi( val );
            if( !strcmp("ufeedback_init_sigma", name ) ) ufeedback_init_sigma = (float)atof( val );
            if( !strcmp("common_latent_space", name ) )  common_latent_space = atoi( val );
            if( !strcmp("common_feedback_space", name ) )  common_feedback_space = atoi( val );
            if( !strcmp("user_nonnegative", name ) )  user_nonnegative = atoi( val );
            if( !strcmp("item_nonnegative", name ) )  item_nonnegative = atoi( val );
        }                
    };
    /*! 
     * \brief default model for SVDFeature
     */    
    struct SVDModel{
        /*! \brief type of the solver */
        SVDTypeParam  mtype;
        /*! \brief model parameters */ 
        SVDModelParam param;
        /*! \brief whether the space is allocated */
        int space_allocated;
        /*! \brief common bias space for user/item feature */
        apex_tensor::CTensor1D ui_bias;
        /*! \brief common latent factor parameter for user/item feature */
        apex_tensor::CTensor2D W_uiset;        
        /*! \brief user bias */
        apex_tensor::CTensor1D u_bias;
        /*! \brief user latent factor */
        apex_tensor::CTensor2D W_user;
        /*! \brief item bias */
        apex_tensor::CTensor1D i_bias;
        /*! \brief item latent factor */
        apex_tensor::CTensor2D W_item;
        /*! \brief global bias */
        apex_tensor::CTensor1D g_bias;
        /*! \brief user feedback bias */
        apex_tensor::CTensor1D ufeedback_bias;
        /*! \brief user feedback latent factor */
        apex_tensor::CTensor2D W_ufeedback;        
        /*! \brief constructor */
        SVDModel( void ){
            space_allocated = 0;
        }
        /*! \brief allocated space for a given model parameter */
        inline void alloc_space( void ){
            {// allocate space for user/item factor
                const int ustart = ( param.common_feedback_space == 0 && mtype.format_type == svd_type::USER_GROUP_FORMAT ) ? param.num_ufeedback : 0; 

                if( param.common_latent_space == 0 ){ 
                    ui_bias.set_param( ustart + param.num_user + param.num_item );
                    W_uiset.set_param( ustart + param.num_user + param.num_item, param.num_factor );
                }else{
                    apex_utils::assert_true( param.num_user == param.num_item, 
                                             "num_user and num_item must be the same to use common latent space" );
                    apex_utils::assert_true( param.common_feedback_space != 0, "common latent space must enforce common feedback space" );
                    ui_bias.set_param( param.num_item );
                    W_uiset.set_param( param.num_item, param.num_factor );                
                }
                                
                apex_tensor::tensor::alloc_space( ui_bias );
                apex_tensor::tensor::alloc_space( W_uiset );


                if( param.common_latent_space == 0 ){                         
                    u_bias = ui_bias.sub_area( ustart, param.num_user );
                    W_user = W_uiset.sub_area( ustart, 0, param.num_user, param.num_factor );
                    i_bias = ui_bias.sub_area( param.num_user + ustart, param.num_item );
                    W_item = W_uiset.sub_area( param.num_user + ustart, 0, param.num_item, param.num_factor );
                }else{
                    W_user = W_uiset.sub_area( ustart, 0, param.num_user, param.num_factor ); 
                    u_bias = ui_bias.sub_area( ustart, param.num_user ); 
                    W_item = W_user;
                    i_bias = u_bias;
                }
            }
            {
                g_bias.set_param( param.num_global );
                apex_tensor::tensor::alloc_space( g_bias );            
            }
            if( mtype.format_type == svd_type::USER_GROUP_FORMAT ){
                if( param.common_feedback_space == 0 ){
                    ufeedback_bias = ui_bias.sub_area( 0, param.num_ufeedback );
                    W_ufeedback = W_uiset.sub_area( 0, 0, param.num_ufeedback, param.num_factor );
                }else{
                    ufeedback_bias = u_bias;
                    W_ufeedback = W_user;
                }
            }
            space_allocated = 1; 
        }
        /*! \brief free space of the model */
        inline void free_space( void ){           
            if( space_allocated == 0 ) return;
            apex_tensor::tensor::free_space( ui_bias );
            apex_tensor::tensor::free_space( W_uiset );            
            apex_tensor::tensor::free_space( g_bias );

            space_allocated = 0;
        }
        /*! 
         * \brief load the model from binary file, no space allocation is needed before loading
         * \param fi pointer to input file
         */
        inline void load_from_file( FILE *fi ){            
            if( fread( &param, sizeof(SVDModelParam) , 1 , fi ) == 0 ){
                printf("error loading CF SVD model\n"); exit( -1 );
            }
            if( space_allocated != 0 ) this->free_space();
            this->alloc_space();
            {// handle for common latent space, a bit complex for compatible issue
                if( param.common_latent_space == 0 ){
                    apex_tensor::cpu_only::load_from_file( u_bias, fi, true );
                    apex_tensor::cpu_only::load_from_file( W_user, fi, true );
                    apex_tensor::cpu_only::load_from_file( i_bias, fi, true );
                    apex_tensor::cpu_only::load_from_file( W_item, fi, true );
                }else{
                    apex_tensor::cpu_only::load_from_file( ui_bias, fi, true );
                    apex_tensor::cpu_only::load_from_file( W_uiset, fi, true );
                }
            }
            {
                apex_tensor::cpu_only::load_from_file( g_bias, fi, true );
            }
            if( mtype.format_type == svd_type::USER_GROUP_FORMAT ){
                if( param.common_feedback_space == 0 ){
                    apex_tensor::cpu_only::load_from_file( ufeedback_bias, fi, true );
                    apex_tensor::cpu_only::load_from_file( W_ufeedback, fi, true );
                }
            }
            space_allocated = 1;
        }
        /*! 
         * \brief save the model to binary file
         * \param fo pointer to output file
         */
        inline void save_to_file( FILE *fo ) const{
            fwrite( &param, sizeof(SVDModelParam) , 1 , fo );
            {// handle for common user/item latent space, make it compatible with previous format
                if( param.common_latent_space == 0 ){
                    apex_tensor::cpu_only::save_to_file( u_bias, fo );
                    apex_tensor::cpu_only::save_to_file( W_user, fo );
                    apex_tensor::cpu_only::save_to_file( i_bias, fo );
                    apex_tensor::cpu_only::save_to_file( W_item, fo );
                }else{
                    apex_tensor::cpu_only::save_to_file( ui_bias, fo );
                    apex_tensor::cpu_only::save_to_file( W_uiset, fo );
                }
            }
            {
                apex_tensor::cpu_only::save_to_file( g_bias, fo );
            }
            if( mtype.format_type == svd_type::USER_GROUP_FORMAT ){
                if( param.common_feedback_space == 0 ){
                    apex_tensor::cpu_only::save_to_file( ufeedback_bias, fo );
                    apex_tensor::cpu_only::save_to_file( W_ufeedback, fo );
                }
            }
        }        
        /*! 
         * \brief random initialize the model parameters, 
         *   the space shall be allocated before calling this function
         */                        
        inline void rand_init( void ){
            ui_bias = 0.0f;
            g_bias  = 0.0f;
            param.base_score = active_type::calc_base_score( param.base_score, mtype.active_type );
            {// initialize ufactor
                apex_tensor::CTensor2D W_uinit;
                if( param.num_randinit_ufactor != 0 ){ 
                    W_uinit = W_user.sub_area( 0, 0, param.num_randinit_ufactor, W_user.x_max );
                }else{
                    W_uinit = W_user;
                }
                apex_tensor::tensor::sample_gaussian( W_uinit, param.u_init_sigma );
                // fix for non negative weight
                if( param.user_nonnegative ){
                    for( int y = 0; y < W_user.y_max; y ++ )
                        for( int x = 0; x < W_user.x_max; x ++ )
                            W_user[ y ][ x ] = fabsf( W_user[y][x] );
                }            
            }                
            // only need to initialize once in common latent space
            if( param.common_latent_space == 0 ){
                // initialize ifactor
                apex_tensor::CTensor2D W_iinit;
                if( param.num_randinit_ifactor != 0 ){ 
                    W_iinit = W_item.sub_area( 0, 0, param.num_randinit_ifactor, W_item.x_max );
                }else{
                    W_iinit = W_item;
                }
                apex_tensor::tensor::sample_gaussian( W_iinit, param.i_init_sigma );
                // fix for non negative weight
                if( param.item_nonnegative ){
                    for( int y = 0; y < W_iinit.y_max; y ++ )
                        for( int x = 0; x < W_iinit.x_max; x ++ )
                            W_iinit[ y ][ x ] = fabsf( W_iinit[y][x] );
                }
            }

            if( mtype.format_type == svd_type::USER_GROUP_FORMAT ){
                apex_tensor::tensor::sample_gaussian( W_ufeedback, param.ufeedback_init_sigma );
            }
        }        
    };    
};

#endif

