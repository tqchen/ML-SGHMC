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
 * \file apex_svd.h
 * \brief header defining the interface of SVD-solver 
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */
#ifndef _APEX_SVD_H_
#define _APEX_SVD_H_

#include "apex_svd_data.h"
#include "apex_svd_model.h"

/*! \brief namespace of SVDFeature */
namespace apex_svd{    
    /*! 
     * \brief virtual interface of SVDFeature-solver
     */
    class ISVDTrainer{
    public:
        // interface for model setting and loading
        // this allows different implementation to use their own model structure
        /*! 
         * \brief set parameters from outside 
         * \param name name of the parameter
         * \param val  value of the parameter
         */
        virtual void set_param( const char *name, const char *val ) = 0;
        /*! 
         * \brief load model from file
         * \param fi file pointer to input file
         */        
        virtual void load_model( FILE *fi ) = 0;
        /*! 
         * \brief save model file
         * \param fo file pointer to output file
         */        
        virtual void save_model( FILE *fo ) = 0;
        /*! 
         * \brief random initialize model parameters for first training,
         * a model can either be random initialized using init_model 
         * or be load by load_model
         */        
        virtual void init_model( void ) = 0;
        /*! 
         * \brief initialize solver before training, called before training
         * this function is reserved for solver to allocate necessary space and 
         * do other preparations 
         */        
        virtual void init_trainer( void ) = 0;
    public:        
        // interface for training procedure
        /*! 
         * \brief set current round number, 
         *   this function is before called every round, reserved for solver to set round dependent learning_rate, etc.
         * \param nround current round number
         */
        virtual void set_round( int nround ){ apex_utils::error("not implemented 3"); }
        /*! 
         * \brief tell the trainer that current round is finished
         *   this function is called after every round, reserved for solver to do post processing
         */
        virtual void finish_round( void ){}
        /*! 
         * \brief update model using feature vector, random order input
         * \param feature input feature
         * \sa SVDFeatureCSR
         */
        virtual void update( const SVDFeatureCSR::Elem &feature ){ apex_utils::error("not implemented 2"); }
        /*! 
         * \brief predict the rate for given feature 
         * \param feature input feature
         * \sa SVDFeatureCSR
         */
        virtual float predict( const SVDFeatureCSR::Elem &feature ){ apex_utils::error("not implemented 1"); return 0.0f; }
    public:
        // SVD++ user-wise style update, for user grouped input
        /*! 
         * \brief update model, user grouped input, used in efficient SVD++ training and rank
         * \param data input user grouped data
         * \sa SVDPlusBlock
         */
        virtual void update( const SVDPlusBlock &data ){ apex_utils::error("not implemented"); }
        /*! 
         * \brief predict for a given user grouped data
         * \param pred output of the predicted value
         * \param data input user grouped data
         * \sa SVDPlusBlock
         */
        virtual void predict( std::vector<float> &pred, const SVDPlusBlock &data ){ apex_utils::error("not implemented"); }
    public:
        virtual ~ISVDTrainer(){}
    };

    /*! 
     * \brief namespace for extension tag in for SVDRanker 
     *  use to specify different kinds of input information in the input feature,
     *  Special convention for Ranker input:
     *     label field: the tag information is specified in the label field
     *     item  index: some kinds of input need to specify item index, item index should be specified in the first index field of user feature
     */    
    namespace svdranker_tag{
        /*! 
         * \brief indicate this line of feature provides an candidate item for ranking, 
         *        the item field and global feature field should specify the features associated with this item,
         *        item set but be specified before all other input, 
         *        each item will be associated with a specific item index which is the order they occurs,
         *        the first item will be numbered 0, second will be numbered 1, etc..
         */
        const int ITEM_TAG   =  0;
        /*! 
         * \brief indicate this line specifies a start of a user section,
         *        user feature field must specify the user features of current user
         */
        const int USER_TAG   =  2;        
        /*! 
         * \brief indicate this line specifies a positive sample that we care about in rank evaluation for current user section,
         *        only item index shoule be specified (in user feature field), multiple item index can be specified
         */
        const int POS_SAMPLE =  1;
        /*! 
         * \brief indicate this line specifies a banned sample that should not be ranked for current user section, 
         *        only item index should be specified(in user feature field), multiple item index can be specified 
         */
        const int BAN_SAMPLE = -1;
        /*! 
         * \brief indicate this line specifies a special sample that may contains extra global feature that should be considered for current user-item pair,
         *        the most common usage of this tag is include the KNN information that's user-item related for those who has the information,
         *        item index shoule be specified(in user feature field), 
         *        item feature field can specify the extra item feature that should be consider, global feature field can specify the extra global feature that should be considered
         */
        const int SPEC_SAMPLE = 3;        
        /*! 
         * \brief indicate this line specifies the end of a user section, 
         *        start processing all the information, rank the candidate items for the given user, and output the result
         *        only tag is need, set all the features as empty,
         *        this tag is a bit duplicated with user tag, but should be included as convention
         */        
        const int PROCESS_TAG=  4;        
    };

    /*! 
     * \brief virtual interface of SVDFeature-ranking util, to rank large-set of items when it's not 
     *   convenient to specify all the rank candidate in a feature file
     */
    class ISVDRanker{
    public:
        /*! 
         * \brief load model from file
         * \param fi file pointer to input file
         */        
        virtual void load_model( FILE *fi ) = 0;
        /*! 
         * \brief initialize ranker before ranking
         * this function is reserved for ranker to allocate necessary space and do other preparations 
         * \param num_item_set the number of items in maximum that's needed to be ranked
         */        
        virtual void init_ranker( int num_item_set ) = 0;
        /*! 
         * \brief set parameters from outside 
         * \param name name of the parameter
         * \param val  value of the parameter
         */
        virtual void set_param( const char *name, const char *val ) = 0;
    public:
        /*! 
         * \brief process the input line of feature and output result if any
         * \param result the output result after process
         * \param feature input feature
         * \sa please refer to the reference manual for input convention for ranker
         */
        virtual void process( std::vector<int> &result, const SVDFeatureCSR::Elem &feature ){ apex_utils::error("not implemented"); }
        /*! 
         * \brief process the input line of feature and output result if any
         * \param result the output result after process
         * \param data input user grouped data
         * \sa please refer to the reference manual for input convention for ranker
         * \sa svdranker_tag
         */
        virtual void process( std::vector<int> &result, const SVDPlusBlock &data ){ apex_utils::error("not implemented"); }
    public:
        virtual ~ISVDRanker(){}
    };
};

namespace apex_svd{
    /*! 
     * \brief create a SVD trainer according to the specified type
     *  when make a new extension of the ISVDTrainer, rewrite this function to add the variant to extesion.
     *
     * Guide for customization: user can write(or modify from the provided implementation) a new version of 
     *  solver, and rewrite this function to return the new solver.
     *
     * \param mtype specify the type of the solver
     * \return a pointer to a solver
     * \sa SVDTypeParam
     */
    ISVDTrainer *create_svd_trainer( SVDTypeParam mtype );

    /*! 
     * \brief create a SVD ranker according to the specified type
     *  when make a new extension of the ISVDRankeer, rewrite this function to add the variant to extesion.
     *
     * \param mtype specify the type of the solver
     * \return a pointer to a solver
     * \sa SVDTypeParam
     */
    ISVDRanker *create_svd_ranker( SVDTypeParam mtype );
};
#endif

