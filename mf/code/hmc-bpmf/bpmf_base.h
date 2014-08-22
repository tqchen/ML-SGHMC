/*!
 * \file bpmf_base.h 
 * \brief base implementation for BPMF, via Gibbs-sampler, ^-^ :)
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#ifndef _BPMF_BASE_H_
#define _BPMF_BASE_H_

#include "../apex_svd_model.h"
#include "../apex-utils/apex_matrix_csr.h"
#include <cstring>
#include <climits>
#include <omp.h>

namespace apex_svd{
    using namespace apex_tensor;
    using namespace apex_utils;
};

// BPMF implementation of SVDFeature
namespace apex_svd{
    /*! 
     * \brief default model for SVDFeature
     */    
    struct SVDBPMFModel{
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

        // momentum variables
        /*! \brief user bias */
        apex_tensor::CTensor1D p_u_bias;
        /*! \brief user latent factor */
        apex_tensor::CTensor2D p_W_user;
        /*! \brief item bias */
        apex_tensor::CTensor1D p_i_bias;
        /*! \brief item latent factor */
        apex_tensor::CTensor2D p_W_item;
        /*! \brief global bias */
        apex_tensor::CTensor1D p_g_bias;
        
        /*! \brief constructor */
        SVDBPMFModel( void ){
            space_allocated = 0;
        }
        /*! \brief allocated space for a given model parameter */
        inline void alloc_space( void ){
            {// allocate space for user/item factor
                const int ustart = 0;
                const int pstart = param.num_user + param.num_item;
                if( param.common_latent_space == 0 ){ 
                    ui_bias.set_param( (param.num_user + param.num_item)*2 );
                    W_uiset.set_param( (param.num_user + param.num_item)*2, param.num_factor );
                }

                apex_tensor::tensor::alloc_space( ui_bias );
                apex_tensor::tensor::alloc_space( W_uiset );

                if( param.common_latent_space == 0 ){                         
                    u_bias = ui_bias.sub_area( ustart, param.num_user );
                    W_user = W_uiset.sub_area( ustart, 0, param.num_user, param.num_factor );
                    i_bias = ui_bias.sub_area( param.num_user + ustart, param.num_item );
                    W_item = W_uiset.sub_area( param.num_user + ustart, 0, param.num_item, param.num_factor );
                    // momentum
                    p_u_bias = ui_bias.sub_area( pstart, param.num_user );
                    p_W_user = W_uiset.sub_area( pstart, 0, param.num_user, param.num_factor );
                    p_i_bias = ui_bias.sub_area( param.num_user + pstart, param.num_item );
                    p_W_item = W_uiset.sub_area( param.num_user + pstart, 0, param.num_item, param.num_factor );                    
                }
            }

            {
                g_bias.set_param( param.num_global );
                p_g_bias.set_param( param.num_global );
                apex_tensor::tensor::alloc_space( g_bias );
                apex_tensor::tensor::alloc_space( p_g_bias );
            }
            space_allocated = 1; 
        }
        /*! \brief free space of the model */
        inline void free_space( void ){           
            if( space_allocated == 0 ) return;
            apex_tensor::tensor::free_space( ui_bias );
            apex_tensor::tensor::free_space( W_uiset );            
            apex_tensor::tensor::free_space( g_bias );
            apex_tensor::tensor::free_space( p_g_bias );
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
                        
            apex_tensor::cpu_only::load_from_file( ui_bias, fi, true );
            apex_tensor::cpu_only::load_from_file( W_uiset, fi, true );
            apex_tensor::cpu_only::load_from_file( g_bias, fi, true );
            apex_tensor::cpu_only::load_from_file( p_g_bias, fi, true );

            space_allocated = 1;
        }
        /*! 
         * \brief save the model to binary file
         * \param fo pointer to output file
         */
        inline void save_to_file( FILE *fo ) const{
            fwrite( &param, sizeof(SVDModelParam) , 1 , fo );
            apex_tensor::cpu_only::save_to_file( ui_bias, fo );
            apex_tensor::cpu_only::save_to_file( W_uiset, fo );
            apex_tensor::cpu_only::save_to_file( g_bias, fo );
            apex_tensor::cpu_only::save_to_file( p_g_bias, fo );
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
            }
        }
        // get a model that reverses user and item factor 
        inline SVDBPMFModel get_item_model( void ) const{
            SVDBPMFModel m =  *this;
            m.u_bias = i_bias; m.p_u_bias = p_i_bias;
            m.i_bias = u_bias; m.p_i_bias = p_u_bias;
            m.W_user = W_item; m.p_W_user = p_W_item;
            m.W_item = W_user; m.p_W_item = p_W_user;
            return m;
        }        
    };
};

namespace apex_svd{
    // base implementation of BPMF
    class BPMFBase : public ISVDTrainer{
    protected:
        // workspace of each thread, for PRNG
        class ThreadWSpace{
        private:
            // random number seed
            unsigned rseed;
        public:
            // preconditionor
            CTensor2D tmp_factor;
        public:
            ThreadWSpace( int nfactor ){
                rseed = 0;
                tmp_factor.set_param( 4, nfactor );
                tensor::alloc_space( tmp_factor );
            }
            ~ThreadWSpace( void ){
                tensor::free_space( tmp_factor );
            }
            /*! \brief set random number seed */
            inline void seed( unsigned sd ){
                this->rseed = sd;
            }
            /*! \brief return a real number uniform in [0,1) */
            inline double next_double( void ){
                return static_cast<double>( rand_r( &rseed ) ) 
                    / (static_cast<double>( RAND_MAX )+1.0);
            }
            /*! \brief return a real numer uniform in (0,1) */
            inline double next_double2(){
                return (static_cast<double>( rand_r( &rseed ) ) + 1.0 ) 
                    / (static_cast<double>(RAND_MAX) + 2.0);
            }
            
            /*! \brief return  x~N(0,1) */
            inline double sample_normal(){
                double x,y,s;
                do{
                    x = 2 * next_double2() - 1.0;
                    y = 2 * next_double2() - 1.0;
                    s = x*x + y*y;
                }while( s >= 1.0 || s == 0.0 );
                
                return x * sqrt( -2.0 * log(s) / s ) ;
            }
            
            /*! \brief return iid x,y ~N(0,1) */
            inline void sample_normal2D( double &xx, double &yy ){
                double x,y,s;
                do{
                    x = 2 * next_double2() - 1.0;
                    y = 2 * next_double2() - 1.0;
                    s = x*x + y*y;
                }while( s >= 1.0 || s == 0.0 );
                double t = sqrt( -2.0 * log(s) / s ) ;
                xx = x * t; 
                yy = y * t;
            }
            
            inline void sample_normal( CTensor1D dst, float sd ){
                double rx, ry = 0.0;
                for( int i = 0; i < dst.x_max; i ++ ){                    
                    if( (i & 1) == 0 ){
                        this->sample_normal2D( rx, ry );
                        dst[ i ] = (TENSOR_FLOAT)( rx * sd );                     
                    }else{
                        dst[ i ] = (TENSOR_FLOAT)( ry * sd );
                    }
                }
            }

            inline void sample_normal( CTensor1D dst, CTensor1D mean, float sd ){
                double rx, ry = 0.0;
                for( int i = 0; i < dst.x_max; i ++ ){                    
                    if( (i & 1) == 0 ){
                        this->sample_normal2D( rx, ry );
                        dst[ i ] = (TENSOR_FLOAT)( mean[i] + rx * sd );                     
                    }else{
                        dst[ i ] = (TENSOR_FLOAT)( mean[i] + ry * sd );
                    }
                }
            }

            inline void sample_normal( CTensor1D dst, CTensor1D mean, 
                                       CTensor1D sd , float scale = 1.0f ){
                double rx, ry = 0.0;
                for( int i = 0; i < dst.x_max; i ++ ){                    
                    if( (i & 1) == 0 ){
                        this->sample_normal2D( rx, ry );
                        dst[ i ] = (TENSOR_FLOAT)( mean[i] + rx * sd[i] * scale );
                    }else{
                        dst[ i ] = (TENSOR_FLOAT)( mean[i] + ry * sd[i] * scale );
                    }
                }
            }
        };
    protected:
        struct BPMFTrainParam: SVDTrainParam{
            // sampling parameters 
            float lambda_user, lambda_ubias;
            float lambda_item, lambda_ibias;
            // prior on weight
            float prior_alpha, prior_beta;
            float prior_alpha_bias, prior_beta_bias;
            // output variance parameter
            float prior_alpha_output, prior_beta_output;
            // output precision/inverse variance
            float lambda_output;
            // output weight
            float weight_output;
            // inverse temperature during parameter sampling
            float temp;
            // meta parameters
            // number of burn in 
            int bpmf_num_burn;
            // round counter
            int round_counter;
            // start_sample
            int start_hsample;
            // when to fix hsample
            int fix_hsample;
            // start sample weight
            int start_wsample;
            // method of sampling
            int sample_method;
            // number of threads
            int nthread;
            // minibatch size
            int batch_size;
            // number of step to go
            int nstep;
            // chunk update
            int omp_chunk_update;
            // sum of sample weight
            float sum_sweight;
            // current sample weight
            float cur_sweight;
            BPMFTrainParam( void ){
                prior_alpha = prior_beta = 1.0f;
                prior_alpha_bias = prior_beta_bias = 1.0f;
                prior_alpha_output = prior_beta_output = 1.0f;
                start_hsample = 0; 
                start_wsample = 0;
                bpmf_num_burn = 1;
                sample_method = 2;
                temp = 1.0f;
                nthread = 1;
                batch_size = 1;
                nstep = 100;
                omp_chunk_update = 40;
                sum_sweight = 0.0f;
                cur_sweight = 1.0f;
                fix_hsample = INT_MAX;
            }
            inline float get_avg_lambda( void ) {
                // record avg value of model
                float lambda = 1.0f;
                if( round_counter > bpmf_num_burn ){
                    sum_sweight += cur_sweight;
                    lambda = cur_sweight / sum_sweight;
                }
                return lambda;
            } 
            inline float get_timer( void ){
                if( round_counter > bpmf_num_burn ){
                    return round_counter - bpmf_num_burn;
                }else{
                    return 0.0f;
                }
            } 
            inline void set_param( const char *name, const char *val ){
                if( !strcmp( "prior_alpha", name ) ) prior_alpha = (float)atof( val );
                if( !strcmp( "prior_beta", name ) )  prior_beta = (float)atof( val );
                if( !strcmp( "prior_alpha_bias", name ) ) prior_alpha_bias = (float)atof( val );
                if( !strcmp( "prior_beta_bias", name ) )  prior_beta_bias = (float)atof( val );
                if( !strcmp( "prior_alpha_output", name ) ) prior_alpha_output = (float)atof( val );
                if( !strcmp( "prior_beta_output", name ) )  prior_beta_output = (float)atof( val );
                if( !strcmp( "temp", name ) )       temp = (float)atof( val );
                if( !strcmp( "start_hsample", name ) )  start_hsample = atoi( val );
                if( !strcmp( "start_wsample", name ) )  start_wsample = atoi( val );
                if( !strcmp( "start_sample", name ) )   start_wsample = start_hsample = atoi( val );
                if( !strcmp( "num_burn", name ) ) bpmf_num_burn = atoi( val );
                if( !strcmp( "sample_method", name ) )  sample_method = atoi( val );
                if( !strcmp( "nthread", name ) )  nthread = atoi( val );
                if( !strcmp( "batch_size", name ) )   batch_size = atoi( val );  
                if( !strcmp( "omp_chunk_update", name ) )  omp_chunk_update = atoi( val );
                if( !strcmp( "nstep", name ) )      nstep = atoi( val );
                if( !strcmp( "fix_hsample", name ) )       fix_hsample = atoi( val );
                SVDTrainParam::set_param( name, val );
            }
            inline bool sample_hyper( void ) const{
                return round_counter > start_hsample && round_counter > 0; 
            }
            inline bool sample_weight( void ) const{
                return round_counter >= start_wsample;
            }
            inline bool skip_hsample( void ) const{
                return round_counter >= fix_hsample;
            }
        };
    protected:
        int init_end;
        // simple model used for current prediction
        SVDBPMFModel model;
        // model used to store sampling parameters of current point 
        SVDBPMFModel pmodel;
        // model used to store delta update
        SVDBPMFModel dmodel;
        // model that used to store mean parameters 
        SVDBPMFModel smodel;
        // training parameters
        BPMFTrainParam param;
        // thread workspace
        std::vector< ThreadWSpace* > twspace;
    protected:
        // number of rounds for burn in
        int bpmf_num_burn;
        // scale label
        float scale_label;
    private:
        CTensor1D tmp_ufactor, tmp_ifactor;
    public:
        BPMFBase( const SVDTypeParam &mtype ){
            pmodel.mtype = mtype;
            smodel.mtype = mtype;
            dmodel.mtype = mtype;
            this->init_end = 0;
            this->bpmf_num_burn = 1;
            this->scale_label   = 1.0f;
            printf("BPMFBase\n");
        }
        virtual ~BPMFBase(){
             pmodel.free_space();
             smodel.free_space();
             dmodel.free_space();
             if( init_end == 0 ) return;
             tensor::free_space( tmp_ufactor );
             tensor::free_space( tmp_ifactor );
             // twspace
             for( size_t i = 0; i < twspace.size(); i ++ ){
                 delete twspace[i];
             }
         }
     public:
         // model related interface
         virtual void set_param( const char *name, const char *val ){
             if( !strcmp( name, "scale_label") ) scale_label = (float)atof( val );
             if( !strcmp( name, "scale_score") ) scale_label = (float)atof( val );
             param.set_param( name, val );
             if( pmodel.space_allocated == 0 ){
                 pmodel.param.set_param( name, val );
             }
             if( smodel.space_allocated == 0 ){
                 smodel.param.set_param( name, val );
                 dmodel.param.set_param( name, val );
             }
         }
         // load model from file
         virtual void load_model( FILE *fi ) {
             pmodel.load_from_file( fi );
             smodel.load_from_file( fi );
             dmodel.free_space();
             dmodel.alloc_space();
             dmodel.rand_init();
             dmodel.W_uiset = 0.0f;
         }
         // save model to file
         virtual void save_model( FILE *fo ) {
             pmodel.save_to_file( fo );
             smodel.save_to_file( fo );
         }
         // initialize model by defined setting
         virtual void init_model( void ){
             pmodel.alloc_space();
             pmodel.rand_init();
             smodel.alloc_space();
             smodel.rand_init();
             smodel.W_uiset = 0.0f;
             dmodel.alloc_space();
             dmodel.rand_init();
             dmodel.W_uiset = 0.0f;
         }
         // initialize trainer before training 
         virtual void init_trainer( void ){
             tmp_ufactor = clone( pmodel.W_user[0] );
             tmp_ifactor = clone( pmodel.W_item[0] );
             this->model = pmodel;
             this->data_init = false;

             // set number of threads to be correct
             omp_set_num_threads( param.nthread );
             #pragma omp parallel
             {
                 apex_utils::assert_true( omp_get_num_threads() == param.nthread, 
                                          "unable to set nthread" );
             }
             for( int i = 0; i < param.nthread; i ++ ){
                 twspace.push_back( new ThreadWSpace( pmodel.param.num_factor ) );
                 twspace.back()->seed( i );
             }
             this->init_end = 1;
         }
     protected:
        inline float pred( unsigned uid, unsigned iid ){
            double sum = model.param.base_score + model.u_bias[ uid ] + model.i_bias[ iid ];
            sum += apex_tensor::cpu_only::dot( model.W_user[ uid ], model.W_item[ iid ] );
            return active_type::map_active( (float)sum, model.mtype.active_type );
        }
    protected:
        struct Entry{
            float label;
            unsigned uid, iid;
        };
    protected:
        // whether it has been initialized
        bool data_init;
        // training data, test data
        std::vector<Entry> dtrain, dtest, dtshuffle;
        // tmp prediction storage for test
        std::vector<float> pred_test, pred_train;
    protected:
        // regularization parameters for 
        float lambda_factor, lambda_bias;
    protected:
        inline double update_predmean( std::vector<float> &predmean, const std::vector<Entry> &data ){
            float lambda = this->get_avg_lambda();
            double sum_mse = 0.0;
            apex_utils::assert_true( predmean.size() == data.size() );
            for( size_t i = 0; i < data.size(); i ++ ){
                predmean[ i ] = predmean[ i ] * ( 1.0f - lambda ) + lambda * pred( data[i].uid, data[i].iid );
                double diff = predmean[i] - data[i].label;
                sum_mse += diff * diff;
            }
            if( data.size() == 0 ) return 0.0;
            return sqrt( sum_mse / data.size() ) * scale_label;
        }
        inline void update_eval( void ){
            this->model = this->pmodel;
            fprintf( stderr, "[%d] train-rmse:%f\ttest-rmse:%f\n", 
                     param.round_counter, 
                     update_predmean( pred_train, dtrain ), 
                     update_predmean( pred_test , dtest ) );            
        }
        inline double calc_mse( const std::vector<Entry> &data ){
            double sum_mse = 0.0;
            for( size_t i = 0; i < data.size(); i ++ ){
                double diff = pred( data[i].uid, data[i].iid ) - data[i].label;
                sum_mse += diff * diff;
            }
            return sum_mse;
        }
    protected:
        virtual float get_avg_lambda( void ){
            return param.get_avg_lambda();
        }
    private:
        // record mean model parameter 
        inline void avg_model( void ){
            const float lambda = this->get_avg_lambda();
            smodel.W_uiset *= (1.0f - lambda);
            smodel.ui_bias *= (1.0f - lambda);
            smodel.g_bias  *= (1.0f - lambda);
            smodel.W_uiset += lambda * pmodel.W_uiset;
            smodel.ui_bias += lambda * pmodel.ui_bias;
            smodel.g_bias  += lambda * pmodel.g_bias;
        }
        // make user major and item major CSR 
        inline void make_data( void ){
            pred_train.resize( dtrain.size(), 0.0f );
            pred_test.resize ( dtest.size() , 0.0f );
            dtshuffle = dtrain;
            this->data_init = true;
        }
    protected:
        virtual void update_direct( const SVDBPMFModel &umodel,
                                    std::vector<Entry> &dtshuffle ){
            // preset all the parameters to model
            this->model = umodel;
            apex_random::shuffle( dtshuffle ); 
        }
        // sample hyper parameters
        virtual void sample_hyper( float &lambda, float prior_alpha, float prior_beta, double psum_sqr, double psum_cnt ){
            if( !param.sample_hyper() ){
                lambda = prior_alpha / prior_beta;
            }else {
                float alpha = prior_alpha + 0.5*psum_cnt;
                float beta  = prior_beta + 0.5*psum_sqr;
                if( param.temp < 1e-6f ){
                    // MAP
                    alpha -= 1.0f;
                    if( alpha < 0.0f ) alpha = 0.0f; 
                    lambda = alpha / beta;
                }else{
                    lambda = apex_random::sample_gamma( alpha, beta );
                }
            }
        }
    private:
        inline static double normsqr( CTensor1D x ){
            double sum = 0.0f;
            for( int i = 0; i < x.x_max; i ++ )
                sum += x[i] * x[i];
            return sum;
        }
        inline static double normsqr( CTensor2D x ){
            double sum = 0.0f;
            for( int y = 0; y < x.y_max; y ++ )
                sum += apex_tensor::cpu_only::dot( x[y], x[y] );
            return sum;
        }    
    public:
        virtual void update( const SVDFeatureCSR::Elem &feature ){
            if( data_init ) return;
            apex_utils::assert_true( feature.num_ufactor == 1, "basic MF only allow uid" );
            apex_utils::assert_true( feature.num_ifactor == 1, "basic MF only allow uid" );
            Entry e; e.label = feature.label;
            e.uid = feature.index_ufactor[0];
            e.iid = feature.index_ifactor[0];
            if( feature.num_global == 0 ){
                dtrain.push_back( e );
            }else{
                dtest.push_back( e );
            }
        }
        virtual float predict( const SVDFeatureCSR::Elem &feature ){ 
            Entry e; e.label = feature.label;
            e.uid = feature.index_ufactor[0];
            e.iid = feature.index_ifactor[0];
            this->model = this->smodel;
            float p = this->pred(e.uid, e.iid);
            this->model = this->pmodel;
            return p;
        }
        virtual void set_round( int nround ){
            param.round_counter = nround;
        }
        virtual void finish_round( void ){
            if( !data_init ) this->make_data();
            if( !param.skip_hsample() ) {// sample hyper parameters
                this->sample_hyper( param.lambda_user, param.prior_alpha, param.prior_beta, 
                                    normsqr( pmodel.W_user ), pmodel.W_user.num_elem() );
                this->sample_hyper( param.lambda_item, param.prior_alpha, param.prior_beta,
                                    normsqr( pmodel.W_item ), pmodel.W_item.num_elem() );
                this->sample_hyper( param.lambda_ubias, param.prior_alpha_bias, param.prior_beta_bias, 
                                    normsqr( pmodel.u_bias ), pmodel.u_bias.num_elem() );
                this->sample_hyper( param.lambda_ibias, param.prior_alpha_bias, param.prior_beta_bias,
                                    normsqr( pmodel.i_bias ), pmodel.i_bias.num_elem() );
                // sample output lambda
                this->sample_hyper( param.lambda_output, param.prior_alpha_output, param.prior_beta_output,
                                    calc_mse( dtrain ), dtrain.size() );
                
            }
            this->update_direct( pmodel, dtshuffle );
            // average model
            this->avg_model();
            // update pred eval
            this->update_eval();
        }
    };
};
#endif
