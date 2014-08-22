/*!
 * \file bpmf_sampler.h 
 * \brief parameter sampler of BPMF
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#ifndef _BPMF_SAMPLER_H_
#define _BPMF_SAMPLER_H_

#include "../apex_svd.h"
#include "../apex_svd_model.h"
#include <cstring>
#include <cmath>

#include "bpmf_base.h"

namespace apex_svd{
    class BPMFSampler: public BPMFBase{
    public:
        BPMFSampler( const SVDTypeParam &mtype )
            :BPMFBase( mtype ){
            printf("BPMF-Sampler\n");
        }
        virtual ~BPMFSampler( void ){
        }
    protected:
        inline float get_lambda_output( void ) const{
            return pmodel.mtype.active_type <= 1 ? param.lambda_output : param.weight_output;
        }
    protected:
        // SGD-HMC training parameter
        struct SHMCTrainParam{
            // base learning_rate
            float base_eta;
            // base momentum
            float base_mdecay;
            // decay schedule parameter a
            float a;
            // decay schedule parameter b
            float b;
            // delta eta decay schedule parameter
            float d_eta;
            // decay eta schedule parameter
            float d_mdecay;
            // current time counter
            float timer;
            // derived momentum value
            float momentum;
            // derived learning_rate
            float eta;
            // derived momentum decay
            float mdecay;
            SHMCTrainParam( void ){
                base_eta = 0.01f;
                base_mdecay = 0.1f;
                a = 1.0f;
                b = 1.0f;
                d_eta = 0.0f;
                d_mdecay = 0.0f;
            }
            inline void set_param( const char *name, const char *val ){
                if( !strcmp( name, "hmc:eta" ) )       base_eta = (float)atof(val);
                if( !strcmp( name, "hmc:mdecay" ) )    base_mdecay = (float)atof(val);
                if( !strcmp( name, "hmc:a" ) )         a = (float)atof(val);
                if( !strcmp( name, "hmc:b" ) )         b = (float)atof(val);                
                if( !strcmp( name, "hmc:d_eta" ) )     d_eta = (float)atof(val);
                if( !strcmp( name, "hmc:d_mdecay" ) )  d_mdecay = (float)atof(val); 
            }
            inline void init( void ){
                this->init_round( 0, 1.0f );
                printf("SHMC: init momentum=%f, eta=%f\n", expf( -mdecay ), eta );
            }
            // initialize round
            // eta scale: scale of eta in turns of training set            
            inline void init_round( float timer, float etascale ){
                this->timer  = timer;
                this->eta    = base_eta * a * pow( b + timer, d_eta ) * etascale; 
                this->mdecay = base_mdecay * a * pow( b + timer, d_mdecay );
                this->momentum = expf( - mdecay );
            }
        };
        // HMC parameters
        SHMCTrainParam shparam;
    public:
        virtual void set_param( const char *name, const char *val ){
            BPMFBase::set_param( name, val );
            shparam.set_param( name, val );
        }
        virtual void init_trainer( void ){
            BPMFBase::init_trainer();
            shparam.init();
        }
    protected:
        // ... direct update method ...
        // Stochastic HMC method
        inline void update_shmc_direct( const std::vector<Entry> &dtrain ){
            shparam.init_round( param.get_timer(), 1.0f/dtrain.size() ); 
            param.cur_sweight = sqrtf( shparam.eta );

            size_t sptr = 0;
            const float lambda_output = get_lambda_output();
            const float rscale = - ((float)param.batch_size) / (dtrain.size() *lambda_output);
            const float mscale = shparam.eta * (float)dtrain.size() * lambda_output / param.batch_size;
            const float sigma = sqrtf( 2 * param.temp * shparam.eta * shparam.mdecay );
            
            for( int step = 0; step < param.nstep; step ++ ){
                // reset gradient
                dmodel.W_user = 0.0f; dmodel.W_item = 0.0f; dmodel.ui_bias = 0.0f;
                for( int t = 0; t < param.batch_size; t ++, sptr = (sptr+1) % dtrain.size() ){
                    const Entry &e = dtrain[ sptr ];
                    const float p = this->pred( e.uid, e.iid );
                    const float g = active_type::cal_grad( e.label, p, pmodel.mtype.active_type );
                    dmodel.W_user[ e.uid ] += g * model.W_item[ e.iid ];
                    dmodel.W_item[ e.iid ] += g * model.W_user[ e.uid ];
                    dmodel.u_bias[ e.uid ] += g;
                    dmodel.i_bias[ e.iid ] += g;
                }
                // regularization
                dmodel.W_user += param.lambda_user * rscale * model.W_user;
                dmodel.W_item += param.lambda_item * rscale * model.W_item;
                dmodel.u_bias += param.lambda_ubias * rscale * model.u_bias;
                for( int i = 0; i < model.param.num_item; i ++ ){
                    dmodel.i_bias[i] += param.lambda_ibias * rscale * model.i_bias[i];
                }

                // update momentum
                model.p_W_user *= shparam.momentum;
                model.p_W_item *= shparam.momentum;
                for( int i = 0; i < model.param.num_user; i ++ ){     
                    model.p_u_bias[i] *= shparam.momentum;
                }
                for( int i = 0; i < model.param.num_item; i ++ ){
                    model.p_i_bias[i] *= shparam.momentum;
                }

                // add dW * eta
                model.p_W_user += dmodel.W_user * mscale;
                model.p_W_item += dmodel.W_item * mscale;
                for( int i = 0; i < model.param.num_user; i ++ ){                
                    model.p_u_bias[i] += dmodel.u_bias[i] * mscale;
                }
                for( int i = 0; i < model.param.num_item; i ++ ){
                    model.p_i_bias[i] += dmodel.i_bias[i] * mscale;
                }

                // add noise
                if( param.sample_weight() ){
                    #pragma omp parallel for schedule( static, param.omp_chunk_update )
                    for( int i = 0; i < model.param.num_user; i ++ ){                
                        const int tid = omp_get_thread_num();
                        twspace[tid]->sample_normal( model.p_W_user[i], model.p_W_user[i], sigma );
                        model.p_u_bias[i] += twspace[tid]->sample_normal() * sigma;
                    }
                    #pragma omp parallel for schedule( static, param.omp_chunk_update )
                    for( int i = 0; i < model.param.num_item; i ++ ){                
                        const int tid = omp_get_thread_num();
                        twspace[tid]->sample_normal( model.p_W_item[i], model.p_W_item[i], sigma );
                        model.p_i_bias[i] += twspace[tid]->sample_normal() * sigma;
                    }
                }

                // update weight
                model.W_user += model.p_W_user;
                model.W_item += model.p_W_item;
                for( int i = 0; i < model.param.num_user; i ++ ){                
                    model.u_bias[i] += model.p_u_bias[i];
                }
                for( int i = 0; i < model.param.num_item; i ++ ){
                    model.i_bias[i] += model.p_i_bias[i];
                }
            }
        }
    protected:
        // Stochastic Langevin Dynamics method
        inline void update_sgd_direct( const std::vector<Entry> &dtrain ){
            shparam.init_round( param.get_timer(), 1.0f/dtrain.size() ); 
            param.cur_sweight = shparam.eta;
            size_t sptr = 0;
            const float lambda_output = get_lambda_output();
            const float rscale = - ((float)param.batch_size) / (dtrain.size() *lambda_output);
            const float mscale = shparam.eta * (float)dtrain.size() * lambda_output / param.batch_size;
            const float sigma = sqrtf( 2 * param.temp * shparam.eta );
            
            for( int step = 0; step < param.nstep; step ++ ){
                // reset gradient
                dmodel.W_user = 0.0f; dmodel.W_item = 0.0f; dmodel.ui_bias = 0.0f;
                for( int t = 0; t < param.batch_size; t ++, sptr = (sptr+1) % dtrain.size() ){
                    const Entry &e = dtrain[ sptr ];
                    const float p = this->pred( e.uid, e.iid );
                    const float g = active_type::cal_grad( e.label, p, pmodel.mtype.active_type );
                    dmodel.W_user[ e.uid ] += g * model.W_item[ e.iid ];
                    dmodel.W_item[ e.iid ] += g * model.W_user[ e.uid ];
                    dmodel.u_bias[ e.uid ] += g;
                    dmodel.i_bias[ e.iid ] += g;
                }
                // regularization
                dmodel.W_user += param.lambda_user * rscale * model.W_user;
                dmodel.W_item += param.lambda_item * rscale * model.W_item;
                dmodel.u_bias += param.lambda_ubias * rscale * model.u_bias;
                for( int i = 0; i < model.param.num_item; i ++ ){
                    dmodel.i_bias[i] += param.lambda_ibias * rscale * model.i_bias[i];
                }

                // add dW * eta
                model.p_W_user = dmodel.W_user * mscale;
                model.p_W_item = dmodel.W_item * mscale;
                for( int i = 0; i < model.param.num_user; i ++ ){                
                    model.p_u_bias[i] = dmodel.u_bias[i] * mscale;
                }
                for( int i = 0; i < model.param.num_item; i ++ ){
                    model.p_i_bias[i] = dmodel.i_bias[i] * mscale;
                }

                // add noise
                if( param.sample_weight() ){
                    #pragma omp parallel for schedule( static, param.omp_chunk_update )
                    for( int i = 0; i < model.param.num_user; i ++ ){                
                        const int tid = omp_get_thread_num();
                        twspace[tid]->sample_normal( model.p_W_user[i], model.p_W_user[i], sigma );
                        model.p_u_bias[i] += twspace[tid]->sample_normal() * sigma;
                    }
                    #pragma omp parallel for schedule( static, param.omp_chunk_update )
                    for( int i = 0; i < model.param.num_item; i ++ ){                
                        const int tid = omp_get_thread_num();
                        twspace[tid]->sample_normal( model.p_W_item[i], model.p_W_item[i], sigma );
                        model.p_i_bias[i] += twspace[tid]->sample_normal() * sigma;
                    }
                }

                // update weight
                model.W_user += model.p_W_user;
                model.W_item += model.p_W_item;
                for( int i = 0; i < model.param.num_user; i ++) {
                    model.u_bias[i] += model.p_u_bias[i];
                }
                for( int i = 0; i < model.param.num_item; i ++ ){
                    model.i_bias[i] += model.p_i_bias[i];
                }
            }
        }
    protected:                
        virtual void update_direct( const SVDBPMFModel &umodel,
                                    std::vector<Entry> &dtrain ){
            BPMFBase::update_direct( umodel, dtrain );
            switch( param.sample_method ){
            case 1: update_sgd_direct( dtrain ); return;
            case 2: update_shmc_direct( dtrain ); return;
            default: apex_utils::error("unknown sample method");
            }                
        }        
    };
};

#endif
