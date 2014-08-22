#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include "apex_svd.h"

#include "hmc-bpmf/bpmf_sampler.h"


namespace apex_svd{
    // return corresponding sub-solvers according to extend type
    ISVDTrainer *create_svd_trainer( SVDTypeParam mtype ){
       return new BPMFSampler(mtype);
    }
    ISVDRanker *create_svd_ranker( SVDTypeParam mtype ){
        return NULL;
    }
};
