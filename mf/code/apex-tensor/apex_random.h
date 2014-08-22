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

#ifndef _APEX_RANDOM_H_
#define _APEX_RANDOM_H_

/*!
 * \file apex_random.h
 * \brief PRNG to support random number generation
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 *
 * Use standard PRNG from stdlib
 */

#include <cmath>
#include <cstdlib>

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int  uint32_t;
#else
#include <inttypes.h>
#endif

/*! namespace of PRNG */
namespace apex_random{
	/*! \brief seed the PRNG */
	inline void seed( uint32_t seed ){
		srand( seed );
	}
			
    /*! \brief return a real number uniform in [0,1) */
	inline double next_double(){
		return static_cast<double>( rand() ) / (static_cast<double>( RAND_MAX )+1.0);
	}
    /*! \brief return a real numer uniform in (0,1) */
    inline double next_double2(){
        return (static_cast<double>( rand() ) + 1.0 ) / (static_cast<double>(RAND_MAX) + 2.0);
    }
};

#include <vector>

namespace apex_random{
	/*! \brief return a random number */
	inline uint32_t next_uint32( void ){
        return (uint32_t)rand();
    }
	/*! \brief return a random number in n */
	inline uint32_t next_uint32( uint32_t n ){
		return (uint32_t) floor( next_double() * n ) ;
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
    /*! \brief return  x~N(mu,sigma^2) */
	inline double sample_normal( double mu, double sigma ){
		return sample_normal() * sigma + mu;
	}

	/*! \brief  return 1 with probability p, coin flip */
	inline int sample_binary( double p ){
		return next_double() <  p;  
	}

	/*! \brief  return distribution from Gamma( alpha, beta ) */
    inline double sample_gamma( double alpha, double beta ) {
        if ( alpha < 1.0 ) {
            double u;
            do {
                u = next_double();
            } while (u == 0.0);
            return sample_gamma(alpha + 1.0, beta) * pow(u, 1.0 / alpha);
        } else {
            double d,c,x,v,u;
            d = alpha - 1.0/3.0;
            c = 1.0 / sqrt( 9.0 * d );
            do {
                do {
                    x = sample_normal();
                    v = 1.0 + c*x;
                } while ( v <= 0.0 );
                v = v * v * v;
                u = next_double();
            } while ( (u >= (1.0 - 0.0331 * (x*x) * (x*x)))
                      && (log(u) >= (0.5 * x * x + d * (1.0 - v + log(v)))) );
            return d * v / beta;
        }
    }

    template<typename T>
    inline void exchange( T &a, T &b ){
        T c;
        c = a;
        a = b;
        b = c;
    }

    template<typename T>
    inline void shuffle( T *data, size_t sz ){
        if( sz == 0 ) return;
        for( uint32_t i = (uint32_t)sz - 1; i > 0; i-- ){
            exchange( data[i], data[ next_uint32( i+1 ) ] );
        } 
    }
    // random shuffle the data inside, require PRNG 
    template<typename T>
    inline void shuffle( std::vector<T> &data ){
        shuffle( &data[0], data.size() );
    }
};

#endif
