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
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 * 
 * Acknowledgement: The MAC support part of this code is provided by Artemy Kolchinsky
 */

#ifndef _APEX_UTILS_H_
#define _APEX_UTILS_H_

#ifdef _MSC_VER
#define fopen64 fopen
#else

// use 64 bit offset, either to include this header in the beginning, or 
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#warning "FILE OFFSET BITS defined to be 32 bit"
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 fopen
#endif

#define _FILE_OFFSET_BITS 64
extern "C"{    
#include <sys/types.h>
};
#include <cstdio>
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace apex_utils{        
    inline void error( const char *msg ){
        fprintf( stderr, "%s\n",msg );
        exit( -1 );
    }
    
    inline void assert_true( bool exp ){
        if( !exp ) error( "assert_error" );
    }

    inline void assert_true( bool exp, const char *msg ){
        if( !exp ) error( msg );
    }

    inline void warning( const char *msg ){
        fprintf( stderr, "warning:%s\n",msg );
    }
    
    inline FILE *fopen_check( const char *fname , const char *flag ){
		FILE *fp = fopen64( fname , flag );
		if( fp == NULL ){
			fprintf( stderr, "can not open file \"%s\"\n",fname );
			exit( -1 );
		}
		return fp;
 	}
    
    
    inline void fseek_page( FILE *fp, size_t psize, size_t pos ){
#ifdef _MSC_VER
        fseek( fp, static_cast<long> ( psize * pos ), SEEK_SET );   
#else
        off_t off = static_cast<off_t>( psize ) * static_cast<off_t>( pos );
        fseeko( fp, off, SEEK_SET ); 
#endif
    }

    inline void system( const char *cmd ){
        int rt = ::system( cmd );
        apex_utils::assert_true( rt == 0 || rt != 0, "anyway..");
    }
};

namespace apex_utils{
    // running queue for schedulers
    class RunQueue{
    private:
        unsigned head, tail;
        std::vector<bool> in_queue;
        std::vector<int>  run_queue;
    public:
        inline void init( int nslaves ){
            in_queue.resize ( nslaves );
            run_queue.resize( nslaves + 1 );
            this->clear();
        }
        inline void clear( void ){
            head = tail = 0;
            std::fill( in_queue.begin(), in_queue.end(), false );
        }
        inline void put( int index ){
            if( in_queue[ index ] ) return;
            in_queue[ index ] = true;
            run_queue[ tail ] = index; 
            tail = ( tail + 1 ) % run_queue.size();
        }
        inline int get( void ){
            int idx = run_queue[ head ];
            in_queue[ idx ] = false;
            head = ( head + 1 ) % run_queue.size();
            return idx;
        }
        inline unsigned size( void ){
            return (unsigned)((head + run_queue.size() - tail) % run_queue.size());
        }
    };

    // simple set of integers
    class SimpleSet{
    private:
        std::vector<bool> flag;
        std::vector<int>  hist;
    public:
        inline void add( int id ){
            while( (int)flag.size() <= id ) flag.push_back( false );
            flag[ id ] = true;
            hist.push_back( id );
        }
        inline bool contain( int id ) const{
            return id < (int)flag.size() && flag[ id ];
        }
        inline void clear( void ){
            for( size_t i = 0; i < hist.size(); i ++ ){
                flag[ hist[i] ] = false;
            } 
            hist.clear();
        }
        inline size_t size( void ) const{
            return hist.size();
        }
    };    
};

namespace apex_utils{
    // simple feature array structure, can store feature text into main memory    
    namespace __sparse_feature_array{
        template<typename FValue>
        inline void load( FILE *fi, unsigned &index, FValue &value );
        template<>
        inline void load<float>( FILE *fi, unsigned &index, float &value ){
            apex_utils::assert_true( fscanf( fi,"%u:%f", &index, &value ) ==2 , "load sparse feature" );
        }
        template<>
        inline void load<unsigned>( FILE *fi, unsigned &index, unsigned &value ){
            apex_utils::assert_true( fscanf( fi,"%u:%u", &index, &value ) ==2 , "load sparse feature" );
        }
    };

    // storage of sparse feature
    template<typename FValue = float>
    class SparseFeatureArray{      
    public:
        struct Entry{
            unsigned   index;
            FValue     value;
        };
        struct Vector{            
            int num_elem;
            const Entry *ptr_elem;            
            inline int size() const{
                return num_elem;
            }
            inline const Entry &operator[]( int idx ) const{
                return ptr_elem[ idx ];
            }
        };
    private:
        unsigned num_row;
        std::vector<unsigned>  row_ptr;
        std::vector<Entry> data;
    public :
        SparseFeatureArray(){ clear(); }
        inline const Vector operator[]( unsigned idx )const{            
            Vector vec;
            if( idx < num_row ){
                vec.num_elem = row_ptr[ idx + 1 ] - row_ptr[ idx ];
                vec.ptr_elem = &data[ row_ptr[idx] ];
            }else{
                vec.num_elem = 0;
            }
            return vec;
        }
        inline void clear(){
            data.clear();
            row_ptr.clear();
            num_row = 0;
        }
        inline void load( const char *fname ){            
            this->clear();
            row_ptr.push_back( 0 );

            int n;
            Entry e; 
            FILE *fi = apex_utils::fopen_check( fname, "r" );

            while( fscanf( fi,"%d", &n ) == 1 ){
                row_ptr.push_back( row_ptr.back() + n ); 
                num_row ++;
                for( int i = 0; i < n; i ++ ){
                    __sparse_feature_array::load<FValue>( fi, e.index, e.value );
                    data.push_back( e );
                }
            } 
            fclose( fi );
        }
    };
};

#endif
