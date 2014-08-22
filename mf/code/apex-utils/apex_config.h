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
 */

#ifndef _APEX_CONFIG_H_
#define _APEX_CONFIG_H_
#define _CRT_SECURE_NO_WARNINGS

#include "apex_utils.h"
#include <cstdio>

/* simple helper for load in configure files */
namespace apex_utils{    
    /* load in config file */
    class ConfigIterator{
    private:
        FILE *fi;        
        char ch_buf;
        char s_name[256],s_val[256],s_buf[246];

        inline void skip_line(){           
            do{
               ch_buf = fgetc( fi );
            }while( ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r' );
        }
        
        inline void parse_str( char tok[] ){
            int i = 0; 
            while( (ch_buf = fgetc(fi)) != EOF ){
                switch( ch_buf ){
                case '\\': tok[i++] = fgetc( fi ); break;
                case '\"': tok[i++] = '\0'; 
						return;                        
                case '\r':
                case '\n': apex_utils::error("unterminated string"); break;
                default: tok[i++] = ch_buf;
                }
            }
            apex_utils::error("unterminated string"); 
        }
        // return newline 
        inline bool get_next_token( char tok[] ){
            int i = 0;
            bool new_line = false; 
            while( ch_buf != EOF ){
                switch( ch_buf ){
                case '#' : skip_line(); new_line = true; break;
                case '\"':
                    if( i == 0 ){
                        parse_str( tok );ch_buf = fgetc(fi); return new_line;
                    }else{
                        apex_utils::error("token followed directly by string"); 
                    }
                case '=':
					if( i == 0 ) {
						ch_buf = fgetc( fi );     
                        tok[0] = '='; 
                        tok[1] = '\0'; 
                    }else{
                        tok[i] = '\0'; 
                    }
					return new_line;
                case '\r':
                case '\n':
					if( i == 0 ) new_line = true;
                case '\t':
                case ' ' :
                    ch_buf = fgetc( fi );
                    if( i > 0 ){
                        tok[i] = '\0'; 
                        return new_line;
                    }               
					break;
                default: 
                    tok[i++] = ch_buf;
                    ch_buf = fgetc( fi );
                    break;                    
				}
			}
			return true;
		}

    public:
        ConfigIterator( const char *fname ){
            fi = apex_utils::fopen_check( fname, "r");
            ch_buf = fgetc( fi );
        }
        ~ConfigIterator(){
            fclose( fi );
        }
        inline const char *name()const{
            return s_name;
        }
        inline const char *val() const{
            return s_val;
        }
        inline bool next(){            
            while( !feof( fi ) ){
                get_next_token( s_name );

                if( s_name[0] == '=')  return false;               
				if( get_next_token( s_buf ) || s_buf[0] != '=' ) return false;			   				
				if( get_next_token( s_val ) || s_val[0] == '=' ) return false;
                return true;
            }
            return false;
        }        
    };            
};

#include <string>

namespace apex_utils{
    /*! \brief a class that save configs temporally and allows to get them out later */
    class ConfigSaver{
    private:
        std::vector<std::string> names;
        std::vector<std::string> values;
        std::vector<std::string> names_high;
        std::vector<std::string> values_high;        
        size_t idx;
    public:
        ConfigSaver( void ){ idx = 0; }
        inline void clear( void ){
            idx = 0;
            names.clear(); values.clear();
            names_high.clear(); values_high.clear();
        }
        inline void push_back( const char *name, const char *val ){
            names.push_back( std::string( name ) );
            values.push_back( std::string( val ) );                        
        }
        inline void push_back_high( const char *name, const char *val ){
            names_high.push_back( std::string( name ) );
            values_high.push_back( std::string( val ) );                        
        }
        inline void before_first( void ){
            idx = 0;
        }
        inline bool next( void ){
            if( idx >= names.size() + names_high.size() ){
                return false;
            }
            idx ++;
            return true;
        }
        inline const char *name( void ) const{
            apex_utils::assert_true( idx > 0, "can't call name before first");
            size_t i = idx - 1;
            if( i >= names.size() ){
                return names_high[ i - names.size() ].c_str();
            }else{
                return names[ i ].c_str();
            }
        }
        inline const char *val( void ) const{
            apex_utils::assert_true( idx > 0, "can't call name before first");
            size_t i = idx - 1;
            if( i >= values.size() ){
                return values_high[ i - values.size() ].c_str();
            }else{
                return values[ i ].c_str();
            }
        }
    };
};
#endif
