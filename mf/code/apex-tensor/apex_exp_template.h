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

#ifndef _APEX_EXP_TEMPLATE_H_
#define _APEX_EXP_TEMPLATE_H_
/*!
 * \file apex_exp_template.h
 * \brief expression template to do lazy evaluation
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */

/*! \brief namespace of expression template */
namespace apex_exp_template{    
    /*! \brief namespace of operators defined in expression template 
     *  the user should using apex_exp_template::operators to enable all the operators
     */
    namespace operators{
    };
    /*! \brief namespace of solvers involved 
     *  the user should specialize the solver for each expressions
     */
    namespace solver_impl{
    };
    /*! \brief namespace of enumeration classes */
    namespace enums{
    };
};

namespace apex_exp_template{
    namespace enums{
        /*! 
         *\brief this class describes how to store result of calculation         
         *\sa SaveTo AddTo SubTo MulTo DivTo
         *
         * the detail storage method inheritate this class
         */
        template<typename Derived>
        class StoreMethod{
        protected:
            StoreMethod(){}
        };    
        /*! \brief operator= */
        class SaveTo: public StoreMethod<SaveTo> {
        private:
            SaveTo(){}
        public:
            /*! \brief singleton of SaveTo */
            static const SaveTo op(){
                return SaveTo();
            }
            /*! \brief string representation */
            static const char *str(){
                return "=";
            }
        };    
        
        /*! \brief operator+= */
        class AddTo : public StoreMethod<AddTo>{
        private:
            AddTo(){}
        public:
            /*! \brief singleton of AddTo */
            static const AddTo  op(){
                return AddTo();
            }
            /*! \brief string representation */
            static const char  *str(){
                return "+=";
            }
        };


        /*! \brief operator-= */
        class SubTo : public StoreMethod<SubTo>{
        private:
            SubTo(){}
        public:
            /*! \brief singleton of SubTo */
            static const SubTo  op(){
                return SubTo();
            }    
            /*! \brief string representation */
            static const char *str(){
                return "-=";
            }
        };


        /*! \brief operator*= */
        class MulTo : public StoreMethod<MulTo>{
        private:
            MulTo(){}
        public:
            /*! \brief singleton of MulTo */
            static const MulTo  op(){
                return MulTo();
            }
            /*! \brief string representation */
            static const char *str(){
                return "*=";
            }
        };

        /*! \brief operator/= */
        class DivTo : public StoreMethod<DivTo>{
        private:
            DivTo(){}
        public:
            /*! \brief singleton of MulTo */
            static const DivTo  op(){
                return DivTo();
            }
            /*! \brief string representation */
            static const char *str(){
                return "/=";
            }
        };
    };
    namespace enums{
        /*! 
         *\brief operator of binary operation 
         *\sa Add Sub Mul Div
         *
         * the detail operator inheritate this class
         */
        template<typename Derived>
        class BinaryOperator{
        protected:
            BinaryOperator(){}
        };    
        /*! \brief operator+ */
        class Add : public BinaryOperator<Add>{
        private:
            Add(){}
        public:
            /*! \brief singleton of Add */
            static const Add  op(){
                return Add();
            }
            /*! \brief string representation */
            static const char *str(){
                return "+";
            }
        };

        /*! \brief operator- */
        class Sub : public BinaryOperator<Sub>{
        private:
            Sub(){}
        public:
            /*! \brief singleton of Add */
            static const Sub  op(){
                return Sub();
            }
            /*! \brief string representation */
            static const char *str(){
                return "-";
            }
        };

        /*! \brief operator* */
        class Mul : public BinaryOperator<Mul>{
        private:
            Mul(){}
        public:
            /*! \brief singleton of Add */
            static const Mul  op(){
                return Mul();
            }
            /*! \brief string representation */
            static const char *str(){
                return "*";
            }
        };
        
        /*! \brief operator/ */
        class Div : public BinaryOperator<Div>{
        private:
            Div(){}
        public:
            /*! \brief singleton of Add */
            static const Div  op(){
                return Div();
            }
            /*! \brief string representation */
            static const char *str(){
                return "/";
            }
        };

    };
    namespace enums{
        /*! 
         *\brief method of convolution
         *\sa Valid Full
         */
        template<typename Derived>
        class ConvType{
        protected:
            ConvType(){}
        };
        /*! \brief method of convolution storage:valid */
        class Valid : public ConvType<Valid>{
        private:
            Valid(){}
        public:
            /*! \brief singleton of Valid */
            static const Valid  op(){
                return Valid();
            }
            /*! \brief string representation */
            static const char *str(){
                return "valid";
            }
        };

        /*! \brief method of convolution storage:full */
        class Full : public ConvType<Full>{
        private:
            Full(){}
        public:
            /*! \brief singleton of Valid */
            static const Full  op(){
                return Full();
            }
            /*! \brief string representation */
            static const char *str(){
                return "full";
            }
        };
    };
};

namespace apex_exp_template{
    template<typename Elem>
    class TransposeExp;
    template<typename Elem>
    class ReverseExp;

    /*! \brief base expression class of all expressions */
    template<typename Name, typename Alias>
    class Exp{
    protected:
        Exp(){}
    public:
        /*! 
         *\brief  return true derived class 
         *\return true derived class
         */
        inline Name &__name(){
            return *static_cast<Name*>(this);
        }
        /*! 
         *\brief  return true derived class 
         *\return true derived class
         */
        inline const Name &__name_const() const{
            return *static_cast<const Name*>(this);
        }
        /*! 
         *\brief  return alias derived class 
         * alias is used to identify the type of each class in composite
         *\return true derived class
         *\sa ContainerExp CompositeExp
         */
        inline const Alias & __alias_const() const{
            return *static_cast<const Alias*>(this);
        }   
        /*! 
         *\brief transpose of a matrix
         *\return transpose of current expression
         */
        inline const TransposeExp<Alias> T() const{
            return TransposeExp<Alias>( this->__alias_const() );
        }
        /*! 
         *\brief reverse of last two dimensions of matrix d.R[k][i][j] = d[k][-i][-j]
         *\return reverse of current expression
         */
        inline const ReverseExp<Alias> R() const{
            return ReverseExp<Alias>( this->__alias_const() );
        }
    };
    /*! \brief base class of all composite expressions
     *  this is a alias class, we use it to identify difference between 
     *  basic variables and composite expressions
     * \sa ContainerExp
     */
    template<typename Derived>
    class CompositeExp: public Exp< Derived, CompositeExp<Derived> >{
    protected:
        CompositeExp(){}
    public:
        /*! 
         *\brief evaluation src and store to dst 
         *\param st storage method
         *\param dst destination to be stored
         *\param src source expression
         *\sa StoreMethod
         */ 
        template<typename ST, typename Dst>
        inline void __eval( const enums::StoreMethod<ST> &st, Dst &dst, const Derived &src ) const; 
    };
    /*! \brief base class of all variables
     *  this is a alias class, we use it to identify difference between 
     *  basic variables and composite expressions
     * \sa CompositeExp
     */
    template<typename Derived>
    class ContainerExp: public Exp< Derived, ContainerExp<Derived> >{
    protected:
        ContainerExp(){}
    public:
        /*! \brief implementation of operator+= */
        inline Derived &operator+=( double s ){
            using namespace operators;
            this->__name() = this->__name_const() + s;
            return this->__name();
        }
        /*! \brief implementation of operator-= */
        inline Derived &operator-=( double s ){
            using namespace operators;
            this->__name() = this->__name_const() - s;
            return this->__name();
        }
        /*! \brief implementation of operator*= */
        inline Derived &operator*=( double s ){
            using namespace operators;
            this->__name() = this->__name_const() * s;
            return this->__name();
        }
        /*! \brief implementation of operator/= */
        inline Derived &operator/=( double s ){
            using namespace operators;
            this->__name() = this->__name_const() / s;
            return this->__name();
        }
    public:
        /*! \brief implementation of operator+= */
        template<typename T>
        inline Derived &operator+=( const ContainerExp<T> &exp ){
            using namespace operators;
            this->__name() = this->__name_const() + exp.__name_const();
            return this->__name();
        }
        /*! \brief implementation of operator-= */
        template<typename T>
        inline Derived &operator-=( const ContainerExp<T> &exp ){
            using namespace operators;
            this->__name() = this->__name_const() - exp.__name_const();
            return this->__name();
        }
        /*! \brief implementation of operator*= */
        template<typename T>
        inline Derived &operator*=( const ContainerExp<T> &exp ){
            using namespace operators;
            this->__name() = this->__name_const() * exp.__name_const();
            return this->__name();
        }
        /*! \brief implementation of operator/= */
        template<typename T>
        inline Derived &operator/=( const ContainerExp<T> &exp ){
            using namespace operators;
            this->__name() = this->__name_const() / exp.__name_const();
            return this->__name();
        }
    public:                
        /*! \brief implementation of operator= */
        template<typename T>
        inline Derived &__assign( const CompositeExp<T> &exp ){
            exp.__name_const().__eval( enums::SaveTo::op(), this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator+= */
        template<typename T>
        inline Derived &operator+=( const CompositeExp<T> &exp ){
            exp.__name_const().__eval( enums::AddTo::op(), this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator-= */
        template<typename T>
        inline Derived &operator-=( const CompositeExp<T> &exp ){
            exp.__name_const().__eval( enums::SubTo::op(), this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator*= */
        template<typename T>
        inline Derived &operator*=( const CompositeExp<T> &exp ){
            exp.__name_const().__eval( enums::MulTo::op(), this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator/= */
        template<typename T>
        inline Derived &operator/=( const CompositeExp<T> &exp ){
            exp.__name_const().__eval( enums::DivTo::op(), this->__name(), exp.__name_const() );
            return this->__name();
        }        
        
    };            
};

namespace apex_exp_template{
    /*! \brief transpose of a expression*/
    template<typename Elem>
    class TransposeExp: public CompositeExp< TransposeExp<Elem> >{
    public:
        /*! \brief expression to be transposed */
        const Elem &exp;
        /*! \brief constructor */
        TransposeExp( const Elem &e ):exp(e){}        
        inline const Elem & T() const{
            return exp;
        }
    };
    /*!
     *\brief reverse of a expression over last 2 dimensions 
     * d.R[k][i][i] = d[k][-i][-j] 
     */
    template<typename Elem>
    class ReverseExp: public CompositeExp< ReverseExp<Elem> >{
    public:
        /*! \brief expression to be reversed */
        const Elem &exp;
        /*! \brief constructor */
        ReverseExp( const Elem &e ):exp(e){}
        inline const Elem & R() const{
            return exp;
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to solve scale problem 
         * \tparam ST StoreMethod in enums
         * \tparam OP BinaryOperator in enums
         *
         * user must specialize the class to create specific solvers of types
         */
        template<typename ST,typename OP,typename T, typename TV>
        struct ScalarMapSolver{
            /*! \brief implement dst [st] src [op] scalar */
            static inline void eval( T &dst, const T &src, TV scalar );
        };
    };
    /*! \brief scale expression which represent exp op scalar */
    template<typename OP,typename Elem,typename TValue>
    class ScalarMapExp: public CompositeExp< ScalarMapExp<OP,Elem,TValue> >{
    public:        
        /*! brief expression parameter */
        const Elem &exp;
        /*! brief scalar parameter */
        TValue scalar;
        /*! \brief constructor */
        ScalarMapExp( const Elem &e, TValue s ):exp(e),scalar(s){}
        /*! \brief basic specialization of basic operation */
        template<typename ST,typename T>
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, const ScalarMapExp< OP,ContainerExp<T>, TValue > &src ) const{
            solver_impl::ScalarMapSolver<ST,OP,T,TValue>::eval( dst, src.exp.__name_const(), src.scalar );
        }
        /*! \brief expand map chain, use dst as temp storage */
        template<typename Dst,typename Src>
        inline void __eval( const enums::SaveTo &s, Dst &dst, const ScalarMapExp< OP,CompositeExp<Src>, TValue > &src ) const{
            dst = src.exp.__name_const();
            solver_impl::ScalarMapSolver<enums::SaveTo,OP,Dst,TValue>::eval( dst, dst, src.scalar );
        }
    };
    namespace operators{
        /*! \brief operator overload for scale */
        template<typename T,typename TT>
        inline const ScalarMapExp<enums::Mul,TT,double> operator*( const Exp<T,TT> &exp, double scalar ){
            return ScalarMapExp<enums::Mul,TT,double>( exp.__alias_const(), scalar ); 
        }
        /*! \brief operator overload for scale */
        template<typename T,typename TT>
        inline const ScalarMapExp<enums::Mul,TT,double> operator*( double scalar, const Exp<T,TT> &exp ){
            return exp *  scalar;
        }
        /*! \brief operator overload for scale */
        template<typename T,typename TT>
        inline const ScalarMapExp<enums::Mul,TT,double> operator/( const Exp<T,TT> &exp, double scalar ){
            return exp * (1.0/scalar); 
        }
        /*! \brief operator overload for scale */
        template<typename TT>
        inline const ScalarMapExp<enums::Mul,TT,double> operator*( const ScalarMapExp<enums::Mul,TT,double> &exp, double scalar ){
            return ScalarMapExp<enums::Mul,TT,double>( exp.exp, exp.scalar * scalar ); 
        }
        /*! \brief operator overload for scale */
        template<typename TT>
        inline const ScalarMapExp<enums::Mul,TT,double> operator*( double scalar, const ScalarMapExp<enums::Mul,TT,double> &exp ){
            return exp * scalar;
        }
        /*! \brief operator overload for scale */
        template<typename TT>
        inline const ScalarMapExp<enums::Mul,TT,double> operator/( const ScalarMapExp<enums::Mul,TT,double> &exp, double scalar ){
            return exp * (1.0/scalar);
        }
       
        /*! \brief operator overload for shift */
        template<typename T,typename TT>
        inline const ScalarMapExp<enums::Add,TT,double> operator+( const Exp<T,TT> &exp, double scalar ){
            return ScalarMapExp<enums::Add,TT,double>( exp.__alias_const(), scalar ); 
        }
        /*! \brief operator overload for shift */
        template<typename T,typename TT>
        inline const ScalarMapExp<enums::Add,TT,double> operator+( double scalar, const Exp<T,TT> &exp ){
            return exp +  scalar;
        }
        /*! \brief operator overload for shift */
        template<typename T,typename TT>
        inline const ScalarMapExp<enums::Add,TT,double> operator-( const Exp<T,TT> &exp, double scalar ){
            return exp + (-scalar); 
        }        
        /*! \brief operator overload for shift */
        template<typename TT>
        inline const ScalarMapExp<enums::Add,TT,double> operator+( const ScalarMapExp<enums::Add,TT,double> &exp, double scalar ){
            return ScalarMapExp<enums::Add,TT,double>( exp.exp, exp.scalar + scalar ); 
        }
        /*! \brief operator overload for shift */
        template<typename TT>
        inline const ScalarMapExp<enums::Add,TT,double> operator+( double scalar, const ScalarMapExp<enums::Add,TT,double> &exp ){
            return exp + scalar;
        }
        /*! \brief operator overload for shift */
        template<typename TT>
        inline const ScalarMapExp<enums::Add,TT,double> operator-( const ScalarMapExp<enums::Add,TT,double> &exp, double scalar ){
            return exp + (-scalar);
        } 
    };    
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to solve binary elementwise operation
         * \tparam ST StoreMethod in enums
         * \tparam OP BinaryOperator in enums
         *
         * user must specialize the class to create specific solvers of types
         */        
        template<typename ST,typename OP, typename Dst, typename Lhs, typename Rhs>
        struct BinaryMapSolver{
            /*! \brief implement dst [st] lhs [op] rhs */
            static inline void eval( Dst &dst, const Lhs &lhs, const Rhs &rhs );
        };
        /*! 
         * \brief solver interface to solve scaleadd
         * \tparam ST StoreMethod in enums
         *
         * user must specialize the class to create specific solvers of types
         */
        template<typename ST,typename T, typename TV>
        struct ScaleAddSolver{
            /*! \brief implement dst [st] a*sa + b*sb */
            static inline void eval( T &dst, const T &a, const T &b, TV sa, TV sb );
        };
    };
    /*! \brief elementwise binary operations */
    template<typename OP,typename Lhs,typename Rhs>
    class BinaryMapExp: public CompositeExp< BinaryMapExp<OP,Lhs,Rhs> >{
    public:
        /*! \brief left operand */
        const Lhs &lhs;
        /*! \brief right operand */
        const Rhs &rhs;
        /*! \brief constructor */
        BinaryMapExp( const Lhs &l, const Rhs &r ):lhs(l),rhs(r){}
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const BinaryMapExp<OP,ContainerExp<TA>, ContainerExp<TB> > &src ) const{
            solver_impl::BinaryMapSolver<ST,OP,Dst,TA,TB>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const() );
        }
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename T,typename TV> 
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, 
                            const BinaryMapExp<enums::Add, 
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > >, 
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > > > &src ) const{
            solver_impl::ScaleAddSolver<ST,T,TV>::eval( dst, 
                                                        src.lhs.__name_const().exp.__name_const(), 
                                                        src.rhs.__name_const().exp.__name_const(),
                                                        src.lhs.__name_const().scalar,
                                                        src.rhs.__name_const().scalar );
        }
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename T,typename TV> 
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, 
                            const BinaryMapExp<enums::Add, 
                            ContainerExp<T> ,
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > > > &src ) const{
            solver_impl::ScaleAddSolver<ST,T,TV>::eval( dst, 
                                                        src.lhs.__name_const(), 
                                                        src.rhs.__name_const().exp.__name_const(),
                                                        1.0,
                                                        src.rhs.__name_const().scalar );
        }
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename T,typename TV> 
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, 
                            const BinaryMapExp<enums::Add, 
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > >, 
                            ContainerExp<T> > &src ) const{
            solver_impl::ScaleAddSolver<ST,T,TV>::eval( dst, 
                                                        src.lhs.__name_const().exp.__name_const(), 
                                                        src.rhs.__name_const(),
                                                        src.lhs.__name_const().scalar,
                                                        1.0 );
        }        
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename T,typename TV> 
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, 
                            const BinaryMapExp<enums::Sub, 
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > >, 
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > > > &src ) const{
            solver_impl::ScaleAddSolver<ST,T,TV>::eval( dst, 
                                                        src.lhs.__name_const().exp.__name_const(), 
                                                        src.rhs.__name_const().exp.__name_const(),
                                                        src.lhs.__name_const().scalar,
                                                        -src.rhs.__name_const().scalar );
        }
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename T,typename TV> 
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, 
                            const BinaryMapExp<enums::Sub, 
                            ContainerExp<T> ,
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > > > &src ) const{
            solver_impl::ScaleAddSolver<ST,T,TV>::eval( dst, 
                                                        src.lhs.__name_const(), 
                                                        src.rhs.__name_const().exp.__name_const(),
                                                        1.0,
                                                        -src.rhs.__name_const().scalar );
        }
        /*! \brief basic specialization of binary calculation */
        template<typename ST,typename T,typename TV> 
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, 
                            const BinaryMapExp<enums::Sub, 
                            CompositeExp< ScalarMapExp< enums::Mul,ContainerExp<T>,TV > >, 
                            ContainerExp<T> > &src ) const{
            solver_impl::ScaleAddSolver<ST,T,TV>::eval( dst, 
                                                        src.lhs.__name_const().exp.__name_const(), 
                                                        src.rhs.__name_const(),
                                                        src.lhs.__name_const().scalar,
                                                        -1.0 );
        }        
    };
    namespace operators{
        /*! \brief operator overload for elementwise+ */
        template<typename TA, typename TB,typename TAA, typename TBB>
        inline const BinaryMapExp<enums::Add,TAA,TBB> operator+( const Exp<TA,TAA> &lhs, const Exp<TB,TBB> &rhs ){
            return BinaryMapExp<enums::Add,TAA,TBB>( lhs.__alias_const(), rhs.__alias_const() );
        }
        /*! \brief operator overload for elementwise- */
        template<typename TA, typename TB,typename TAA, typename TBB>
        inline const BinaryMapExp<enums::Sub,TAA,TBB> operator-( const Exp<TA,TAA> &lhs, const Exp<TB,TBB> &rhs ){
            return BinaryMapExp<enums::Sub,TAA,TBB>( lhs.__alias_const(), rhs.__alias_const() );
        }
        /*! \brief operator overload for elementwise* */
        template<typename TA, typename TB,typename TAA, typename TBB>
        inline const BinaryMapExp<enums::Mul,TAA,TBB> operator*( const Exp<TA,TAA> &lhs, const Exp<TB,TBB> &rhs ){
            return BinaryMapExp<enums::Mul,TAA,TBB>( lhs.__alias_const(), rhs.__alias_const() );
        }
        /*! \brief operator overload for elementwise/ */
        template<typename TA, typename TB,typename TAA, typename TBB>
        inline const BinaryMapExp<enums::Div,TAA,TBB> operator/( const Exp<TA,TAA> &lhs, const Exp<TB,TBB> &rhs ){
            return BinaryMapExp<enums::Div,TAA,TBB>( lhs.__alias_const(), rhs.__alias_const() );
        }
    };        
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to solve matrix multiplication
         * \tparam ST StoreMethod in enums
         * \tparam transposeLeft  whether to transpose lhs before dot
         * \tparam transposeRight whether to transpose rhs before dot
         *
         * user must specialize the class to create specific solvers of types
         */        
        template<typename ST, typename Dst, typename Lhs, typename Rhs, bool transposeLeft, bool transposeRight>
        struct DotSolver{
            /*! \brief implement dst [st] dot( lhs[.T],rhs[.T] ) */
            static inline void eval( Dst &dst, const Lhs &lhs, const Rhs &rhs );
        };        
    };
    /*! \brief matrix multiplication */
    template<typename Lhs,typename Rhs>
    class DotExp: public CompositeExp< DotExp<Lhs,Rhs> >{
    public:
        /*! \brief left operand */
        const Lhs &lhs;
        /*! \brief right operand */
        const Rhs &rhs;
        /*! \brief constructor */
        DotExp( const Lhs &l, const Rhs &r ):lhs(l),rhs(r){}
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const DotExp<ContainerExp<TA>, ContainerExp<TB> > &src ) const{
            solver_impl::DotSolver<ST,Dst,TA,TB,false,false>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const() );
        }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const DotExp<CompositeExp< TransposeExp< ContainerExp<TA> > >, ContainerExp<TB> > &src ) const{
            solver_impl::DotSolver<ST,Dst,TA,TB,true,false>::eval( dst, src.lhs.__name_const().exp.__name_const(), src.rhs.__name_const() );
        }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const DotExp<ContainerExp<TA>, CompositeExp< TransposeExp< ContainerExp<TB> > > > &src ) const{
            solver_impl::DotSolver<ST,Dst,TA,TB,false,true>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const().exp.__name_const() );
        }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, 
                            const DotExp<CompositeExp< TransposeExp< ContainerExp<TA> > >, CompositeExp< TransposeExp< ContainerExp<TB> > > > &src ) const{
            solver_impl::DotSolver<ST,Dst,TA,TB,true,true>::eval( dst, src.lhs.__name_const().exp.__name_const(), src.rhs.__name_const().exp.__name_const() );
        }        
    };
    
    namespace operators{
        /*! \brief operator overload for matrix multiplication */
        template<typename TA, typename TB,typename TAA, typename TBB>
        inline const DotExp<TAA,TBB> dot( const Exp<TA,TAA> &lhs, const Exp<TB,TBB> &rhs ){
            return DotExp<TAA,TBB>( lhs.__alias_const(), rhs.__alias_const() );
        }
    };

    namespace solver_impl{
        /*! 
         * \brief solver interface to solve matrix multiplication with scale
         * \tparam ST StoreMethod in enums
         * \tparam transposeLeft  whether to transpose lhs before dot
         * \tparam transposeRight whether to transpose rhs before dot
         *
         * user must specialize the class to create specific solvers of types
         */        
        template<typename ST, typename Dst, typename Lhs, typename Rhs, bool transposeLeft, bool transposeRight>
        struct ScaleDotSolver{
            /*! \brief implement dst [st] scale * dot( lhs[.T],rhs[.T] ) */
            static inline void eval( Dst &dst, const Lhs &lhs, const Rhs &rhs, double scale );
        };        
    };
    
    /*! \brief matrix multiplication */
    template<typename Lhs,typename Rhs>
    class ScaleDotExp: public CompositeExp< ScaleDotExp<Lhs,Rhs> >{
    public:
        /*! \brief scale parameter */
        double scale;
        /*! \brief left operand */
        const Lhs &lhs;
        /*! \brief right operand */
        const Rhs &rhs;
        /*! \brief constructor */
        ScaleDotExp( const Lhs &l, const Rhs &r, double s ):lhs(l),rhs(r){ this->scale = s; }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const ScaleDotExp<ContainerExp<TA>, ContainerExp<TB> > &src ) const{
            solver_impl::ScaleDotSolver<ST,Dst,TA,TB,false,false>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const(), scale );
        }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const ScaleDotExp<CompositeExp< TransposeExp< ContainerExp<TA> > >, ContainerExp<TB> > &src ) const{
            solver_impl::ScaleDotSolver<ST,Dst,TA,TB,true,false>::eval( dst, src.lhs.__name_const().exp.__name_const(), src.rhs.__name_const(), scale );
        }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const ScaleDotExp<ContainerExp<TA>, CompositeExp< TransposeExp< ContainerExp<TB> > > > &src ) const{
            solver_impl::ScaleDotSolver<ST,Dst,TA,TB,false,true>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const().exp.__name_const(), scale );
        }
        /*! \brief basic specialization of dot calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, 
                            const ScaleDotExp<CompositeExp< TransposeExp< ContainerExp<TA> > >, CompositeExp< TransposeExp< ContainerExp<TB> > > > &src ) const{
            solver_impl::ScaleDotSolver<ST,Dst,TA,TB,true,true>::eval( dst, src.lhs.__name_const().exp.__name_const(), src.rhs.__name_const().exp.__name_const(), scale );
        }        
    };
    
    namespace operators{
        /*! \brief operator overload for scale over dot*/
        template<typename TA,typename TB>
        inline const ScaleDotExp<TA,TB> operator*( const DotExp<TA,TB> &exp, double scalar ){
            return ScaleDotExp<TA,TB>( exp.lhs, exp.rhs, scalar ); 
        }
        /*! \brief operator overload for scale over dot */
        template<typename TA,typename TB>
        inline const ScaleDotExp<TA,TB> operator*( double scalar, const DotExp<TA,TB> &exp ){
            return exp * scalar;
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to 2D convolution
         * \tparam ST StoreMethod in enums
         * \tparam reverseLeft whether to reverse lhs first before conv2
         * \tparam reverseRight whether to reverse rhs first before conv2
         * \tparam CT ConvType in enums
         *
         * user must specialize the class to create specific solvers of types
         */        
        template<typename ST, typename Dst, typename Lhs, typename Rhs, bool reverseLeft, bool reverseRight, typename CT >
        struct Conv2Solver{
            /*! \brief implement dst [st] conv2( lhs[.R],rhs[.R], option )
             * option = 'V'(valid) or 'F'(full) or 'E'(equal) 
             */
            static inline void eval( Dst &dst, const Lhs &lhs, const Rhs &rhs );
        };        
    };

    /*! \brief 2D convolution  */
    template<typename Lhs,typename Rhs,typename CT>
    class Conv2Exp: public CompositeExp< Conv2Exp<Lhs,Rhs,CT> >{
    public:
        /*! \brief left operand */
        const Lhs &lhs;
        /*! \brief right operand */
        const Rhs &rhs;
        /*! \brief constructor */
        Conv2Exp( const Lhs &l, const Rhs &r ):lhs(l),rhs(r){}
        /*! \brief basic specialization of conv2 calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const Conv2Exp<ContainerExp<TA>, ContainerExp<TB>, CT > &src ) const{
            solver_impl::Conv2Solver<ST,Dst,TA,TB,false,false,CT>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const() );
        }
        /*! \brief basic specialization of conv2 calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const Conv2Exp<CompositeExp< ReverseExp< ContainerExp<TA> > >, ContainerExp<TB>, CT > &src ) const{
            solver_impl::Conv2Solver<ST,Dst,TA,TB,true,false,CT>::eval( dst, src.lhs.__name_const().exp.__name_const(), src.rhs.__name_const() );
        }
        /*! \brief basic specialization of conv2 calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const Conv2Exp<ContainerExp<TA>, CompositeExp< ReverseExp< ContainerExp<TB> > >, CT > &src ) const{
            solver_impl::Conv2Solver<ST,Dst,TA,TB,false,true,CT>::eval( dst, src.lhs.__name_const(), src.rhs.__name_const().exp.__name_const() );
        }
        /*! \brief basic specialization of conv2 calculation */
        template<typename ST,typename Dst, typename TA, typename TB> 
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, 
                            const Conv2Exp<CompositeExp< ReverseExp< ContainerExp<TA> > >, CompositeExp< ReverseExp< ContainerExp<TB> > >, CT > &src ) const{
            solver_impl::Conv2Solver<ST,Dst,TA,TB,true,true,CT>::eval( dst, src.lhs.__name_const().exp.__name_const(), src.rhs.__name_const().exp.__name_const() );
        }        
    };
    namespace operators{
        /*! \brief operator overload for 2D convolution */
        template<typename TA, typename TB,typename TAA, typename TBB,typename CT>
        inline const Conv2Exp<TAA,TBB,CT> conv2( const Exp<TA,TAA> &lhs, const Exp<TB,TBB> &rhs, const enums::ConvType<CT> &ct ){
            return Conv2Exp<TAA,TBB,CT>( lhs.__alias_const(), rhs.__alias_const() );
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to cloning
         * user must specialize the class to create specific solvers of types 
         */        
        template<typename Dst, typename Src>
        struct CloneSolver{
            /*! \brief implement dst = clone( src ) */
            static inline void eval( Dst &dst, const Src &src  );
        };
    };    

    /*! \brief clone a container */
    template<typename Elem>
    class CloneExp: public CompositeExp< CloneExp<Elem> >{
    public:
        /*! \brief expression to be cloned  */
        const Elem &exp;
        /*! \brief constructor */
        CloneExp( const Elem &e ):exp(e){}        
        /*! \brief basic specialization of clone  */
        template<typename Dst, typename Src>
        inline void __eval( const enums::SaveTo &s, Dst &dst, const CloneExp< ContainerExp<Src> > &src ) const{
            solver_impl::CloneSolver<Dst,Src>::eval( dst, src.exp.__name_const() );
        }
    };
    namespace operators{
        /*! \brief operator implementation clone */
        template<typename T>
        inline const CloneExp< ContainerExp<T> > clone( const ContainerExp<T> &exp ){
            return CloneExp< ContainerExp<T> >( exp.__alias_const() );
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to alloc_like
         * user must specialize the class to create specific solvers of types 
         */        
        template<typename Dst, typename Src>
        struct AllocLikeSolver{
            /*! \brief implement dst = alloc_like( src ) */
            static inline void eval( Dst &dst, const Src &src  );
        };
    };    
    /*! \brief allocate same shape of memory*/
    template<typename Elem>
    class AllocLikeExp: public CompositeExp< AllocLikeExp<Elem> >{
    public:
        /*! \brief expression to define the shape */
        const Elem &exp;
        /*! \brief constructor */
        AllocLikeExp( const Elem &e ):exp(e){}        
        /*! \brief basic specialization of alloc like  */
        template<typename Dst, typename Src>
        inline void __eval( const enums::SaveTo &s, Dst &dst, const AllocLikeExp< ContainerExp<Src> > &src ) const{
            solver_impl::AllocLikeSolver<Dst,Src>::eval( dst, src.exp.__name_const() );
        }
    };
    namespace operators{
        /*! \brief operator implementation alloc_like */
        template<typename T>
        inline const AllocLikeExp< ContainerExp<T> > alloc_like( const ContainerExp<T> &exp ){
            return AllocLikeExp< ContainerExp<T> >( exp.__alias_const() );
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface sum2D
         * \tparam ST StoreMethod in enums
         *
         * user must specialize the class to create specific solvers of types 
         */        
        template<typename ST, typename Dst, typename Src>
        struct Sum2DSolver{
            /*! \brief implement dst = sum_2D( src ) */
            static inline void eval( Dst &dst, const Src &src  );
        };
    };    

    /*! \brief sum over last 2 dimensions */
    template<typename Elem>
    class Sum2DExp: public CompositeExp< Sum2DExp<Elem> >{
    public:
        /*! \brief expression to be cloned  */
        const Elem &exp;
        /*! \brief constructor */
        Sum2DExp( const Elem &e ):exp(e){}        
        /*! \brief basic specialization of clone  */
        template<typename ST, typename Dst, typename Src>
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const Sum2DExp< ContainerExp<Src> > &src ) const{
            solver_impl::Sum2DSolver<ST,Dst,Src>::eval( dst, src.exp.__name_const() );
        }
    };
    namespace operators{
        /*! \brief operator implementation sum2D */
        template<typename T>
        inline const Sum2DExp< ContainerExp<T> > sum_2D( const ContainerExp<T> &exp ){
            return Sum2DExp< ContainerExp<T> >( exp.__alias_const() );
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface sumRow
         * \tparam ST StoreMethod in enums
         *
         * user must specialize the class to create specific solvers of types 
         */        
        template<typename ST, typename Dst, typename Src>
        struct SumRowSolver{
            /*! \brief implement dst = sum_2D( src ) */
            static inline void eval( Dst &dst, const Src &src  );
        };
    };    

    /*! \brief sum over last 2 dimensions */
    template<typename Elem>
    class SumRowExp: public CompositeExp< SumRowExp<Elem> >{
    public:
        /*! \brief expression to be cloned  */
        const Elem &exp;
        /*! \brief constructor */
        SumRowExp( const Elem &e ):exp(e){}        
        /*! \brief basic specialization of clone  */
        template<typename ST, typename Dst, typename Src>
        inline void __eval( const enums::StoreMethod<ST> &s, Dst &dst, const SumRowExp< ContainerExp<Src> > &src ) const{
            solver_impl::SumRowSolver<ST,Dst,Src>::eval( dst, src.exp.__name_const() );
        }
    };
    namespace operators{
        /*! \brief operator implementation sum2D */
        template<typename T>
        inline const SumRowExp< ContainerExp<T> > sum_row( const ContainerExp<T> &exp ){
            return SumRowExp< ContainerExp<T> >( exp.__alias_const() );
        }
    };
};

#endif

