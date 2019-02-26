
// MIT License
//
// Copyright (c) 2019 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <sax/iostream.hpp>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/access.hpp>

#include <frozen/unordered_map.h>
#include <frozen/string.h>

#include <plf/plf_nanotimer.h>

#include <sax/singleton.hpp>
#include <sax/stl.hpp>

#include "types.hpp"
#include "random.hpp"


namespace cgp {

namespace function {

// Node function add. Returns the sum of all the inputs.
template<typename Real> [[ nodiscard ]] Real f_add ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sub. Returns the first input minus minus the second input.
template<typename Real> [[ nodiscard ]] Real f_sub ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> [[ nodiscard ]] Real f_mul ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function div. Returns the first input divided by the second.
template<typename Real> [[ nodiscard ]] Real f_div ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function reci. Returns the reciproke of the first input.
template<typename Real> [[ nodiscard ]] Real f_reci ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function idiv.Returns the first input (cast to int) divided by the second
// input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_idiv ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function irem. Returns the remainder of the first input (cast to int) divided
// by the second input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_irem ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function abs. Returns the negation of the first input. This is useful if one
// doesn't want to use the mathematically crazy sub function, then negate can be
// applied to add.
template<typename Real> [[ nodiscard ]] Real f_neg ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function abs. Returns the absolute of the first input.
template<typename Real> [[ nodiscard ]] Real f_abs ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sqrt. Returns the square root of the first input.
template<typename Real> [[ nodiscard ]] Real f_sqrt ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function cbrt. Returns the cube root of the first input.
template<typename Real> [[ nodiscard ]] Real f_cbrt ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sqr. Returns the square of the first input.
template<typename Real> [[ nodiscard ]] Real f_sqr ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function cub. Returns the cube of the first input.
template<typename Real> [[ nodiscard ]] Real f_cube ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function power. Returns the first output to the power of the second.
template<typename Real> [[ nodiscard ]] Real f_pow ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function exp. Returns the exponential of the first input.
template<typename Real> [[ nodiscard ]] Real f_exp ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function exp2. Returns the 2 ^ x of the first input.
template<typename Real> [[ nodiscard ]] Real f_exp2 ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function log. Returns the natural logarith of the first input.
template<typename Real> [[ nodiscard ]] Real f_log ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function log2. Returns the  log base 2 of the first input.
template<typename Real> [[ nodiscard ]] Real f_log2 ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sin. Returns the sine of the first input.
template<typename Real> [[ nodiscard ]] Real f_sin ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function cos. Returns the cosine of the first input.
template<typename Real> [[ nodiscard ]] Real f_cos ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function tan. Returns the tangent of the first input.
template<typename Real> [[ nodiscard ]] Real f_tan ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function asin. Returns the arc sine of the first input.
template<typename Real> [[ nodiscard ]] Real f_asin ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function acos. Returns the arc cosine of the first input.
template<typename Real> [[ nodiscard ]] Real f_acos ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function atan. Returns the arc tangent of the first input.
template<typename Real> [[ nodiscard ]] Real f_atan ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function 0. Always returns 0, etc below.
template<typename Real> [[ nodiscard ]] Real f_0 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_1 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_2 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_3 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_4 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_5 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_6 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_7 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_8 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_9 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_10 ( const stl::vector<Real> & inputs_ ) noexcept;
template<typename Real> [[ nodiscard ]] Real f_16 ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function e. Always returns Euler's number.
template<typename Real> [[ nodiscard ]] Real f_e ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function pi. Always returns Pi.
template<typename Real> [[ nodiscard ]] Real f_pi ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function rand. Returns a random number [ -1, 1 ].
template<typename Real> [[ nodiscard ]] Real f_rand ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function bern. Returns a random -1 or 1.
template<typename Real> [[ nodiscard ]] Real f_bern ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function and. Return logical AND, returns 1 if all inputs_ are 1 else, 1.
template<typename Real> [[ nodiscard ]] Real f_and ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function nand. Returns logical NAND, returns 0 if all inputs_ are 1 else, 1.
template<typename Real> [[ nodiscard ]] Real f_nand ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function or. Returns logical OR, returns 0 if all inputs_ are 0 else, 1.
template<typename Real> [[ nodiscard ]] Real f_or ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function nor. Returns logical NOR, returns 1 if all inputs_ are 0 else, 0.
template<typename Real> [[ nodiscard ]] Real f_nor ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function xor. Returns logical XOR, returns 1 iff one of the inputs_ is 1
// else, 0. a.k.a. 'one hot'.
template<typename Real> [[ nodiscard ]] Real f_xor ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function xnor. Returns logical XNOR, returns 0 iff one of the inputs_ is 1
// else, 1.
template<typename Real> [[ nodiscard ]] Real f_xnor ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function not. Returns logical NOT, returns 1 if first input is 0, else 1.
template<typename Real> [[ nodiscard ]] Real f_not ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function wire. Simply acts as a wire returning the first input.
template<typename Real> [[ nodiscard ]] Real f_wire ( const stl::vector<Real> & inputs_ ) noexcept;

} // namespace function


template<typename Real>
using FunctionPointer = Real ( * ) ( const stl::vector<Real> & inputs_ );


template<typename Real>
struct FunctionSet {

    using Pointer = FunctionPointer<Real>;
    using Label = frozen::string;

    static constexpr int variableNumInputs = std::numeric_limits<int>::max ( );

    private:

    struct FunctionData {
        const Pointer function;
        const Real cost;
        const int arity = variableNumInputs;
    };

    public:

    std::vector<Label> label;
    std::vector<Pointer> function;
    std::vector<Real> cost;
    std::vector<int> arity;

    int size = 0;

    [[ nodiscard ]] int getRandomFunction ( ) const noexcept {
        return Rng::randInt ( size );
    }

    friend class cereal::access;

    template<typename Archive>
    void save ( Archive & archive_ ) const {
        archive_ ( size );
        for ( const auto & name : label )
            archive_ ( std::string { name.data ( ), name.size ( ) } );
    }

    template<typename Archive>
    void load ( Archive & archive_ ) {
        archive_ ( size );
        std::string name;
        for ( int i = 0; i < size; ++i ) {
            name.clear ( );
            archive_ ( name );
            auto [ f, c, a ] { m_function_set.at ( label.emplace_back ( name.data ( ), name.size ( ) ) ) };
            function.push_back ( f );
            cost.push_back ( c );
            arity.push_back ( a );
        }
    }

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        ( addPresetNodeFunction ( args_ ), ... );
    }

    void addPresetNodeFunction ( Label && label_ ) {
        auto [ f, c, a ] { m_function_set.at ( label.emplace_back ( std::move ( label_ ) ) ) };
        function.push_back ( f );
        cost.push_back ( c );
        arity.push_back ( a );
        ++size;
    }

    template<typename PointerType>
    void addCustomNodeFunction ( const Label & label_, const PointerType function_, const Real cost_, const int arity_ ) {
        label.push_back ( label_ );
        function.emplace_back ( function_ );
        cost.push_back ( cost_ );
        arity.push_back ( arity_ );
        ++size;
    }

    void clear ( ) noexcept {
        label.clear ( );
        function.clear ( );
        cost.clear ( );
        arity.clear ( );
        size = 0;
    }

    void printActiveFunctionSet ( ) const noexcept {
        std::cout << "Active Function Set:";
        for ( const auto & name : label )
            std::cout << ' ' << name.data ( );
        std::cout << " (" << size << ')' << nl;
    }

    static constexpr void printBuiltinFunctionSet ( ) noexcept {
        std::cout << "Built-in Function Set:";
        for ( const auto & name : m_function_set )
            std::cout << ' ' << name.first.data ( );
        std::cout << " (" << m_function_set.size ( ) << ')' << nl;
    }

    [[ nodiscard ]] static constexpr int sizeBuiltinFunctionSet ( ) noexcept {
        return static_cast<int> ( m_function_set.size ( ) );
    }

    [[ nodiscard ]] static constexpr const Label & builtinLabel ( const int i_ ) noexcept {
        return m_function_set.begin ( ) [ i_ ].first;
    }
    [[ nodiscard ]] static constexpr const FunctionData & builtinFunction ( const int i_ ) noexcept {
        return m_function_set.begin ( ) [ i_ ].second;
    }

    // private:

    static constexpr frozen::unordered_map<Label, FunctionData, 48> m_function_set {
        { "0", { function::f_0, 2.7, 0 } },
        { "1", { function::f_1, 2.8, 0 } },
        { "10", { function::f_10, 2.7, 0 } },
        { "16", { function::f_16, 2.7, 0 } },
        { "2", { function::f_2, 2.8, 0 } },
        { "3", { function::f_3, 2.7, 0 } },
        { "4", { function::f_4, 2.8, 0 } },
        { "5", { function::f_5, 2.7, 0 } },
        { "6", { function::f_6, 2.7, 0 } },
        { "7", { function::f_7, 2.7, 0 } },
        { "8", { function::f_8, 2.7, 0 } },
        { "9", { function::f_9, 2.7, 0 } },
        { "abs", { function::f_abs, 3.2, 1 } },
        { "acos", { function::f_acos, 17.2, 1 } },
        { "add", { function::f_add, 7.2 } },
        { "and", { function::f_and, 5.1 } },
        { "asin", { function::f_asin, 17.7, 1 } },
        { "atan", { function::f_atan, 11.5, 1 } },
        { "bern", { function::f_bern, 7.1, 0 } },
        { "cbrt", { function::f_cbrt, 65.8, 1 } },
        { "cos", { function::f_cos, 8.2, 1 } },
        { "cube", { function::f_cube, 2.7, 1 } },
        { "div", { function::f_div, 3.2, 2 } },
        { "e", { function::f_e, 2.7, 0 } },
        { "exp", { function::f_exp, 9.8, 1 } },
        { "exp2", { function::f_exp2, 76.3, 1 } },
        { "idiv", { function::f_idiv, 3.3, 2 } },
        { "irem", { function::f_irem, 3.3, 2 } },
        { "log", { function::f_log, 11.1, 1 } },
        { "log2", { function::f_log2, 44.7, 1 } },
        { "mul", { function::f_mul, 8.8 } },
        { "nand", { function::f_nand, 5.6 } },
        { "neg", { function::f_neg, 3.2, 1 } },
        { "nor", { function::f_nor, 3.8 } },
        { "not", { function::f_not, 3.4, 1 } },
        { "or", { function::f_or, 3.9 } },
        { "pi", { function::f_pi, 2.7, 0 } },
        { "pow", { function::f_pow, 24.4, 2 } },
        { "rand", { function::f_rand, 14.3, 0 } },
        { "reci", { function::f_reci, 3.2, 1 } },
        { "sin", { function::f_sin, 8.1, 1 } },
        { "sqr", { function::f_sqr, 2.7, 1 } },
        { "sqrt", { function::f_sqrt, 3.5, 1 } },
        { "sub", { function::f_sub, 2.7, 2 } },
        { "tan", { function::f_tan, 9.1, 1 } },
        { "wire", { function::f_wire, 2.7, 1 } },
        { "xnor", { function::f_xnor, 5.7 } },
        { "xor", { function::f_xor, 7.4 } }
    };
};


namespace detail {
sax::singleton<FunctionSet<Float>> singletonFunctionSet;
}


const auto functionSet = [ ] { return detail::singletonFunctionSet.instance ( ); } ( );


namespace function {

// Node function add. Returns the sum of all the inputs.
template<typename Real> [[ nodiscard ]] Real f_add ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 0 }, std::plus<Real> ( ) );
}

// Node function sub. Returns the first input minus minus the second input.
template<typename Real> [[ nodiscard ]] Real f_sub ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] - inputs_ [ 1 ];
}

// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> [[ nodiscard ]] Real f_mul ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 1 }, std::multiplies<Real> ( ) );
}

// Node function div. Returns the first input divided by the second input.
template<typename Real> [[ nodiscard ]] Real f_div ( const stl::vector<Real> & inputs_ ) noexcept {
    return Real { 0 } != inputs_ [ 1 ] ? inputs_ [ 0 ] / inputs_ [ 1 ] : Real { 0 };
}

// Node function reci. Returns the reciproke of the first input.
template<typename Real> [[ nodiscard ]] Real f_reci ( const stl::vector<Real> & inputs_ ) noexcept {
    return Real { 0 } != inputs_ [ 0 ] ? Real { 1 } / inputs_ [ 0 ] : Real { 0 };
}

// Node function idiv.Returns the first input (cast to int) divided by the second
// input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_idiv ( const stl::vector<Real> & inputs_ ) noexcept {
    return 0 != static_cast<int> ( inputs_ [ 1 ] ) ? static_cast< Real > ( static_cast<int> ( inputs_ [ 0 ] ) / static_cast<int> ( inputs_ [ 1 ] ) ) : Real { 0 };
}

// Node function irem. Returns the remainder of the first input (cast to int) divided
// by the second input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_irem ( const stl::vector<Real> & inputs_ ) noexcept {
    return 0 != static_cast<int> ( inputs_ [ 1 ] ) ? static_cast< Real > ( static_cast<int> ( inputs_ [ 0 ] ) % static_cast<int> ( inputs_ [ 1 ] ) ) : Real { 0 };
}

// Node function abs. Returns the negation of the first input. This is useful if one
// doesn't want to use the mathematically crazy sub function, then negate can be
// applied to add.
template<typename Real> [[ nodiscard ]] Real f_neg ( const stl::vector<Real> & inputs_ ) noexcept {
    return -inputs_ [ 0 ];
}

// Node function abs. Returns the absolute of the first input.
template<typename Real> [[ nodiscard ]] Real f_abs ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::abs ( inputs_ [ 0 ] );
}

// Node function sqrt. Returns the square root of the first input.
template<typename Real> [[ nodiscard ]] Real f_sqrt ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] < Real { 0 } ? -std::sqrt ( std::abs ( inputs_ [ 0 ] ) ) : std::sqrt ( inputs_ [ 0 ] );
}

// Node function cbrt. Returns the cube root of the first input.
template<typename Real> [[ nodiscard ]] Real f_cbrt ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::cbrt ( inputs_ [ 0 ] );
}

// Node function sqr. Returns the square of the first input.
template<typename Real> [[ nodiscard ]] Real f_sqr ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function cub. Returns the cube of the first input.
template<typename Real> [[ nodiscard ]] Real f_cube ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function pow. Returns the first output to the power of the second.
template<typename Real> [[ nodiscard ]] Real f_pow ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] < Real { 0 } ? -std::pow ( std::abs ( inputs_ [ 0 ] ), inputs_ [ 1 ] ) : std::pow ( inputs_ [ 0 ], inputs_ [ 1 ] );
}

// Node function exp. Returns the exponential of the first input.
template<typename Real> [[ nodiscard ]] Real f_exp ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::exp ( inputs_ [ 0 ] );
}

// Node function exp2. Returns the 2 ^ x of the first input.
template<typename Real> [[ nodiscard ]] Real f_exp2 ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::exp2 ( inputs_ [ 0 ] );
}

// Node function log. Returns the natural logarith of the first input.
template<typename Real> [[ nodiscard ]] Real f_log ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] < Real { 0 } ? -std::log ( std::abs ( inputs_ [ 0 ] ) ) : std::log ( inputs_ [ 0 ] );
}

// Node function log2. Returns the  log base 2 of the first input.
template<typename Real> [[ nodiscard ]] Real f_log2 ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] < Real { 0 } ? -std::log2 ( std::abs ( inputs_ [ 0 ] ) ) : std::log2 ( inputs_ [ 0 ] );
}

// Node function sin. Returns the sine of the first input.
template<typename Real> [[ nodiscard ]] Real f_sin ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::sin ( inputs_ [ 0 ] );
}

// Node function cos. Returns the cosine of the first input.
template<typename Real> [[ nodiscard ]] Real f_cos ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::cos ( inputs_ [ 0 ] );
}

// Node function tan. Returns the tangent of the first input.
template<typename Real> [[ nodiscard ]] Real f_tan ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::tan ( inputs_ [ 0 ] );
}

// Node function asin. Returns the arc sine of the first input.
template<typename Real> [[ nodiscard ]] Real f_asin ( const stl::vector<Real> & inputs_ ) noexcept {
    Real i = Real { 0 };
    return std::asin ( std::modf ( inputs_ [ 0 ], & i ) );
}

// Node function acos. Returns the arc cosine of the first input.
template<typename Real> [[ nodiscard ]] Real f_acos ( const stl::vector<Real> & inputs_ ) noexcept {
    Real i = Real { 0 };
    return std::acos ( std::modf ( inputs_ [ 0 ], & i ) );
}

// Node function atan. Returns the arc tangent of the first input.
template<typename Real> [[ nodiscard ]] Real f_atan ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::atan ( inputs_ [ 0 ] );
}

// Node function 0. Always returns 0, etc below.
template<typename Real> [[ nodiscard ]] Real f_0 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 0.0;
}

template<typename Real> [[ nodiscard ]] Real f_1 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 1.0;
}

template<typename Real> [[ nodiscard ]] Real f_2 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 2.0;
}

template<typename Real> [[ nodiscard ]] Real f_3 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 3.0;
}

template<typename Real> [[ nodiscard ]] Real f_4 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 4.0;
}

template<typename Real> [[ nodiscard ]] Real f_5 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 5.0;
}

template<typename Real> [[ nodiscard ]] Real f_6 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 6.0;
}

template<typename Real> [[ nodiscard ]] Real f_7 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 7.0;
}

template<typename Real> [[ nodiscard ]] Real f_8 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 8.0;
}
template<typename Real> [[ nodiscard ]] Real f_9 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 9.0;
}

template<typename Real> [[ nodiscard ]] Real f_10 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 10.0;
}

template<typename Real> [[ nodiscard ]] Real f_16 ( const stl::vector<Real> & inputs_ ) noexcept {
    return 16.0;
}

// Node function e. Always returns Euler's number.
template<typename Real> [[ nodiscard ]] Real f_e ( const stl::vector<Real> & inputs_ ) noexcept {
    return 2.718'281'828'459'045'091;
}

// Node function pi. Always returns Pi.
template<typename Real> [[ nodiscard ]] Real f_pi ( const stl::vector<Real> & inputs_ ) noexcept {
    return 3.141'592'653'589'793'116;
}

// Node function rand. Returns a random number [ -1, 1 ].
template<typename Real> [[ nodiscard ]] Real f_rand ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::uniform_real_distribution<Real> ( -1.0, 1.0 ) ( Rng::gen );
}

// Node function bern. Returns a random -1 or 1.
template<typename Real> [[ nodiscard ]] Real f_bern ( const stl::vector<Real> & inputs_ ) noexcept {
    return static_cast< Real > ( std::bernoulli_distribution ( ) ( Rng::gen ) * 2 - 1 );
}

// Node function and. Return logical AND, returns 1 if all inputs_ are 1 else, 1.
template<typename Real> [[ nodiscard ]] Real f_and ( const stl::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( not ( i ) )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function nand. Returns logical NAND, returns 0 if all inputs_ are 1 else, 1.
template<typename Real> [[ nodiscard ]] Real f_nand ( const stl::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( not ( i ) )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function or. Returns logical OR, returns 0 if all inputs_ are 0 else, 1.
template<typename Real> [[ nodiscard ]] Real f_or ( const stl::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( i )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function nor. Returns logical NOR, returns 1 if all inputs_ are 0 else, 0.
template<typename Real> [[ nodiscard ]] Real f_nor ( const stl::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( i )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function xor. Returns logical XOR, returns 1 iff one of the inputs_ is 1
// else, 0. a.k.a. 'one hot'.
template<typename Real> [[ nodiscard ]] Real f_xor ( const stl::vector<Real> & inputs_ ) noexcept {
    int numOnes = 0;
    for ( const auto i : inputs_ ) {
        if ( i )
            ++numOnes;
        if ( numOnes > 1 )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function xnor. Returns logical XNOR, returns 0 iff one of the inputs_ is 1
// else, 1.
template<typename Real> [[ nodiscard ]] Real f_xnor ( const stl::vector<Real> & inputs_ ) noexcept {
    int numOnes = 0;
    for ( const auto i : inputs_ ) {
        if ( i )
            ++numOnes;
        if ( numOnes > 1 )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function not. Returns logical NOT, returns 1 if first input is 0, else 1.
template<typename Real> [[ nodiscard ]] Real f_not ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] == Real { 0 };
}

// Node function wire. Simply acts as a wire returning the first input.
template<typename Real> [[ nodiscard ]] Real f_wire ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ];
}

} // namespace function


template<typename Real>
struct FunctionStats {

    std::string name;
    double time = 0.0;
    int n = 0;

    void update ( const double elapsed_time_ ) noexcept {
        time += ( elapsed_time_ - time ) / ++n;
    }

    template<typename Stream, typename Real>
    friend Stream & operator << ( Stream & out_, const FunctionStats<Real> & v_ ) noexcept {
        const int arity = cgp::FunctionSet<Real>::m_function_set.at ( frozen::string { v_.name.data ( ), v_.name.length ( ) } ).arity;
        if ( cgp::FunctionSet<Real>::variableNumInputs == arity )
            out_ << "       { \"" << v_.name << "\", { function::f_" << v_.name << ", " << std::fixed << std::setprecision ( 1 ) << ( v_.time / 1000.0 ) << " } }," << nl;
        else
            out_ << "       { \"" << v_.name << "\", { function::f_" << v_.name << ", " << std::fixed << std::setprecision ( 1 ) << ( v_.time / 1000.0 ) << ", " << arity << " } }," << nl;
        return out_;
    }
};


template<typename Real>
stl::vector<Real> getInputs ( const int arity_ ) noexcept {
    const int arity = arity_ == cgp::FunctionSet<Real>::variableNumInputs ? std::geometric_distribution<> ( ) ( cgp::Rng::gen ) + 2 : arity_;
    stl::vector<Real> v ( arity );
    std::generate ( std::begin ( v ), std::end ( v ), [ ] { return std::uniform_real_distribution<Real> ( Real { -1 }, Real { 1 } ) ( cgp::Rng::gen ); } );
    return v;
}

template<typename Real>
float timeRandomFunction ( stl::vector<FunctionStats<Real>> & stats_ ) noexcept {
    const int i = cgp::Rng::randInt ( cgp::FunctionSet<Real>::sizeBuiltinFunctionSet ( ) );
    const auto f = cgp::FunctionSet<Real>::builtinFunction ( i );
    const stl::vector<Real> input = getInputs<Real> ( f.arity );
    Real r = Real { 0 };
    static plf::nanotimer timer;
    timer.start ( );
    for ( int i = 0; i < 1'000; ++i )
        r += f.function ( input );
    stats_ [ i ].update ( timer.get_elapsed_ns ( ) );
    return r / 1000.0f;
}

template<typename Real>
void generateCostTable ( ) {
    stl::vector<FunctionStats<Real>> stats ( cgp::FunctionSet<Real>::sizeBuiltinFunctionSet ( ) );
    for ( int i = 0; i < cgp::FunctionSet<Real>::sizeBuiltinFunctionSet ( ); ++i )
        stats [ i ].name = cgp::FunctionSet<Real>::builtinLabel ( i ).data ( );
    Real r = Real { 0 };
    for ( int i = 0; i < 10'000'000; ++i )
        r += timeRandomFunction<Real> ( stats );
    std::cout << ' ' << stats << nl;
}


} // namespace cgp
