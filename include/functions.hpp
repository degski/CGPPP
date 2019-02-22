
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
#include <sax/iostream.hpp>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <absl/container/fixed_array.h>
#include <absl/container/inlined_vector.h>

#include <cereal/cereal.hpp>
#include <cereal/access.hpp>

#include <frozen/unordered_map.h>
#include <frozen/string.h>

// https://github.com/degski/Sax/


#include <sax/singleton.hpp>
#include <sax/stl.hpp>

#include "types.hpp"
#include "random.hpp"


namespace cgp {

namespace function {

// Node function add. Returns the sum of all the inputs.
template<typename Real> [[ nodiscard ]] Real f_add ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sub. Returns the first input minus all remaining inputs_.
template<typename Real> [[ nodiscard ]] Real f_sub ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> [[ nodiscard ]] Real f_mul ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function div. Returns the first input divided by the second.
template<typename Real> [[ nodiscard ]] Real f_divide ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function reci. Returns the reciproke of the first input.
template<typename Real> [[ nodiscard ]] Real f_reciprocal ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function idiv.Returns the first input (cast to int) divided by the second
// input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_idiv ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function irem. Returns the remainder of the first input (cast to int) divided
// by the second input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_irem ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function abs. Returns the negation of the first input. This is useful if one
// doesn't want to use the mathematically crazy sub function, then negate can be
// applied to add.
template<typename Real> [[ nodiscard ]] Real f_negate ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function abs. Returns the absolute of the first input.
template<typename Real> [[ nodiscard ]] Real f_absolute ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sqrt. Returns the square root of the first input.
template<typename Real> [[ nodiscard ]] Real f_squareRoot ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function sqr. Returns the square of the first input.
template<typename Real> [[ nodiscard ]] Real f_square ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function cub. Returns the cube of the first input.
template<typename Real> [[ nodiscard ]] Real f_cube ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function power. Returns the first output to the power of the second.
template<typename Real> [[ nodiscard ]] Real f_power ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function exp. Returns the exponential of the first input.
template<typename Real> [[ nodiscard ]] Real f_exponential ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function exp2. Returns the 2 ^ x of the first input.
template<typename Real> [[ nodiscard ]] Real f_exponential2 ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function log. Returns the natural logarith of the first input.
template<typename Real> [[ nodiscard ]] Real f_logarithm ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function log2. Returns the  log base 2 of the first input.
template<typename Real> [[ nodiscard ]] Real f_logarithm2 ( const stl::vector<Real> & inputs_ ) noexcept;
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
template<typename Real> [[ nodiscard ]] Real f_Euler ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function pi. Always returns Pi.
template<typename Real> [[ nodiscard ]] Real f_Pi ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function rand. Returns a random number [ -1, 1 ].
template<typename Real> [[ nodiscard ]] Real f_randFloat ( const stl::vector<Real> & inputs_ ) noexcept;
// Node function bern. Returns a random -1 or 1.
template<typename Real> [[ nodiscard ]] Real f_randBernoulli ( const stl::vector<Real> & inputs_ ) noexcept;
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

    private:

    struct FunctionData {
        const Pointer function;
        const int maxNumInputs = std::numeric_limits<int>::max ( );
    };

    public:

    stl::vector<frozen::string> functionNames;
    stl::vector<Pointer> function;
    stl::vector<int> maxNumInputs;

    int numFunctions = 0;

    friend class cereal::access;

    template<typename Archive>
    void save ( Archive & archive_ ) const {
        archive_ ( numFunctions );
        for ( const auto & name : functionNames )
            archive_ ( std::string { name.data ( ), name.size ( ) } );
    }

    template<typename Archive>
    void load ( Archive & archive_ ) {
        archive_ ( numFunctions );
        std::string name;
        for ( int i = 0; i < numFunctions; ++i ) {
            name.clear ( );
            archive_ ( name );
            auto [ f, n ] { m_function_set.at ( functionNames.emplace_back ( name.data ( ), name.size ( ) ) ) };
            function.push_back ( f );
            maxNumInputs.push_back ( n );
        }
    }

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        ( addPresetNodeFunction ( args_ ), ... );
    }

    void addPresetNodeFunction ( frozen::string && functionName_ ) {
        auto [ f, n ] { m_function_set.at ( functionNames.emplace_back ( std::move ( functionName_ ) ) ) };
        function.push_back ( f );
        maxNumInputs.push_back ( n );
        ++numFunctions;
    }

    template<typename PointerType>
    void addCustomNodeFunction ( const frozen::string & functionName_, PointerType function_, int maxNumInputs_ ) {
        functionNames.push_back ( functionName_ );
        function.emplace_back ( function_ );
        maxNumInputs.push_back ( maxNumInputs_ );
        ++numFunctions;
    }

    void clear ( ) noexcept {
        functionNames.clear ( );
        function.clear ( );
        maxNumInputs.clear ( );
        numFunctions = 0;
    }

    void printActiveFunctionSet ( ) const noexcept {
        std::cout << "Active Function Set:";
        for ( const auto & name : functionNames )
            std::cout << ' ' << name.data ( );
        std::cout << " (" << numFunctions << ')' << nl;
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

    [[ nodiscard ]] static constexpr frozen::string builtinFunctionName ( const int i_ ) noexcept {
        return m_function_set.begin ( ) [ i_ ].first;
    }
    [[ nodiscard ]] static constexpr const FunctionData & builtinFunction ( const int i_ ) noexcept {
        return m_function_set.begin ( ) [ i_ ].second;
    }

    // private:

    static constexpr frozen::unordered_map<frozen::string, FunctionData, 47> m_function_set {
        { "0", { function::f_0, 0 } },
        { "1", { function::f_1, 0 } },
        { "10", { function::f_10, 0 } },
        { "16", { function::f_16, 0 } },
        { "2", { function::f_2, 0 } },
        { "3", { function::f_3, 0 } },
        { "4", { function::f_4, 0 } },
        { "5", { function::f_5, 0 } },
        { "6", { function::f_6, 0 } },
        { "7", { function::f_7, 0 } },
        { "8", { function::f_8, 0 } },
        { "9", { function::f_9, 0 } },
        { "abs", { function::f_absolute, 1 } },
        { "acos", { function::f_acos, 1 } },
        { "add", { function::f_add } },
        { "and", { function::f_and } },
        { "asin", { function::f_asin, 1 } },
        { "atan", { function::f_atan, 1 } },
        { "bern", { function::f_randBernoulli, 0 } },
        { "cos", { function::f_cos, 1 } },
        { "cube", { function::f_cube, 1 } },
        { "div", { function::f_divide, 2 } },
        { "e", { function::f_Euler, 0 } },
        { "exp", { function::f_exponential, 1 } },
        { "exp2", { function::f_exponential2, 1 } },
        { "idiv", { function::f_idiv, 2 } },
        { "irem", { function::f_irem, 2 } },
        { "log", { function::f_logarithm, 1 } },
        { "log2", { function::f_logarithm2, 1 } },
        { "mul", { function::f_mul } },
        { "nand", { function::f_nand } },
        { "neg", { function::f_negate, 1 } },
        { "nor", { function::f_nor } },
        { "not", { function::f_not, 1 } },
        { "or", { function::f_or } },
        { "pi", { function::f_Pi, 0 } },
        { "pow", { function::f_power, 2 } },
        { "rand", { function::f_randFloat, 0 } },
        { "reci", { function::f_reciprocal, 1 } },
        { "sin", { function::f_sin, 1 } },
        { "sqr", { function::f_square, 1 } },
        { "sqrt", { function::f_squareRoot, 1 } },
        { "sub", { function::f_sub, 2 } },
        { "tan", { function::f_tan, 1 } },
        { "wire", { function::f_wire, 1 } },
        { "xnor", { function::f_xnor } },
        { "xor", { function::f_xor } }
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

// Node function sub. Returns the first input minus all remaining inputs_.
template<typename Real> [[ nodiscard ]] Real f_sub ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] - inputs_ [ 1 ];
    // return std::accumulate ( std::next ( std::begin ( inputs_ ) ), std::end ( inputs_ ), inputs_ [ 0 ], std::minus<Real> ( ) );
}

// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> [[ nodiscard ]] Real f_mul ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 1 }, std::multiplies<Real> ( ) );
}

// Node function div. Returns the first input divided by the second input divided by
// the third input etc.
template<typename Real> [[ nodiscard ]] Real f_divide ( const stl::vector<Real> & inputs_ ) noexcept {
    if ( inputs_ [ 1 ] )
        return inputs_ [ 0 ] / inputs_ [ 1 ];
    return Real { 0 };
    // return std::accumulate ( std::next ( std::begin ( inputs_ ) ), std::end ( inputs_ ), inputs_ [ 0 ], std::divides<Real> ( ) );
}

// Node function reci. Returns the reciproke of the first input.
template<typename Real> [[ nodiscard ]] Real f_reciprocal ( const stl::vector<Real> & inputs_ ) noexcept {
    if ( inputs_ [ 0 ] )
        return Real { 1 } / inputs_ [ 0 ];
    return Real { 0 };
}

// Node function idiv.Returns the first input (cast to int) divided by the second
// input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_idiv ( const stl::vector<Real> & inputs_ ) noexcept {
    if ( 0 != static_cast<int> ( inputs_ [ 1 ] ) )
        return static_cast< Real > ( static_cast<int> ( inputs_ [ 0 ] ) / static_cast<int> ( inputs_ [ 1 ] ) );
    return Real { 0 };
}

// Node function irem. Returns the remainder of the first input (cast to int) divided
// by the second input (cast to int). This function allows for integer arithmatic.
template<typename Real> [[ nodiscard ]] Real f_irem ( const stl::vector<Real> & inputs_ ) noexcept {
    if ( 0 != static_cast<int> ( inputs_ [ 1 ] ) )
        return static_cast< Real > ( static_cast<int> ( inputs_ [ 0 ] ) % static_cast<int> ( inputs_ [ 1 ] ) );
    return Real { 0 };
}

// Node function abs. Returns the negation of the first input. This is useful if one
// doesn't want to use the mathematically crazy sub function, then negate can be
// applied to add.
template<typename Real> [[ nodiscard ]] Real f_negate ( const stl::vector<Real> & inputs_ ) noexcept {
    return -inputs_ [ 0 ];
}

// Node function abs. Returns the absolute of the first input.
template<typename Real> [[ nodiscard ]] Real f_absolute ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::abs ( inputs_ [ 0 ] );
}

// Node function sqrt. Returns the square root of the first input.
template<typename Real> [[ nodiscard ]] Real f_squareRoot ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::sqrt ( inputs_ [ 0 ] );
}

// Node function sqr. Returns the square of the first input.
template<typename Real> [[ nodiscard ]] Real f_square ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function cub. Returns the cube of the first input.
template<typename Real> [[ nodiscard ]] Real f_cube ( const stl::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function power. Returns the first output to the power of the second.
template<typename Real> [[ nodiscard ]] Real f_power ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::pow ( inputs_ [ 0 ], inputs_ [ 1 ] );
}

// Node function exp. Returns the exponential of the first input.
template<typename Real> [[ nodiscard ]] Real f_exponential ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::exp ( inputs_ [ 0 ] );
}

// Node function exp2. Returns the 2 ^ x of the first input.
template<typename Real> [[ nodiscard ]] Real f_exponential2 ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::exp2 ( inputs_ [ 0 ] );
}

// Node function log. Returns the natural logarith of the first input.
template<typename Real> [[ nodiscard ]] Real f_logarithm ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::log ( inputs_ [ 0 ] );
}

// Node function log2. Returns the  log base 2 of the first input.
template<typename Real> [[ nodiscard ]] Real f_logarithm2 ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::log2 ( inputs_ [ 0 ] );
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
    return std::asin ( inputs_ [ 0 ] );
}

// Node function acos. Returns the arc cosine of the first input.
template<typename Real> [[ nodiscard ]] Real f_acos ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::acos ( inputs_ [ 0 ] );
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
template<typename Real> [[ nodiscard ]] Real f_Euler ( const stl::vector<Real> & inputs_ ) noexcept {
    return 2.718'281'828'459'045'091;
}

// Node function pi. Always returns Pi.
template<typename Real> [[ nodiscard ]] Real f_Pi ( const stl::vector<Real> & inputs_ ) noexcept {
    return 3.141'592'653'589'793'116;
}

// Node function rand. Returns a random number [ -1, 1 ].
template<typename Real> [[ nodiscard ]] Real f_randFloat ( const stl::vector<Real> & inputs_ ) noexcept {
    return std::uniform_real_distribution<Real> ( -1.0, 1.0 ) ( Rng::gen );
}

// Node function bern. Returns a random -1 or 1.
template<typename Real> [[ nodiscard ]] Real f_randBernoulli ( const stl::vector<Real> & inputs_ ) noexcept {
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


} // namespace cgp
