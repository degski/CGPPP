
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
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <mpark/variant.hpp> // https://github.com/mpark/variant
#include <vector>

#include <absl/container/fixed_array.h>
#include <absl/container/inlined_vector.h>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

#include <frozen/unordered_map.h>
#include <frozen/string.h>

#include "random.hpp"

#ifndef nl
#define DEF_NL
#define nl '\n'
#endif


namespace cgp {

namespace function {

// Node function defines in CGP-Library.
template<typename Real = float> Real f_add ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_sub ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_mul ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_divide ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_idiv ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_irem ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_negate ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_absolute ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_squareRoot ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_square ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_cube ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_power ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_exponential ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_sine ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_cosine ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_tangent ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_randFloat ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_constTwo ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_constOne ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_constZero ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_constPI ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_and ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_nand ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_or ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_nor ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_xor ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_xnor ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_not ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_wire ( const std::vector<Real> & inputs_ ) noexcept;
template<typename Real = float> Real f_sigmoid ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_gaussian ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_step ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_softsign ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_hyperbolicTangent ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
}

template<typename Real = float>
using FunctionPointer = Real ( * ) ( const std::vector<Real> & inputs_ );
template<typename Real = float>
using FunctionPointerANN = Real ( * ) ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ );


template<typename Real = float>
struct FunctionSet {

    using Pointer = mpark::variant<FunctionPointer<Real>, FunctionPointerANN<Real>>;

    private:

    struct FunctionData {
        Pointer function;
        int maxNumInputs;
        constexpr FunctionData ( FunctionPointer<Real> fp_, int n_ ) noexcept :
            function ( mpark::in_place_type<FunctionPointer<Real>>, fp_ ),
            maxNumInputs ( n_ ) {
        }
        constexpr FunctionData ( FunctionPointerANN<Real> fp_, int n_ ) noexcept :
            function ( mpark::in_place_type<FunctionPointerANN<Real>>, fp_ ),
            maxNumInputs ( n_ ) {
        }
    };

    public:

    std::vector<frozen::string> functionNames;
    std::vector<Pointer> function;
    std::vector<int> maxNumInputs;

    int numFunctions = 0;

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        ( addPresetNodeFunction ( args_ ), ... );
    }

    void addPresetNodeFunction ( const frozen::string & functionName_ ) {
        auto [ f, n ] { function_set.at ( functionName_ ) };
        functionNames.push_back ( functionName_ );
        function.push_back ( f );
        maxNumInputs.push_back ( n );
        ++numFunctions;
    }

    template<typename PointerType>
    void addCustomNodeFunction ( const frozen::string & functionName_, PointerType function_, int maxNumInputs_ ) {
        functionNames.push_back ( functionName_ );
        function.emplace_back ( mpark::in_place_type<PointerType>, function_ );
        maxNumInputs.push_back ( maxNumInputs_ );
        ++numFunctions;
    }

    void clear ( ) noexcept {
        functionNames.clear ( );
        function.clear ( );
        maxNumInputs.clear ( );
        numFunctions = 0;
    }

    void print ( ) const noexcept {
        std::cout << "Function Set:";
        for ( const auto & name : functionNames )
            std::cout << ' ' << name.data ( );
        std::cout << " (" << numFunctions << ")\n";
    }

    [[ nodiscard ]] static constexpr std::size_t sizeBuiltinFunctionSet ( ) noexcept {
        return function_set.size ( );
    }

    private:

    static constexpr frozen::unordered_map<frozen::string, FunctionData, 34> function_set {
        { "add", { function::f_add, -1 } },
        { "sub", { function::f_sub, -1 } },
        { "mul", { function::f_mul, -1 } },
        { "div", { function::f_divide, -1 } },
        { "idiv", { function::f_idiv, 2 } },
        { "irem", { function::f_irem, 2 } },
        { "neg", { function::f_negate, 1 } },
        { "abs", { function::f_absolute, 1 } },
        { "sqrt", { function::f_squareRoot, 1 } },
        { "sq", { function::f_square, 1 } },
        { "cube", { function::f_cube, 1 } },
        { "pow", { function::f_power, 2 } },
        { "exp", { function::f_exponential, 1 } },
        { "sin", { function::f_sine, 1 } },
        { "cos", { function::f_cosine, 1 } },
        { "tan", { function::f_tangent, 1 } },
        { "rand", { function::f_randFloat, 0 } },
        { "2", { function::f_constTwo, 0 } },
        { "1", { function::f_constOne, 0 } },
        { "0", { function::f_constZero, 0 } },
        { "pi", { function::f_constPI, 0 } },
        { "and", { function::f_and, -1 } },
        { "nand", { function::f_nand, -1 } },
        { "or", { function::f_or, -1 } },
        { "nor", { function::f_nor, -1 } },
        { "xor", { function::f_xor, -1 } },
        { "xnor", { function::f_xnor, -1 } },
        { "not", { function::f_not, 1 } },
        { "wire", { function::f_wire, 1 } },
        { "sig", { function::f_sigmoid, -1 } },
        { "gauss", { function::f_gaussian, -1 } },
        { "step", { function::f_step, -1 } },
        { "soft", { function::f_softsign, -1 } },
        { "tanh", { function::f_hyperbolicTangent, -1 } }
    };
};


namespace function {

// Node function add. Returns the sum of all the inputs.
template<typename Real> Real f_add ( const std::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 0 }, std::plus<Real> ( ) );
}

// Node function sub. Returns the first input minus all remaining inputs_.
template<typename Real> Real f_sub ( const std::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::next ( std::begin ( inputs_ ) ), std::end ( inputs_ ), inputs_ [ 0 ], std::minus<Real> ( ) );
}

// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> Real f_mul ( const std::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 1 }, std::multiplies<Real> ( ) );
}

// Node function div. Returns the first input divided by the second input divided by the third input etc
template<typename Real> Real f_divide ( const std::vector<Real> & inputs_ ) noexcept {
    return std::accumulate ( std::next ( std::begin ( inputs_ ) ), std::end ( inputs_ ), inputs_ [ 0 ], std::divides<Real> ( ) );
}

// Node function idiv.Returns the first input (cast to int) divided by the second input (cast to int),
// This function allows for integer arithmatic.
template<typename Real> Real f_idiv ( const std::vector<Real> & inputs_ ) noexcept {
    if ( 0 != static_cast<int> ( inputs_ [ 1 ] ) )
        return static_cast<Real> ( static_cast<int> ( inputs_ [ 0 ] ) / static_cast<int> ( inputs_ [ 1 ] ) );
    return Real { 0 };
}

// Node function irem. Returns the remainder of the first input (cast to int) divided by the second input (cast to int),
// This function allows for integer arithmatic.
template<typename Real> Real f_irem ( const std::vector<Real> & inputs_ ) noexcept {
    if ( 0 != static_cast<int> ( inputs_ [ 1 ] ) )
        return static_cast<Real> ( static_cast<int> ( inputs_ [ 0 ] ) % static_cast<int> ( inputs_ [ 1 ] ) );
    return Real { 0 };
}

// Node function abs. Returns the negation of the first input,
// This is useful if one doen't want to use the mathematically
// crazy sub function, then negate can be applied to add.
template<typename Real> Real f_negate ( const std::vector<Real> & inputs_ ) noexcept {
    return -inputs_ [ 0 ];
}

// Node function abs. Returns the absolute of the first input
template<typename Real> Real f_absolute ( const std::vector<Real> & inputs_ ) noexcept {
    return std::abs ( inputs_ [ 0 ] );
}

// Node function sqrt.  Returns the square root of the first input
template<typename Real> Real f_squareRoot ( const std::vector<Real> & inputs_ ) noexcept {
    return std::sqrt ( inputs_ [ 0 ] );
}

// Node function squ.  Returns the square of the first input
template<typename Real> Real f_square ( const std::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function cub.  Returns the cube of the first input
template<typename Real> Real f_cube ( const std::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function power.  Returns the first output to the power of the second
template<typename Real> Real f_power ( const std::vector<Real> & inputs_ ) noexcept {
    return std::pow ( inputs_ [ 0 ], inputs_ [ 1 ] );
}

// Node function exp.  Returns the exponential of the first input
template<typename Real> Real f_exponential ( const std::vector<Real> & inputs_ ) noexcept {
    return std::exp ( inputs_ [ 0 ] );
}

// Node function sin.  Returns the sine of the first input
template<typename Real> Real f_sine ( const std::vector<Real> & inputs_ ) noexcept {
    return std::sin ( inputs_ [ 0 ] );
}

// Node function cos.  Returns the cosine of the first input
template<typename Real> Real f_cosine ( const std::vector<Real> & inputs_ ) noexcept {
    return std::cos ( inputs_ [ 0 ] );
}

// Node function tan.  Returns the tangent of the first input
template<typename Real> Real f_tangent ( const std::vector<Real> & inputs_ ) noexcept {
    return std::tan ( inputs_ [ 0 ] );
}

// Node function one.  Always returns 1
template<typename Real> Real f_constTwo ( const std::vector<Real> & inputs_ ) noexcept {
    return 2.0;
}

// Node function one.  Always returns 1
template<typename Real> Real f_constOne ( const std::vector<Real> & inputs_ ) noexcept {
    return 1.0;
}

// Node function one.  Always returns 0
template<typename Real> Real f_constZero ( const std::vector<Real> & inputs_ ) noexcept {
    return 0.0;
}

// Node function one.  Always returns PI
template<typename Real> Real f_constPI ( const std::vector<Real> & inputs_ ) noexcept {
    return 3.141592653589793116;
}

// Node function rand.  Returns a random number between minus one and positive one
template<typename Real> Real f_randFloat ( const std::vector<Real> & inputs_ ) noexcept {
    return std::uniform_real_distribution<Real> ( -1.0, 1.0 ) ( rng ( ) );
}

// Node function and. logical AND, returns '1' if all inputs_ are '1'
//    else, '0'
template<typename Real> Real f_and ( const std::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( not ( i ) )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function and. logical NAND, returns '0' if all inputs_ are '1'
//    else, '1'
template<typename Real> Real f_nand ( const std::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( not ( i ) )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function or. logical OR, returns '0' if all inputs_ are '0'
//    else, '1'
template<typename Real> Real f_or ( const std::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( i )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function nor. logical NOR, returns '1' if all inputs_ are '0'
//    else, '0'
template<typename Real> Real f_nor ( const std::vector<Real> & inputs_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( i )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function xor. logical XOR, returns '1' iff one of the inputs_ is '1'
//    else, '0'. AKA 'one hot'.
template<typename Real> Real f_xor ( const std::vector<Real> & inputs_ ) noexcept {
    int numOnes = 0;
    for ( const auto i : inputs_ ) {
        if ( i )
            ++numOnes;
        if ( numOnes > 1 )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function xnor. logical XNOR, returns '0' iff one of the inputs_ is '1'
//    else, '1'.
template<typename Real> Real f_xnor ( const std::vector<Real> & inputs_ ) noexcept {
    int numOnes = 0;
    for ( const auto i : inputs_ ) {
        if ( i )
            ++numOnes;
        if ( numOnes > 1 )
            return Real { 1 };
        }
        return Real { 0 };
}

// Node function not. logical NOT, returns '1' if first input is '0', else '1'
template<typename Real> Real f_not ( const std::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ] == Real { 0 };
}

// Node function wire. simply acts as a wire returning the first input
template<typename Real> Real f_wire ( const std::vector<Real> & inputs_ ) noexcept {
    return inputs_ [ 0 ];
}

// Node function sigmoid. returns the sigmoid of the sum of weighted inputs_.
//    The specific sigmoid function used in the logistic function.
//    range: [0,1]
template<typename Real> Real f_sigmoid ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    const Real sumWeigtedInputs = std::inner_product ( std::begin ( inputs_ ), std::end ( inputs_ ), std::begin ( connectionWeights_ ), Real { 0 } );
    return Real { 1 } / ( Real { 1 } + std::exp ( -sumWeigtedInputs ) );
}

// Node function Gaussian. returns the Gaussian of the sum of weighted inputs_.
//    range: [0,1]
template<typename Real> Real f_gaussian ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    constexpr int centre = 0, width = 1;
    const Real sumWeigtedInputs = std::inner_product ( std::begin ( inputs_ ), std::end ( inputs_ ), std::begin ( connectionWeights_ ), Real { 0 } );
    return std::exp ( -( std::pow ( sumWeigtedInputs - centre, Real { 2 } ) ) / ( Real { 2 } * std::pow ( width, 2 ) ) );
}

// Node function step. returns the step function of the sum of weighted inputs_.
//    range: [0,1]
template<typename Real> Real f_step ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    const Real sumWeigtedInputs = std::inner_product ( std::begin ( inputs_ ), std::end ( inputs_ ), std::begin ( connectionWeights_ ), Real { 0 } );
    return sumWeigtedInputs >= Real { 0 };
}

// Node function step. returns the step function of the sum of weighted inputs_.
//    range: [-1,1]
template<typename Real> Real f_softsign ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    const Real sumWeigtedInputs = std::inner_product ( std::begin ( inputs_ ), std::end ( inputs_ ), std::begin ( connectionWeights_ ), Real { 0 } );
    return sumWeigtedInputs / ( Real { 1 } + std::abs ( sumWeigtedInputs ) );
}

// Node function tanh. returns the tanh function of the sum of weighted inputs_.
//    range: [-1,1]
template<typename Real> Real f_hyperbolicTangent ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    const Real sumWeigtedInputs = std::inner_product ( std::begin ( inputs_ ), std::end ( inputs_ ), std::begin ( connectionWeights_ ), Real { 0 } );
    return std::tanh ( sumWeigtedInputs );
}

} // namespace function
} // namespace cgp


#if defined ( DEF_NL )
#undef nl
#undef DEF_NL
#endif
