
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
#include <cstdlib>

#include <random>

#include <sax/prng.hpp>
#include <sax/uniform_int_distribution.hpp>

#if defined ( _DEBUG )
#define RANDOM 0
#else
#define RANDOM 1
#endif


namespace cgp {

struct Rng {

    static void seed ( const std::uint64_t s_ = 0u ) noexcept {
        Rng::gen ( ).seed ( s_ ? s_ : sax::os_seed ( ) );
    }

    [[ nodiscard ]] static int randInt ( const int n_ ) noexcept {
        if ( not ( n_ ) )
            return 0;
        return sax::uniform_int_distribution<int> ( 0, n_ - 1 ) ( Rng::gen ( ) );
    }

    [[ nodiscard ]] static sax::Rng & gen ( ) noexcept {
        static thread_local sax::Rng generator ( RANDOM ? sax::os_seed ( ) : sax::fixed_seed ( ) );
        return generator;
    }
};

}

#undef RANDOM
