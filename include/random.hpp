
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

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <random>


#if UINTPTR_MAX == 0xFFFF'FFFF
#define M32 1
#define M64 0
#elif UINTPTR_MAX == 0xFFFF'FFFF'FFFF'FFFF
#define M32 0
#define M64 1
#else
#error funny pointers detected
#endif

#if M64
#if defined ( __clang__ ) or defined ( __GNUC__ )
#include <lehmer.hpp>       // https://github.com/degski/Sax/blob/master/lehmer.hpp
#else
#include <splitmix.hpp>     // https://github.com/degski/Sax/blob/master/splitmix.hpp
#endif
#endif

#include <singleton.hpp>    // https://github.com/degski/Sax/blob/master/singleton.hpp


namespace cgp {

#if M64
#if defined ( __clang__ ) or defined ( __GNUC__ )
using Rng = mcg128_fast;
#else
using Rng = splitmix64;
#endif
[[ nodiscard ]] std::uint64_t getSystemSeed ( ) noexcept {
    std::cout << "init\n";
    return static_cast<std::uint64_t> ( std::random_device { } ( ) ) << 32 | static_cast<std::uint64_t> ( std::random_device { } ( ) );
}
#else
using Rng = std::minstd_rand;
[[ nodiscard ]] std::uint32_t getSystemSeed ( ) noexcept {
    return std::random_device { } ( );
}
#endif

singleton<Rng> rng;

auto seedFromSystem = [ ] { const auto s = getSystemSeed ( ); rng.instance ( ).seed ( s ); return s; } ( );

} // namespace cgp

#undef M64
#undef M32
