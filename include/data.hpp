
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
#include <charconv>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sax/iostream.hpp>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;

// https://github.com/degski/Sax/

#include <sax/singleton.hpp>
#include <sax/stl.hpp>
#include <sax/string_split.hpp>

#include "types.hpp"


namespace cgp {

template<typename Real>
struct Data {

    struct Sample {
        std::span<const Real> input;
        std::span<const Real> output;
    };

    Data ( ) noexcept { };
    Data ( fs::path && path_, std::string && file_name_ ) {
        loadFromFile ( std::move ( path_ ), std::move ( file_name_ ) );
    }

    private:

    int in_arity, out_arity, record_size, num_records;
    std::vector<Real> data;

    struct const_iterator {

        using difference_type = std::ptrdiff_t;
        using value_type = const Real;
        using pointer = const Real * ;
        using reference = const Real & ;
        using iterator_category = std::bidirectional_iterator_tag;

        const Real * rec = nullptr;
        short in, out, size, _pad;

        const_iterator ( ) noexcept { } // ?
        const_iterator ( const const_iterator & ) noexcept = default;
        template<typename It>
        const_iterator ( It it_, const int in_, const int out_ ) noexcept :
            rec ( &*it_ ),
            in ( static_cast<short> ( in_ ) ),
            out ( static_cast<short> ( out_ ) ),
            size ( in + out ) {
            assert ( ( in_ + out_ ) < static_cast<int> ( std::numeric_limits<short>::max ( ) ) );
        }

        [[ nodiscard ]] Sample operator * ( ) noexcept {
            assert ( rec );
            return { { rec, in }, { rec + in, out } };
        }

        [[ nodiscard ]] const_iterator & operator ++ ( ) noexcept {
            assert ( rec );
            rec += size;
            return * this;
        }
        [[ nodiscard ]] const_iterator & operator -- ( ) noexcept {
            assert ( rec );
            rec -= size;
            return * this;
        }

        [[ nodiscard ]] const_iterator operator ++ ( int ) noexcept {
            assert ( rec );
            const_iterator tmp = * this;
            rec += size;
            return tmp;
        }
        [[ nodiscard ]] const_iterator operator -- ( int ) noexcept {
            assert ( rec );
            const_iterator tmp = * this;
            rec -= size;
            return tmp;
        }

        [[ nodiscard ]] bool operator == ( const const_iterator & rhs_ ) const noexcept {
            assert ( rec );
            return rec == rhs_.rec;
        }
        [[ nodiscard ]] bool operator != ( const const_iterator & rhs_ ) const noexcept {
            assert ( rec );
            return rec != rhs_.rec;
        }

        [[ nodiscard ]] const_iterator & operator = ( const const_iterator & ) noexcept = default;
    };

    public:

    [[ nodiscard ]] const_iterator begin ( ) const noexcept { return const_iterator ( data.begin ( ), in_arity, out_arity ); }
    [[ nodiscard ]] const_iterator cbegin ( ) const noexcept { return const_iterator ( data.cbegin ( ), in_arity, out_arity ); }

    [[ nodiscard ]] const_iterator end ( ) const noexcept { return const_iterator ( data.end ( ), in_arity, out_arity ); }
    [[ nodiscard ]] const_iterator cend ( ) const noexcept { return const_iterator ( data.cend ( ), in_arity, out_arity ); }

    [[ nodiscard ]] const_iterator rbegin ( ) const noexcept { return const_iterator ( data.rbegin ( ), in_arity, out_arity ); }
    [[ nodiscard ]] const_iterator crbegin ( ) const noexcept { return const_iterator ( data.crbegin ( ), in_arity, out_arity ); }

    [[ nodiscard ]] const_iterator rend ( ) const noexcept { return const_iterator ( data.rend ( ), in_arity, out_arity ); }
    [[ nodiscard ]] const_iterator crend ( ) const noexcept { return const_iterator ( data.crend ( ), in_arity, out_arity ); }

    private:

    [[ nodiscard ]] int stringToInt ( const std::string & s_ ) const noexcept {
        int i;
        std::from_chars ( s_.data ( ), s_.data ( ) + s_.length ( ), i, 10 );
        return i;
    }

    [[ nodiscard ]] Real stringToReal ( const std::string & s_ ) const noexcept {
        Real r;
        std::from_chars ( s_.data ( ), s_.data ( ) + s_.length ( ), r, std::chars_format::fixed );
        return r;
    }

    public:

    void loadFromFile ( fs::path && path_, std::string && file_name_ ) {
        std::ifstream istream ( std::move ( path_ ) / std::move ( file_name_ ), std::ios::in );
        if ( not ( istream.is_open ( ) ) ) {
            std::cout << "Error: " << file_name_ << " cannot be found" << nl << "Terminating CGPPP-Library" << nl;
            std::abort ( );
        }
        {
            std::string line;
            std::getline ( istream, line );
            const auto params = sax::string_split ( line, ",", " ", "\t" );
            if ( 3 != std::size ( params ) ) {
                std::cout << "Error: data parameters: \"" << line << "\" are invalid" << nl << "Terminating CGPPP-Library" << nl;
                std::abort ( );
            }
            in_arity = stringToInt ( params [ 0 ] ), out_arity = stringToInt ( params [ 1 ] ), record_size = out_arity + in_arity, num_records = stringToInt ( params [ 2 ] );
            data.clear ( );
            data.reserve ( record_size * num_records );
            while ( std::getline ( istream, line ) ) {
                const auto record = sax::string_split ( line, ",", " ", "\t" );
                if ( record_size != record.size ( ) ) {
                    std::cout << "Error: the size " << record.size ( ) << " of the record on line " << data.size ( ) << " differs from the parameters size " << record_size << nl << "Terminating CGPPP-Library" << nl;
                    std::abort ( );
                }
                for ( int i = 0; i < record_size; ++i ) {
                    data.emplace_back ( stringToReal ( record [ i ] ) );
                }
            }
            if ( num_records * record_size != data.size ( ) ) {
                std::cout << "Error: the actual number of records " << data.size ( ) << " differs from the parameters number of records " << num_records << nl << "Terminating CGPPP-Library" << nl;
                std::abort ( );
            }
        }
        istream.close ( );
    }
};

namespace detail {
sax::singleton<Data<Float>> singletonDataSet;
}

using DataSet = Data<Float>;

auto dataSet = [ ] { return detail::singletonDataSet.instance ( ); } ( );

} // namespace cgp
