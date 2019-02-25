
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
#include <cstring>

#include <limits>
#include <memory>
#include <type_traits>


using Float = float;


#if defined ( USE_PECTOR )
#include <pector/pector.h> // Use my fork at https://github.com/degski/pector, or hell will come upon you.
#include <pector/malloc_allocator.h>
#include <cereal/types/pector.hpp>
#else
#include <vector>
#include <cereal/types/vector.hpp>
#endif

namespace stl {

#if defined ( USE_PECTOR )
template<typename T>
using vector = pt::pector<T, pt::malloc_allocator<T, true, false>, int, pt::default_recommended_size, false>;
#else
template<typename T>
using vector = std::vector<T>;
#endif

namespace detail {

template<typename T>
class null_allocator {

    public:

    using value_type = T;
    using pointer = value_type * ;
    using const_pointer = typename std::pointer_traits<pointer>::template rebind<const value_type>;

    template<typename U>
    struct rebind {
        using other = null_allocator<U>;
    };

    null_allocator ( ) noexcept { }
    template<typename U>
    null_allocator ( null_allocator<U> const & ) noexcept { }

    constexpr pointer allocate ( std::size_t ) const noexcept { return nullptr; }
    constexpr void deallocate ( pointer, std::size_t ) const noexcept { }

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::is_empty<null_allocator>;
};

}

template<typename T, typename U>
bool operator == ( const detail::null_allocator<T> &, const detail::null_allocator<U> & ) noexcept {
    return true;
}

template<typename T, typename U>
bool operator != ( const detail::null_allocator<T> &, const detail::null_allocator<U> & ) noexcept {
    return false;
}


#define CAPACITY( D ) D [ -2 ]
#define SIZE( D ) D [ -1 ]

#if defined ( _DEBUG )
#define ALIGNED_MALLOC(A,S) _aligned_malloc_dbg((S),(A),NULL,NULL)
#define ALIGNED_REALLOC(D,A,S) _aligned_realloc_dbg((D),(S),(A),NULL,NULL)
#define ALIGNED_FREE(D) _aligned_free_dbg((D))
#else
#define ALIGNED_MALLOC(A,S) _aligned_malloc((S),(A))
#define ALIGNED_REALLOC(D,A,S) _aligned_realloc((D),(S),(A))
#define ALIGNED_FREE(D) _aligned_free((D))
#endif


// A simplified dynamic vector of minimal foot-print for storing signed integers.
template<typename Type, typename = typename std::enable_if<std::conjunction<std::is_integral<Type>, std::is_signed<Type>>::value>::type>
struct sintor {

    using value_type = Type;
    using pointer = value_type * ;
    using const_pointer = const value_type*;

    using reference = value_type & ;
    using const_reference = const value_type &;
    using rv_reference = value_type && ;

    using size_type = Type;
    using difference_type = size_type;

    using allocator_type = detail::null_allocator<Type>;

    using iterator = pointer;
    using const_iterator = const pointer;
    using reverse_iterator = pointer;
    using const_reverse_iterator = const pointer;

    pointer m_data;

    explicit sintor ( ) :
        m_data ( nullptr ) {
    }
    // Creates a sintor of capacity n and size n;
    explicit sintor ( const size_type n_ ) :
        m_data ( alloc ( n_ ) ) {
        SIZE ( m_data ) = n_;
    }
    sintor ( sintor && other_ ) noexcept :
        m_data ( other_.m_data ) {
        other_.m_data = nullptr;
    }
    sintor ( const sintor & other_ ) :
        m_data ( alloc ( other_.size ( ) ) ) {
        SIZE ( other_.size ( ) );
        std::memcpy ( m_data, other_.m_data, sizeof ( value_type ) * size ( ) );
    }

    ~sintor ( ) {
        free ( );
    }

    [[ maybe_unused ]] sintor & operator = ( sintor && other_ ) noexcept {
        m_data = other_.m_data;
        other_.m_data = nullptr;
        return * this;
    }
    [[ maybe_unused ]] sintor & operator = ( const sintor & other_ ) {
        if ( capacity ( ) < other_.size ( ) ) {
            free ( );
            m_data = alloc ( other_.size ( ) );
        }
        SIZE ( other_.size ( ) );
        std::memcpy ( m_data, other_.m_data, sizeof ( value_type ) * size ( ) );
        return *this;
    }

    [[ nodiscard ]] reference operator [ ] ( const size_type i_ ) noexcept {
        return m_data [ i_ ];
    }
    [[ nodiscard ]] const_reference operator [ ] ( const size_type i_ ) const noexcept {
        return m_data [ i_ ];
    }

    [[ maybe_unused ]] reference push_back ( const value_type & v_ ) {
        if ( m_data ) {
            if ( SIZE ( m_data ) == CAPACITY ( m_data ) )
                m_data = realloc ( CAPACITY ( m_data ) + CAPACITY ( m_data ) / 2 );
            reference r = m_data [ SIZE ( m_data ) ] = v_;
            ++SIZE ( m_data );
            return r;
        }
        else {
            m_data = alloc ( 2 );
            reference r = m_data [ 0 ] = v_;
            SIZE ( m_data ) = 1;
            return r;
        }
    }
    [[ maybe_unused ]] reference emplace_back ( value_type && v_ ) {
        if ( m_data ) {
            if ( SIZE ( m_data ) == CAPACITY ( m_data ) )
                m_data = realloc ( CAPACITY ( m_data ) + CAPACITY ( m_data ) / 2 );
            reference r = m_data [ SIZE ( m_data ) ] = std::move ( v_ );
            ++SIZE ( m_data );
            return r;
        }
        else {
            m_data = alloc ( 2 );
            reference r = m_data [ 0 ] = std::move ( v_ );
            SIZE ( m_data ) = 1;
            return r;
        }
    }

    [[ nodiscard ]] pointer data ( ) noexcept {
        return m_data;
    }
    [[ nodiscard ]] const_pointer data ( ) const noexcept {
        return m_data;
    }

    [[ nodiscard ]] reference back ( ) noexcept {
        return m_data + SIZE ( m_data ) - 1;
    }
    [[ nodiscard ]] const_reference back ( ) const noexcept {
        return m_data + SIZE ( m_data ) - 1;
    }

    [[ nodiscard ]] size_type size ( ) const noexcept {
        return SIZE ( m_data );
    }
    [[ nodiscard ]] size_type capacity ( ) const noexcept {
        return CAPACITY ( m_data );
    }

    [[ nodiscard ]] bool empty ( ) const noexcept {
        return not ( SIZE ( m_data ) );
    }

    void resize ( const size_type n_ ) {
        if ( capacity ( ) < n_ )
            m_data = realloc ( n_ );
        else
            SIZE ( m_data ) = n_;
    }

    void clear ( ) noexcept {
        SIZE ( m_data ) = 0;
    }

    void reserve ( const size_type n_ ) {
        if ( capacity ( ) < n_ )
            m_data = realloc ( n_ );
    }

    [[ nodiscard ]] bool operator == ( const sintor & rhs_ ) const noexcept {
        if ( size ( ) != rhs_.size ( ) )
            return false;
        return not ( std::memcmp ( data ( ), rhs_.data ( ), sizeof ( value_type ) * size ( ) ) );
    }
    [[ nodiscard ]] bool operator != ( const sintor & rhs_ ) const noexcept {
        if ( size ( ) != rhs_.size ( ) )
            return true;
        return std::memcmp ( data ( ), rhs_.data ( ), sizeof ( value_type ) * size ( ) );
    }

    [[ nodiscard ]] iterator begin ( ) noexcept { return iterator ( m_data ); }
    [[ nodiscard ]] const_iterator begin ( ) const noexcept { return const_iterator ( m_data ); }
    [[ nodiscard ]] const_iterator cbegin ( ) const noexcept { return const_iterator ( m_data ); }

    [[ nodiscard ]] iterator end ( ) noexcept { return iterator ( m_data + SIZE ( m_data ) ); }
    [[ nodiscard ]] const_iterator end ( ) const noexcept { return const_iterator ( m_data + SIZE ( m_data ) ); }
    [[ nodiscard ]] const_iterator cend ( ) const noexcept { return const_iterator ( m_data + SIZE ( m_data ) ); }

    [[ nodiscard ]] iterator rbegin ( ) noexcept { return iterator ( m_data + SIZE ( m_data ) - 1 ); }
    [[ nodiscard ]] const_iterator rbegin ( ) const noexcept { return const_iterator ( m_data + SIZE ( m_data ) - 1 ); }
    [[ nodiscard ]] const_iterator crbegin ( ) const noexcept { return const_iterator ( m_data + SIZE ( m_data ) - 1 ); }

    [[ nodiscard ]] iterator rend ( ) noexcept { return iterator ( m_data - 1 ); }
    [[ nodiscard ]] const_iterator rend ( ) const noexcept { return const_iterator ( m_data - 1 ); }
    [[ nodiscard ]] const_iterator crend ( ) const noexcept { return const_iterator ( m_data - 1 ); }


    static size_type max_size ( ) noexcept {
        return std::numeric_limits<difference_type>::max ( ) - 3;
    }

    private:

    [[ nodiscard ]] pointer alloc ( const size_type n_ ) const noexcept {
        pointer p = static_cast<pointer> ( ALIGNED_MALLOC ( alignof ( value_type ), sizeof ( value_type ) * ( n_ + 2 ) ) );
        p [ 0 ] = n_; p [ 1 ] = 0;
        return p + 2;
    }
    [[ nodiscard ]] pointer realloc ( const size_type n_ ) const noexcept {
        pointer p = static_cast<pointer> ( ALIGNED_REALLOC ( static_cast<void*> ( m_data - 2 ), alignof ( value_type ), sizeof ( value_type ) * ( n_ + 2 ) ) );
        p [ 0 ] = n_;
        return p + 2;
    }
    void free ( ) noexcept {
        if ( m_data )
            ALIGNED_FREE ( m_data - 2 );
    }
};


#undef SIZE
#undef CAPACITY

};


template<typename Stream, typename Container>
Stream & operator << ( Stream & out_, const Container & v_ ) noexcept {
    for ( const auto & v : v_ )
        out_ << v << ' ';
    out_ << '\b';
    return out_;
}
