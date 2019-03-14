
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
#include <utility>


using Float = float;


#if defined ( USE_PECTOR )
#include <pector/pector.h> // Use my fork at https://github.com/degski/pector, or hell will come upon you.
#include <pector/malloc_allocator.h>
#include <cereal/types/pector.hpp>
#else
#include <vector>
#include <cereal/types/vector.hpp>
#endif

#include <cereal/cereal.hpp>



namespace stl {

template<typename Type>
using is_signed_integral = std::conjunction<std::is_integral<Type>, std::is_signed<Type>>;

template<typename Type, typename SFINAE = typename std::enable_if<is_signed_integral<Type>::value>::type>
struct sintor;

}

namespace cereal {

template <typename Archive, typename T, typename SFINAE = typename std::enable_if<stl::is_signed_integral<T>::value>::type> inline
typename std::enable_if<traits::is_output_serializable<BinaryData<T>, Archive>::value and stl::is_signed_integral<T>::value, void>::type
    CEREAL_SAVE_FUNCTION_NAME ( Archive & ar, stl::sintor<T, SFINAE> const & sintor ) {
    ar ( make_size_tag ( static_cast<size_type> ( sintor.size ( ) ) ) ); // number of elements
    ar ( binary_data ( sintor.data ( ) - 2, ( sintor.size ( ) + 2 ) * sizeof ( T ) ) );
}

template <typename Archive, typename T, typename SFINAE = typename std::enable_if<stl::is_signed_integral<T>::value>::type> inline
typename std::enable_if<traits::is_input_serializable<BinaryData<T>, Archive>::value and stl::is_signed_integral<T>::value, void>::type
    CEREAL_LOAD_FUNCTION_NAME ( Archive & ar, stl::sintor<T, SFINAE> & sintor ) {
    typename stl::sintor<T, SFINAE>::size_type sintorSize;
    ar ( make_size_tag ( sintorSize ) );
    sintor.resize ( sintorSize );
    ar ( binary_data ( sintor.data ( ) - 2, ( static_cast<std::size_t> ( sintorSize ) + 2 ) * sizeof ( T ) ) );
}

} // namespace cereal


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


#define CAPACITY m_data [ -2 ]
#define SIZE m_data [ -1 ]

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
template<typename Type, typename SFINAE>
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
        SIZE = n_;
    }
    sintor ( sintor && other_ ) noexcept :
        m_data ( std::exchange ( other_.m_data, nullptr ) ) {
    }
    sintor ( const sintor & other_ ) :
        m_data ( alloc ( other_.size ( ) ) ) {
        SIZE = CAPACITY;
        std::memcpy ( m_data, other_.m_data, sizeof ( value_type ) * SIZE );
    }
    template<typename It>
    sintor ( It b_, It e_ ) :
         m_data ( alloc ( static_cast<size_type> ( std::distance ( b_, e_ ) ) ) ) {
         SIZE = CAPACITY;
         std::uninitialized_copy ( b_, e_, begin ( ) );
    }

    ~sintor ( ) {
        free ( );
    }

    [[ maybe_unused ]] sintor & operator = ( sintor && other_ ) noexcept {
        m_data = std::exchange ( other_.m_data, nullptr );
        return * this;
    }
    [[ maybe_unused ]] sintor & operator = ( const sintor & other_ ) {
        if ( CAPACITY < other_.size ( ) ) {
            free ( );
            m_data = alloc ( other_.size ( ) );
        }
        SIZE = other_.size ( );
        std::memcpy ( m_data, other_.m_data, sizeof ( value_type ) * SIZE );
        return * this;
    }

    // UB, iff sinter un-allocated, or out-of-bounds.
    [[ nodiscard ]] reference operator [ ] ( const size_type i_ ) noexcept {
        return m_data [ i_ ];
    }
    // UB, iff sinter un-allocated, or out-of-bounds.
    [[ nodiscard ]] const_reference operator [ ] ( const size_type i_ ) const noexcept {
        return m_data [ i_ ];
    }

    [[ maybe_unused ]] reference push_back ( const value_type & v_ ) {
        if ( m_data ) {
            if ( SIZE == CAPACITY )
                m_data = realloc ( CAPACITY + CAPACITY / 2 );
            return m_data [ SIZE++ ] = v_;
        }
        else {
            m_data = alloc ( 2 );
            SIZE = 1;
            return m_data [ 0 ] = v_;
        }
    }
    [[ maybe_unused ]] reference emplace_back ( value_type && v_ ) {
        if ( m_data ) {
            if ( SIZE == CAPACITY )
                m_data = realloc ( CAPACITY + CAPACITY / 2 );
            return m_data [ SIZE++ ] = std::move ( v_ );
        }
        else {
            m_data = alloc ( 2 );
            SIZE = 1;
            return m_data [ 0 ] = std::move ( v_ );
        }
    }

    [[ nodiscard ]] pointer data ( ) noexcept {
        return m_data;
    }
    [[ nodiscard ]] const_pointer data ( ) const noexcept {
        return m_data;
    }

    // UB, iff sinter un-allocated.
    [[ nodiscard ]] reference front ( ) noexcept {
        return m_data [ 0 ];
    }
    // UB, iff sinter un-allocated.
    [[ nodiscard ]] const_reference front ( ) const noexcept {
        return m_data [ 0 ];
    }

    // UB, iff sinter un-allocated.
    [[ nodiscard ]] reference back ( ) noexcept {
        return m_data [ SIZE - 1 ];
    }
    // UB, iff sinter un-allocated.
    [[ nodiscard ]] const_reference back ( ) const noexcept {
        return m_data [ SIZE - 1 ];
    }

    [[ nodiscard ]] size_type size ( ) const noexcept {
        return m_data ? SIZE : 0;
    }
    [[ nodiscard ]] size_type capacity ( ) const noexcept {
        return m_data ? CAPACITY : 0;
    }

    [[ nodiscard ]] bool empty ( ) const noexcept {
        return m_data ? not ( SIZE ) : true;
    }

    void resize ( const size_type n_ ) {
        if ( m_data ) {
            if ( CAPACITY < n_ )
                m_data = realloc ( n_ );
            else
                SIZE = n_;
        }
        else {
            m_data = alloc ( n_ );
            SIZE = n_;
        }
    }

    void clear ( ) noexcept {
        if ( m_data )
            SIZE = 0;
    }

    void reserve ( const size_type n_ ) {
        if ( m_data ) {
            if ( CAPACITY < n_ )
                m_data = realloc ( n_ );
        }
        else {
            m_data = alloc ( n_ );
            SIZE = n_;
        }
    }

    [[ nodiscard ]] bool operator == ( const sintor & rhs_ ) const noexcept {
        if ( not ( m_data ) and not ( rhs_.m_data ) )
            return true;
        else if ( ( m_data and not ( rhs_.m_data ) ) or ( not ( m_data ) and rhs_.m_data ) or ( SIZE != rhs_.size ( ) ) )
            return false;
        return not ( std::memcmp ( data ( ), rhs_.data ( ), sizeof ( value_type ) * SIZE ) );
    }
    [[ nodiscard ]] bool operator != ( const sintor & rhs_ ) const noexcept {
        return not ( operator == ( rhs_ ) );
    }

    [[ nodiscard ]] iterator begin ( ) noexcept { return static_cast<iterator> ( m_data ); }
    [[ nodiscard ]] const_iterator begin ( ) const noexcept { return static_cast<const_iterator> ( m_data ); }
    [[ nodiscard ]] const_iterator cbegin ( ) const noexcept { return static_cast<const_iterator> ( m_data ); }

    [[ nodiscard ]] iterator end ( ) noexcept { return static_cast<iterator> ( m_data + SIZE ); }
    [[ nodiscard ]] const_iterator end ( ) const noexcept { return static_cast<const_iterator> ( m_data + SIZE ); }
    [[ nodiscard ]] const_iterator cend ( ) const noexcept { return static_cast<const_iterator> ( m_data + SIZE ); }

    [[ nodiscard ]] iterator rbegin ( ) noexcept { return static_cast<iterator> ( m_data + SIZE - 1 ); }
    [[ nodiscard ]] const_iterator rbegin ( ) const noexcept { return static_cast<const_iterator> ( m_data + SIZE - 1 ); }
    [[ nodiscard ]] const_iterator crbegin ( ) const noexcept { return static_cast<const_iterator> ( m_data + SIZE - 1 ); }

    [[ nodiscard ]] iterator rend ( ) noexcept { return static_cast<iterator> ( m_data - 1 ); }
    [[ nodiscard ]] const_iterator rend ( ) const noexcept { return static_cast<const_iterator> ( m_data - 1 ); }
    [[ nodiscard ]] const_iterator crend ( ) const noexcept { return static_cast<const_iterator> ( m_data - 1 ); }


    [[ nodiscard ]] static constexpr size_type max_size ( ) noexcept {
        return std::numeric_limits<difference_type>::max ( ) - 3;
    }

    private:

    // Always set size after call to alloc.
    [[ nodiscard ]] pointer alloc ( const size_type n_ ) const noexcept {
        pointer p = static_cast<pointer> ( ALIGNED_MALLOC ( alignof ( value_type ), sizeof ( value_type ) * ( n_ + 2 ) ) );
        p [ 0 ] = static_cast<value_type> ( n_ );
        return p + 2;
    }
    // Size is copied together with the data.
    [[ nodiscard ]] pointer realloc ( const size_type n_ ) const noexcept {
        pointer p = static_cast<pointer> ( ALIGNED_REALLOC ( m_data - 2, alignof ( value_type ), sizeof ( value_type ) * ( n_ + 2 ) ) );
        p [ 0 ] = static_cast<value_type> ( n_ );
        return p + 2;
    }
    void free ( ) noexcept {
        if ( m_data )
            ALIGNED_FREE ( m_data - 2 );
    }
};


#undef ALIGNED_FREE
#undef ALIGNED_REALLOC
#undef ALIGNED_MALLOC

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
