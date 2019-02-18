
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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <fstream>
#include <sax/iostream.hpp>
#include <iterator>
#include <list>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "../include/cgpcpp.hpp"

#include <SFML/System.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/pector.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/csv.hpp>
#include <cereal/types/vector.hpp>


#include <cereal/types/string.hpp>

namespace fs = std::filesystem;

#include <sax/string_split.hpp>


#include <experimental/fixed_capacity_vector> // https://github.com/gnzlbg/static_vector

int main ( ) {

    auto p = cgp::initialize ( 2, 32, 1, 2 );

    p.setDimensions ( 2, 32, 1, 2 );

    cgp::FunctionSet<Float> fs;

    fs.addPresetNodeFunction ( "add" );
    fs.addPresetNodeFunction ( "sub" );
    fs.addPresetNodeFunction ( "mul" );

    // std::cout << sizeof ( cgp::Node<Float> ) << nl;
    // std::cout << sizeof ( cgp::Chromosome<Float> ) << nl;

    cgp::Data<Float> ds;

    ds.loadFromFile ( "../data/", "table.data" );


    return EXIT_SUCCESS;
}



#if 0


template<typename T>
void saveToFile ( const T & t_, fs::path && path_, std::string && file_name_ ) {
    std::ofstream ostream ( path_ / ( file_name_ + std::string ( ".txt" ) ), std::ios::out );
    {
        cereal::CSVOutputArchive archive ( ostream );
        archive ( t_ );
    }
    ostream << std::endl;
    ostream.close ( );
}

template<typename T>
void loadFromFile ( T & t_, fs::path && path_, std::string && file_name_ ) {
    std::ifstream istream ( path_ / ( file_name_ + std::string ( ".txt" ) ), std::ios::in );
    {
        cereal::CSVInputArchive archive ( istream );
        archive ( t_ );
    }
    istream.close ( );
}


int main ( ) {

    std::vector<int> output0 { 123, 456, 789 };
    std::vector<float> output1 { 1.2f, -2.5f / 196415, 3.9f };
    std::vector<std::string> output2 { "foo", "bar", "baz" };

    saveToFile ( output0, "z://", "testi" );

    std::vector<int> input0;
    loadFromFile ( input0, "z://", "testi" );

    for ( auto i : input0 ) {
        std::cout << i << ' ';
    }
    std::cout << nl;

    saveToFile ( output1, "z://", "testf" );

    std::vector<float> input1;
    loadFromFile ( input1, "z://", "testf" );

    for ( auto i : input1 ) {
        std::cout << i << ' ';
    }
    std::cout << nl;

    return EXIT_SUCCESS;
}

#endif

