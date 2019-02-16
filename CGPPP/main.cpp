
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
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#ifndef nl
#define nl '\n'
#endif

#include "../include/cgpcpp.hpp"

#include <SFML/System.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/pector.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>


namespace fs = std::filesystem;

template<typename T>
void saveToFile ( const T & t_, fs::path && path_, std::string && file_name_ ) {
    std::ofstream ostream ( path_ / ( file_name_ + std::string ( ".txt" ) ), std::ios::out );
    {
        cereal::JSONOutputArchive archive ( ostream );
        archive ( CEREAL_NVP ( t_ ) );
    }
    ostream.flush ( );
    ostream.close ( );
}

template<typename T>
void loadFromFile ( T & t_, fs::path && path_, std::string && file_name_ ) {
    std::ifstream istream ( path_ / ( file_name_ + std::string ( ".txt" ) ), std::ios::in );
    {
        cereal::JSONInputArchive archive ( istream );
        archive ( CEREAL_NVP ( t_ ) );
    }
    istream.close ( );
}


int main ( ) {

    std::vector<int> output { 1, 2, 3 };

    saveToFile ( output, "z://", "test" );

    /*

    auto p = cgp::initialize ( 2, 32, 1, 2 );

    p.setDimensions ( 2, 32, 1, 2 );

    cgp::FunctionSet<Float> fs;

    fs.addPresetNodeFunction ( "add" );

    std::cout << sizeof ( cgp::Node<Float> ) << nl;
    std::cout << sizeof ( cgp::Chromosome<Float> ) << nl;

    */

    return EXIT_SUCCESS;
}
