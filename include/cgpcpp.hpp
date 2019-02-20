
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
#include <experimental/fixed_capacity_vector> // https://github.com/gnzlbg/static_vector
#include <filesystem>
#include <fstream>
#include <functional>
#include <sax/iostream.hpp>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

#include <absl/container/fixed_array.h>
#include <absl/container/inlined_vector.h>

#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>

#include <frozen/unordered_map.h>
#include <frozen/string.h>

// https://github.com/degski/Sax/

#include <sax/prng.hpp>
#include <sax/stl.hpp>
#include <sax/string_split.hpp>

#include "functions.hpp"


namespace cgp {

template<typename T>
using NodeArray = absl::FixedArray<T>;


template<typename Real>
struct Data {

    struct Record {
        stl::vector<Real> input;
        stl::vector<Real> output;
    };

    using DataSet = stl::vector<Record>;

    using iterator = typename DataSet::iterator;
    using const_iterator = typename DataSet::const_iterator;

    DataSet data;

    [[ nodiscard ]] iterator begin ( ) noexcept { return data.begin ( ); }
    [[ nodiscard ]] const_iterator begin ( ) const noexcept { return data.cbegin ( ); }
    [[ nodiscard ]] const_iterator cbegin ( ) const noexcept { return data.cbegin ( ); }

    [[ nodiscard ]] iterator end ( ) noexcept { return data.end ( ); }
    [[ nodiscard ]] const_iterator end ( ) const noexcept { return data.cend ( ); }
    [[ nodiscard ]] const_iterator cend ( ) const noexcept { return data.cend ( ); }

    [[ nodiscard ]] int stringToInt ( const std::string & s_ ) noexcept {
        int i;
        std::from_chars ( s_.data ( ), s_.data ( ) + s_.length ( ), i, 10 );
        return i;
    }

    [[ nodiscard ]] Real stringToReal ( const std::string & s_ ) noexcept {
        Real r;
        std::from_chars ( s_.data ( ), s_.data ( ) + s_.length ( ), r, std::chars_format::fixed );
        return r;
    }

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
            const int in_arity = stringToInt ( params [ 0 ] ), out_arity = stringToInt ( params [ 1 ] ), io_arity = out_arity + in_arity, num_records = stringToInt ( params [ 2 ] );
            data.clear ( );
            data.reserve ( num_records );
            while ( std::getline ( istream, line ) ) {
                auto back = data.emplace_back ( );
                const auto record = sax::string_split ( line, ",", " ", "\t" );
                if ( io_arity != record.size ( ) ) {
                    std::cout << "Error: the size " << record.size ( ) << " of the record on line " << data.size ( ) << " differs from the parameters size " << io_arity << nl << "Terminating CGPPP-Library" << nl;
                    std::abort ( );
                }
                back.input.reserve ( in_arity ); back.output.reserve ( out_arity );
                int i = 0;
                for ( ; i < in_arity; ++i ) {
                    back.input.emplace_back ( stringToReal ( record [ i ] ) );
                }
                for ( ; i < io_arity; ++i ) {
                    back.output.emplace_back ( stringToReal ( record [ i ] ) );
                }
            }
            if ( num_records != data.size ( ) ) {
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

using DataSet = typename Data<Float>::DataSet;

auto dataSet = [ ] { return detail::singletonDataSet.instance ( ).data; } ( );


// Forward declarations.

template<typename Real>
struct Parameters;

template<typename Real>
struct Node;

template<typename Real>
struct Chromosome;

template<typename Real>
using ChromosomePtr = std::unique_ptr<Chromosome<Real>>;
template<typename Real>
using ChromosomePtrVec = stl::vector<ChromosomePtr<Real>>;

template<typename Real>
void probabilisticMutation ( Chromosome<Real> & chromo_ ) noexcept;
template<typename Real>
Real supervisedLearning ( Chromosome<Real> & chromo_, const DataSet & data_ ) noexcept;
template<typename Real>
void selectFittest ( ChromosomePtrVec<Real> & parents_, ChromosomePtrVec<Real> && candidateChromos_ ) noexcept;
template<typename Real>
void mutateRandomParent ( ChromosomePtrVec<Real> & children_, const ChromosomePtrVec<Real> & parents_ ) noexcept;


template<typename Real>
struct Parameters {

    int mu;
    int lambda;
    char evolutionaryStrategy;

    private:

    Real mutationRate;
    std::bernoulli_distribution mutationDistribution;

    public:

    Real recurrentConnectionProbability;

    Real targetFitness;
    int updateFrequency;
    bool shortcutConnections;
    void ( * mutationType ) ( Chromosome<Real> & chromo_ );
    std::string mutationTypeName;
    Real ( * fitnessFunction ) ( Chromosome<Real> & chromo_, const DataSet & data_ );
    std::string fitnessFunctionName;
    void ( * selectionScheme ) ( ChromosomePtrVec<Real> & parents_, ChromosomePtrVec<Real> && candidateChromos_ );
    std::string selectionSchemeName;
    void ( * reproductionScheme )( ChromosomePtrVec<Real> & children_, const ChromosomePtrVec<Real> & parents_ );
    std::string reproductionSchemeName;

    int numInputs;
    int numNodes;
    int numOutputs;
    int arity;

    int numThreads;

    static thread_local sax::Rng prng;

    Parameters ( ) noexcept :

        mu { 1 },
        lambda { 4 },
        evolutionaryStrategy { '+' },
        mutationRate { Real { 0.05 } },
        mutationDistribution { mutationRate },
        recurrentConnectionProbability { Real { 0 } },
        targetFitness { Real { 0 } },
        updateFrequency { 1 },
        shortcutConnections { true },
        mutationType { probabilisticMutation },
        mutationTypeName { "probabilisticMutation" },
        fitnessFunction { supervisedLearning },
        fitnessFunctionName { "supervisedLearning" },
        selectionScheme { selectFittest },
        selectionSchemeName { "selectFittest" },
        reproductionScheme { mutateRandomParent },
        reproductionSchemeName { "mutateRandomParent" },
        numInputs { 0 },
        numNodes { 0 },
        numOutputs { 0 },
        arity { 0 },
        numThreads { 1 } {
    }

    [[ maybe_unused ]] Parameters & setDimensions ( const int numInputs_, const int numNodes_, const int numOutputs_, const int arity_ ) noexcept {
        numInputs = numInputs_;
        numNodes = numNodes_;
        numOutputs = numOutputs_;
        arity = arity_;
        if ( not ( hasValidatedParameters ( ) ) ) {
            std::cout << "\nError: Parameters not valid.\n\n";
            print ( );
            std::abort ( );
        }
        return * this;
    }

    // Validate the current parameters.
    [[ nodiscard ]] bool hasValidatedParameters ( ) const noexcept {
        return
            mu > 0 and
            lambda > 1 and
            ( evolutionaryStrategy == '+' or evolutionaryStrategy == ',' ) and
            mutationRate > Real { 0 } and
            numInputs > 0 and
            numNodes >= 0 and
            numOutputs > 0 and
            arity > 0;
    }

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        functionSet.addNodeFunction ( std::forward<Args> ( args_ ) ... );
        assert ( functionSet.numFunctions > 0 );
    }

    template<typename ... Args>
    void addCustomNodeFunction ( Args && ... args_ ) {
        functionSet.addCustomNodeFunction ( std::forward<Args> ( args_ ) ... );
    }

    // Mutation.

    void setMutationRate ( const Real mutationRate_ ) noexcept {
        assert ( mutationRate_ >= Real { 0 } and mutationRate_ <= Real { 1 } );
        mutationRate = mutationRate_;
        mutationDistribution = std::bernoulli_distribution { mutationRate };
    }

    [[ nodiscard ]] bool mutate ( ) const noexcept {
        return mutationDistribution ( FunctionSet<Real>::prng );
    }

    [[ nodiscard ]] int getRandomFunction ( ) const noexcept {
        return Parameters::randInt ( functionSet.numFunctions );
    }

    [[ nodiscard ]] int getRandomNodeInput ( const int nodePosition_ ) const noexcept {
        return std::bernoulli_distribution ( recurrentConnectionProbability ) ( FunctionSet<Real>::prng ) ?
            Parameters::randInt ( numNodes - nodePosition_ ) + nodePosition_ + numInputs :
            Parameters::randInt ( numInputs + nodePosition_ );
    }

    [[ nodiscard ]] int getRandomChromosomeOutput ( ) const noexcept {
        return shortcutConnections ? Parameters::randInt ( numInputs + numNodes ) : Parameters::randInt ( numNodes ) + numInputs;
    }

    // Run.

    void run ( ) noexcept {

        if ( hasValidatedParameters ( ) ) {

        }

        else {
            std::cout << "\nError: Parameters not valid.\n\n";
            print ( );
            std::abort ( );
        }
    }


    // Random generator.

    [[ nodiscard ]] static int randInt ( const int n_ ) noexcept {
        if ( not ( n_ ) )
            return 0;
        return std::uniform_int_distribution<int> ( 0, n_ - 1 ) ( FunctionSet<Real>::prng );
    }

    static void seedRng ( const std::uint64_t s_ = 0u ) noexcept {
        FunctionSet<Real>::seedRng ( s_ );
    }

    // Output.

    void print ( ) const noexcept {

        std::printf ( "-----------------------------------------------------------\n" );
        std::printf ( "                       Parameters                          \n" );
        std::printf ( "-----------------------------------------------------------\n" );
        std::printf ( "Evolutionary Strategy:\t\t\t(%d%c%d)-ES\n", mu, evolutionaryStrategy, lambda );
        std::printf ( "Inputs:\t\t\t\t\t%d\n", numInputs );
        std::printf ( "Nodes:\t\t\t\t\t%d\n", numNodes );
        std::printf ( "Outputs:\t\t\t\t%d\n", numOutputs );
        std::printf ( "Node Arity:\t\t\t\t%d\n", arity );
        std::printf ( "Mutation Type:\t\t\t\t%s\n", mutationTypeName.c_str ( ) );
        std::printf ( "Mutation rate:\t\t\t\t%f\n", mutationRate );
        std::printf ( "Recurrent Connection Probability:\t%f\n", recurrentConnectionProbability );
        std::printf ( "Shortcut Connections:\t\t\t%d\n", shortcutConnections );
        std::printf ( "Fitness Function:\t\t\t%s\n", fitnessFunctionName.c_str ( ) );
        std::printf ( "Target Fitness:\t\t\t\t%f\n", targetFitness );
        std::printf ( "Selection scheme:\t\t\t%s\n", selectionSchemeName.c_str ( ) );
        std::printf ( "Reproduction scheme:\t\t\t%s\n", reproductionSchemeName.c_str ( ) );
        std::printf ( "Update frequency:\t\t\t%d\n", updateFrequency );
        std::printf ( "Threads:\t\t\t%d\n", numThreads );
        functionSet.printActiveFunctionSet ( );
        std::printf ( "-----------------------------------------------------------\n\n" );
    }
};


namespace detail {
Parameters<Float> & params ( ) noexcept {
    static Parameters<Float> parameters;
    return parameters;
}
}

auto params = detail::params ( );

Parameters<Float> & initialize ( const int numInputs_, const int numNodes_, const int numOutputs_, const int arity_ ) noexcept {
    return params.setDimensions ( numInputs_, numNodes_, numOutputs_, arity_ );
}


template<typename Real>
struct Node {

    stl::vector<int> inputs;

    int function;
    bool active;
    Real output;
    int actArity;

    Node ( ) = delete;
    Node ( const Node & ) = default;
    Node ( Node && ) noexcept = default;
    explicit Node ( const int nodePosition_ ) :

        function { params.getRandomFunction ( ) },
        active { true },
        output { Real { 0 } },
        actArity { params.arity } {

        inputs.reserve ( params.arity );
        std::generate_n ( sax::back_emplacer ( inputs ), params.arity, [ nodePosition_ ] ( ) noexcept { return params.getRandomNodeInput ( nodePosition_ ); } );
    }

    [[ maybe_unused ]] Node & operator = ( const Node & ) = default;
    [[ maybe_unused ]] Node & operator = ( Node && ) noexcept = default;

    [[ nodiscard ]] bool operator == ( const Node & rhs_ ) const noexcept {
        function == rhs_.function;
        inputs == rhs_.inputs;
    }
    [[ nodiscard ]] bool operator != ( const Node & rhs_ ) const noexcept {
        return not ( operator == ( rhs_ ) );
    }

    void reset ( ) noexcept {
        output = Real { 0 };
    }

    friend class cereal::access;

    template<typename Archive>
    void serialize ( Archive & archive_ ) {
        archive_ ( inputs, function, active, output, actArity );
    }
};


template<typename Real>
struct Chromosome {

    stl::vector<Node<Real>> nodes;
    stl::vector<int> outputNodes;
    stl::vector<int> activeNodes;
    stl::vector<Real> outputValues;
    stl::vector<Real> nodeInputsHold;

    Real fitness;
    int generation;

    friend class cereal::access;

    template<typename Archive>
    void serialize ( Archive & archive_ ) {
        archive_ ( params.numInputs, params.numNodes, params.numOutputs, params.arity );
        archive_ ( functionSet );
        archive_ ( nodes, outputNodes, activeNodes );
    }

    void saveToFile ( fs::path && path_, std::string && file_name_ ) {
        std::ofstream ostream ( std::move ( path_ ) / ( std::move ( file_name_ ) + std::string ( ".chromo" ) ), std::ios::binary );
        {
            cereal::BinaryOutputArchive archive ( ostream );
            archive ( *this );
        }
        ostream.close ( );
    }

    void loadFromFile ( fs::path && path_, std::string && file_name_ ) {
        std::ifstream istream ( std::move ( path_ ) / ( std::move ( file_name_ ) + std::string ( ".chromo" ) ), std::ios::binary );
        {
            cereal::BinaryInputArchive archive ( istream );
            archive ( *this );
        }
        istream.close ( );
    }

    Chromosome ( ) :

        outputValues ( params.numOutputs ),
        nodeInputsHold ( params.arity ),
        fitness { Real { -1 } },
        generation { 0 } {

        nodes.reserve ( params.numNodes );
        std::generate_n ( sax::back_emplacer ( nodes ), params.numNodes, [ this ] ( ) noexcept { return Node<Real> ( static_cast< int > ( nodes.size ( ) ) ); } );

        outputNodes.reserve ( params.numOutputs );
        std::generate_n ( sax::back_emplacer ( outputNodes ), params.numOutputs, [ ] ( ) noexcept { return params.getRandomChromosomeOutput ( ); } );

        activeNodes.reserve ( params.numNodes );
        setChromosomeActiveNodes ( );
    }
    Chromosome ( const Chromosome & rhs_ ) :
        nodes ( rhs_.nodes ),
        outputNodes ( rhs_.outputNodes ),
        activeNodes ( rhs_.activeNodes ),
        fitness ( rhs_.fitness ),
        generation ( rhs_.generation ) {
    }

    Chromosome ( Chromosome && ) noexcept = default;

    [[ maybe_unused ]] Chromosome & operator = ( const Chromosome & rhs_ ) {
        nodes = rhs_.nodes;
        outputNodes = rhs_.outputNodes;
        activeNodes = rhs_.activeNodes;
        fitness = rhs_.fitness;
        generation = rhs_.generation;
        return * this;
    }

    [[ maybe_unused ]] Chromosome & operator = ( Chromosome && ) noexcept = default;

    [[ nodiscard ]] bool operator == ( const Chromosome & rhs_ ) const noexcept {
        return nodes == rhs_.nodes and outputNodes == rhs_.outputNodes;
    }
    [[ nodiscard ]] bool operator != ( const Chromosome & rhs_ ) const noexcept {
        return not ( operator == ( rhs_ ) );
    }

    // Executes the given chromosome.
    void execute ( const stl::vector<Real> & inputs_ ) noexcept {
        // For all of the active nodes.
        for ( const int currentActiveNode : activeNodes ) {
            const int nodeArity = nodes [ currentActiveNode ].actArity;
            // For each of the active nodes inputs.
            for ( int i = 0; i < nodeArity; ++i ) {
                // Gather the nodes input locations.
                const int nodeInputLocation = nodes [ currentActiveNode ].inputs [ i ];
                nodeInputsHold [ i ] = nodeInputLocation < params.numInputs ? inputs_ [ nodeInputLocation ] : nodes [ nodeInputLocation - params.numInputs ].output;
            }
            // Get the functionality of the active node under evaluation and
            // calculate the output of the active node under evaluation.
            nodes [ currentActiveNode ].output = functionSet.function [ nodes [ currentActiveNode ].function ] ( nodeInputsHold );
            // Deal with Real's becoming NAN.
            if ( std::isnan ( nodes [ currentActiveNode ].output ) ) {
                nodes [ currentActiveNode ].output = Real { 0 };
            }
            // Prevent Real's form going to inf and -inf.
            else if ( std::isinf ( nodes [ currentActiveNode ].output )) {
                nodes [ currentActiveNode ].output = nodes [ currentActiveNode ].output > Real { 0 } ? std::numeric_limits<Real>::max ( ) : std::numeric_limits<Real>::min ( );
            }
        }
        // Set the chromosome outputs.
        for ( int i = 0; i < params.numOutputs; ++i ) {
            outputValues [ i ] = outputNodes [ i ] < params.numInputs ? inputs_ [ outputNodes [ i ] ] : nodes [ outputNodes [ i ] - params.numInputs ].output;
        }
    }

    [[ nodiscard ]] std::unique_ptr<Chromosome> mutate ( ) const noexcept {
        auto mutated = std::make_unique<Chromosome> ( * this );
        params.mutationType ( * mutated );
        mutated->setChromosomeActiveNodes ( );
        return mutated;
    }

    void setFitness ( const DataSet & data_set_ ) noexcept {
        setChromosomeActiveNodes ( );
        for ( auto & node : nodes )
            node.reset ( );
        fitness = params.fitnessFunction ( * this, data_set_ );
    }

    // Set the active nodes in the given chromosome.
    void setChromosomeActiveNodes ( ) noexcept {
        // Reset the active nodes.
        activeNodes.clear ( );
        for ( auto & node : nodes )
            node.active = false;
        // Start the recursive search for active nodes from
        // the output nodes for the number of output nodes.
        for ( auto & nodeIndex : outputNodes ) {
            // If the output connects to a chromosome input, skip.
            if ( nodeIndex < params.numInputs )
                continue;
            // Begin a recursive search for active nodes.
            recursivelySetActiveNodes ( nodeIndex );
        }
        // Place active nodes in order.
        std::sort ( std::begin ( activeNodes ), std::end ( activeNodes ) );
    }

    // Used by setActiveNodes to recursively search for active nodes.
    void recursivelySetActiveNodes ( int nodeIndex_ ) noexcept {
        nodeIndex_ -= params.numInputs;
        // If the given node is an input or has already been flagged as active, stop.
        if ( nodeIndex_ < 0 or nodes [ nodeIndex_ ].active )
            return;
        // Log the node as active.
        activeNodes.push_back ( nodeIndex_ );
        nodes [ nodeIndex_ ].active = true;
        // Set the nodes actual arity.
        nodes [ nodeIndex_ ].actArity = getChromosomeNodeArity ( nodes [ nodeIndex_ ] );
        // Recursively log all the nodes to which the current nodes connect as active.
        for ( int i = 0; i < nodes [ nodeIndex_ ].actArity; ++i )
            recursivelySetActiveNodes ( nodes [ nodeIndex_ ].inputs [ i ] );
    }

    // Gets the chromosome node arity.
    [[ nodiscard ]] int getChromosomeNodeArity ( const Node<Real> & node_ ) {
        const int functionArity = functionSet.maxNumInputs [ node_.function ];
        return functionArity == -1 or params.arity < functionArity ? params.arity : functionArity;
    }

    // Output.

    void print ( ) noexcept {
        // Set the active nodes in the given chromosome.
        setChromosomeActiveNodes ( );
        // For all the chromo inputs.
        int i = 0;
        for ( ; i < params.numInputs; ++i ) {
            std::printf ( "(%d):\tinput\n", i );
        }
        // For all the hidden nodes.
        for ( auto & node : nodes ) {
            // Print the node function.
            std::printf ( "(%d):\t%s\t", i, functionSet.functionNames [ node.function ] );
            // For the arity of the node.
            for ( int j = 0; j < getChromosomeNodeArity ( node ); ++j ) {
                // Print the node input information.
                std::printf ( "%d ", node.inputs [ j ] );
            }
            // Highlight active nodes.
            if ( node.active )
                std::printf ( "*" );
            std::printf ( "\n" );
            ++i;
        }
        // For all of the outputs.
        std::printf ( "outputs: " );
        for ( const int outputNode : outputNodes ) {
            // Print the output node locations.
            std::printf ( "%d ", outputNode );
        }
        std::printf ( "\n\n" );
    }
};

template<typename Real>
struct Results {
    int numRuns;
    Chromosome<Real> **bestChromosomes;
};


// Conductions probabilistic mutation on the given chromosome. Each
// chromosome gene is changed to a random valid allele with a
// probability specified in parameters.
template<typename Real>
void probabilisticMutation ( Chromosome<Real> & chromo_ ) noexcept {
    int nodePosition = 0;
    for ( auto & node : chromo_.nodes ) {
        // Mutate the function gene.
        if ( params.mutate ( ) )
            node.function = params.getRandomFunction ( );
        for ( auto & input : node.inputs ) {
            if ( params.mutate ( ) )
                input = params.getRandomNodeInput ( nodePosition );
        }
        ++nodePosition;
    }
    for ( auto & output : chromo_.outputNodes ) {
        if ( params.mutate ( ) )
            output = params.getRandomChromosomeOutput ( );
    }
}


// The default fitness function used by CGP-Library. simply assigns an
// error of the sum of the absolute differences between the target and
// actual outputs for all outputs over all samples.
template<typename Real>
Real supervisedLearning ( Chromosome<Real> & chromo_, const DataSet & data_ ) noexcept {
    Real error = Real { 0 };
    for ( const auto & sample : data_ ) {
        chromo_.execute ( sample.input );
        error = std::inner_product ( std::begin ( chromo_.outputValues ), std::end ( chromo_.outputValues ), std::begin ( sample.output ), error, std::plus<> ( ), [ ] ( const Real a, const Real b ) noexcept { return std::abs ( a - b ); } );
     }
    return error;
}


// Selection scheme which selects the fittest members of the population
// to be the parents.
//
// The candidateChromos contains the current children followed by the
// current parents. This means that using a stable sort to order
// candidateChromos results in children being selected over parents if
// their fitnesses are equal. A desirable property in CGPPP to
// facilitate neutral genetic drift.
template<typename Real>
void selectFittest ( ChromosomePtrVec<Real> & parents_, ChromosomePtrVec<Real> && candidateChromos_ ) noexcept {
    std::stable_sort ( std::execution::par_unseq, std::begin ( candidateChromos_ ), std::end ( candidateChromos_ ), [ ] ( const ChromosomePtr<Real> & a, const ChromosomePtr<Real> & b ) noexcept { return a->fitness < b->fitness; } );
    candidateChromos_.resize ( params.mu );
    parents_ = std::move ( candidateChromos_ );
}

template<typename Real>
using chromos_iterator = typename ChromosomePtrVec<Real>::iterator;

template<typename Real>
void selectFittest ( chromos_iterator<Real> parents_begin_, chromos_iterator<Real> parents_end_, chromos_iterator<Real> candidates_begin_, chromos_iterator<Real> candidates_end_ ) noexcept {
    std::stable_sort ( std::execution::par_unseq, candidates_begin_, candidates_end_, [ ] ( const ChromosomePtr<Real> & a, const ChromosomePtr<Real> & b ) noexcept { return a->fitness < b->fitness; } );
}


// Mutate Random parent reproduction method.
template<typename Real>
void mutateRandomParent ( ChromosomePtrVec<Real> & children_, const ChromosomePtrVec<Real> & parents_ ) noexcept {
    children_.resize ( params.lambda );
    std::for_each ( std::execution::par_unseq, std::begin ( children_ ), std::end ( children_ ), [ & parents_ ] ( ChromosomePtr<Real> & child ) noexcept { child = parents_ [ Parameters<Real>::randInt ( parents_.size ( ) ) ]->mutate ( ); } );
}


template<typename Real>
ChromosomePtr<Real> runCGPPP ( const int numGens_ ) {

    // BestChromo found using runCGPPP.
    ChromosomePtr<Real> bestChromo;

    // Vectors of unique_ptr's to default constructed parents and children.
    ChromosomePtrVec<Real> parentChromos ( params.mu );
    ChromosomePtrVec<Real> childrenChromos ( params.lambda );

    // Set fitness of the parents.
    std::for_each ( std::execution::par_unseq, std::begin ( parentChromos ), std::end ( parentChromos ), [ ] ( ChromosomePtr<Real> & parent ) noexcept { parent->setFitness ( dataSet ); } );

    // show the user whats going on.
    if ( params.updateFrequency != 0 ) {
        std::printf ( "\n-- Starting CGPPP --\n\n" );
        std::printf ( "Gen\tfitness\n" );
    }

    int gen = 0;

    // for each generation.
    for ( gen = 0; gen < numGens_; ++gen ) {

        // set fitness of the children of the population.
        std::for_each ( std::execution::par_unseq, std::begin ( childrenChromos ), std::end ( childrenChromos ), [ ] ( ChromosomePtr<Real> & child ) { child->setFitness ( dataSet ); } );

        // Get best chromosome.
        const auto parentMin = std::min_element ( std::execution::par_unseq, std::begin ( parentChromos ), std::end ( parentChromos ), [ ] ( const ChromosomePtr<Real> & a, const ChromosomePtr<Real> & b ) noexcept { return a->fitness < b->fitness; } );
        const auto childMin = std::min_element ( std::execution::par_unseq, std::begin ( childrenChromos ), std::end ( childrenChromos ), [ ] ( const ChromosomePtr<Real> & a, const ChromosomePtr<Real> & b ) noexcept { return a->fitness < b->fitness; } );
        bestChromo = parentMin->fitness < childMin->fitness ? std::make_unique<Chromosome<Real>> ( * parentMin ) : std::make_unique<Chromosome<Real>> ( * childMin );

        // Check termination conditions.
        if ( bestChromo->fitness <= params.targetFitness ) {
            if ( params.updateFrequency != 0 )
                std::printf ( "%d\t%f - Solution Found\n", gen, bestChromo->fitness );
            break;
        }

        // Display progress to the user at the update frequency specified.
        if ( params.updateFrequency != 0 and ( gen % params.updateFrequency == 0 or gen >= numGens_ - 1 ) ) {
            std::printf ( "%d\t%f\n", gen, bestChromo->fitness );
        }

        // Select and reproduce.

        // Set the chromosomes which will be used by the selection scheme
        // dependant upon the evolutionary strategy. i.e. '+' all are used
        // by the selection scheme, ',' only the children are.
        if ( params.evolutionaryStrategy == '+' ) {

            // Note: the children are placed before the parents to
            // ensure 'new blood' is always selected over old if the
            // fitness are equal.

            std::move ( std::begin ( parentChromos ), std::end ( parentChromos ), sax::back_emplacer ( childrenChromos ) );
        }

        // Select the parents from the candidateChromos.
        parentChromos.clear ( );
        params.selectionScheme ( parentChromos, childrenChromos );

        // Create the children from the parents.
        childrenChromos.clear ( );
        params.reproductionScheme ( childrenChromos, parentChromos );
    }

    // Deal with formatting for displaying progress.
    if ( params.updateFrequency != 0 ) {
        std::printf ( "\n" );
    }

    bestChromo->generation = gen;

    return bestChromo;
}


} // namespace cgp


#if  0
/*
    This file is part of CGP-Library
    Copyright (c) Andrew James Turner 2014, 2015 (andrew.turner@york.ac.uk)

    CGP-Library is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CGP-Library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CGP-Library. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cfloat>

#include <algorithm>
#include <iostream>
#include <random>

#include "cgp.hpp"

/*
    Hard limits on the size of the function set
    and the names of various functions.
    (could make the function set size dynamic)
*/
#define FUNCTIONSETSIZE 50
#define FUNCTIONNAMELENGTH 11
#define FITNESSFUNCTIONNAMELENGTH 21
#define MUTATIONTYPENAMELENGTH 21
#define SELECTIONSCHEMENAMELENGTH 21
#define REPRODUCTIONSCHEMENAMELENGTH 21

/*
    Structure definitions
*/

struct parameters {
    int mu;
    int lambda;
    char evolutionaryStrategy;
    double mutationRate;
    double recurrentConnectionProbability;
    double connectionWeightRange;
    int numInputs;
    int numNodes;
    int numOutputs;
    int arity;
    struct functionSet *functionSet;
    double targetFitness;
    int updateFrequency;
    int shortcutConnections;
    void ( *mutationType )( struct parameters *params, struct chromosome *chromo );
    char mutationTypeName [ MUTATIONTYPENAMELENGTH ];
    double ( *fitnessFunction )( struct parameters *params, struct chromosome *chromo, struct dataSet *dat );
    char fitnessFunctionName [ FITNESSFUNCTIONNAMELENGTH ];
    void ( *selectionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **candidateChromos, int numParents, int numCandidateChromos );
    char selectionSchemeName [ SELECTIONSCHEMENAMELENGTH ];
    void ( *reproductionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **children, int numParents, int numChildren );
    char reproductionSchemeName [ REPRODUCTIONSCHEMENAMELENGTH ];
    int numThreads;
};

struct chromosome {
    int numInputs;
    int numOutputs;
    int numNodes;
    int numActiveNodes;
    int arity;
    struct node **nodes;
    int *outputNodes;
    int *activeNodes;
    double fitness;
    double *outputValues;
    struct functionSet *functionSet;
    double *nodeInputsHold;
    int generation;
};

struct node {
    int function;
    int *inputs;
    double *weights;
    int active;
    double output;
    int maxArity;
    int actArity;
};

struct functionSet {
    int numFunctions;
    char functionNames [ FUNCTIONSETSIZE ] [ FUNCTIONNAMELENGTH ];
    int maxNumInputs [ FUNCTIONSETSIZE ];
    double ( *functions [ FUNCTIONSETSIZE ] )( const int numInputs, const double *inputs, const double *connectionWeights );
};

struct dataSet {
    int numSamples;
    int numInputs;
    int numOutputs;
    double **inputData;
    double **outputData;
};

struct results {
    int numRuns;
    struct chromosome **bestChromosomes;
};



/*
    sets num chromosome inputs in parameters
*/
DLL_EXPORT void setNumInputs ( struct parameters *params, int numInputs ) {

    /* error checking */
    if ( numInputs <= 0 ) {
        printf ( "Error: number of chromosome inputs cannot be less than one; %d is invalid.\nTerminating CGP-Library.\n", numInputs );
        exit ( 0 );
    }

    params.numInputs = numInputs;
}


/*
    sets num chromosome nodes in parameters
*/
DLL_EXPORT void setNumNodes ( struct parameters *params, int numNodes ) {

    /* error checking */
    if ( numNodes < 0 ) {
        printf ( "Warning: number of chromosome nodes cannot be negative; %d is invalid.\nTerminating CGP-Library.\n", numNodes );
        exit ( 0 );
    }

    params.numNodes = numNodes;
}


/*
    sets num chromosome outputs in parameters
*/
DLL_EXPORT void setNumOutputs ( struct parameters *params, int numOutputs ) {

    /* error checking */
    if ( numOutputs < 0 ) {
        printf ( "Warning: number of chromosome outputs cannot be less than one; %d is invalid.\nTerminating CGP-Library.\n", numOutputs );
        exit ( 0 );
    }

    params.numOutputs = numOutputs;
}


/*
    sets chromosome arity in parameters
*/
DLL_EXPORT void setArity ( struct parameters *params, int arity ) {

    /* error checking */
    if ( arity < 0 ) {
        printf ( "Warning: node arity cannot be less than one; %d is invalid.\nTerminating CGP-Library.\n", arity );
        exit ( 0 );
    }

    params.arity = arity;
}


/*
    Sets the mu value in given parameters to the new given value. If mu value
    is invalid a warning is displayed and the mu value is left unchanged.
*/
DLL_EXPORT void setMu ( struct parameters *params, int mu ) {

    if ( mu > 0 ) {
        params.mu = mu;
    }
    else {
        printf ( "\nWarning: mu value '%d' is invalid. Mu value must have a value of one or greater. Mu value left unchanged as '%d'.\n", mu, params.mu );
    }
}


/*
    Sets the lambda value in given parameters to the new given value.
    If lambda value is invalid a warning is displayed and the lambda value
    is left unchanged.
*/
DLL_EXPORT void setLambda ( struct parameters *params, int lambda ) {

    if ( lambda > 0 ) {
        params.lambda = lambda;
    }
    else {
        printf ( "\nWarning: lambda value '%d' is invalid. Lambda value must have a value of one or greater. Lambda value left unchanged as '%d'.\n", lambda, params.lambda );
    }
}


/*
    Sets the evolutionary strategy given in parameters to '+' or ','.
    If an invalid option is given a warning is displayed and the evolutionary
    strategy is left unchanged.
*/
DLL_EXPORT void setEvolutionaryStrategy ( struct parameters *params, char evolutionaryStrategy ) {

    if ( evolutionaryStrategy == '+' or evolutionaryStrategy == ',' ) {
        params.evolutionaryStrategy = evolutionaryStrategy;
    }
    else {
        printf ( "\nWarning: the evolutionary strategy '%c' is invalid. The evolutionary strategy must be '+' or ','. The evolutionary strategy has been left unchanged as '%c'.\n", evolutionaryStrategy, params.evolutionaryStrategy );
    }
}


/*
    Sets the recurrent connection probability given in parameters. If an invalid
    value is given a warning is displayed and the value is left	unchanged.
*/
DLL_EXPORT void setRecurrentConnectionProbability ( struct parameters *params, double recurrentConnectionProbability ) {

    if ( recurrentConnectionProbability >= 0 and recurrentConnectionProbability <= 1 ) {
        params.recurrentConnectionProbability = recurrentConnectionProbability;
    }
    else {
        printf ( "\nWarning: recurrent connection probability '%f' is invalid. The recurrent connection probability must be in the range [0,1]. The recurrent connection probability has been left unchanged as '%f'.\n", recurrentConnectionProbability, params.recurrentConnectionProbability );
    }
}


/*
    Sets the whether shortcut connections are used. If an invalid
    value is given a warning is displayed and the value is left	unchanged.
*/
DLL_EXPORT void setShortcutConnections ( struct parameters *params, int shortcutConnections ) {

    if ( shortcutConnections == 0 or shortcutConnections == 1 ) {
        params.shortcutConnections = shortcutConnections;
    }
    else {
        printf ( "\nWarning: shortcut connection '%d' is invalid. The shortcut connections takes values 0 or 1. The shortcut connection has been left unchanged as '%d'.\n", shortcutConnections, params.shortcutConnections );
    }
}



/*
    sets the fitness function to the fitnessFunction passed. If the fitnessFunction is NULL
    then the default supervisedLearning fitness function is used.
*/
DLL_EXPORT void setCustomFitnessFunction ( struct parameters *params, double ( *fitnessFunction )( struct parameters *params, struct chromosome *chromo, struct dataSet *data ), char const *fitnessFunctionName ) {

    if ( fitnessFunction == NULL ) {
        params.fitnessFunction = supervisedLearning;
        strncpy ( params.fitnessFunctionName, "supervisedLearning", FITNESSFUNCTIONNAMELENGTH );
    }
    else {
        params.fitnessFunction = fitnessFunction;
        strncpy ( params.fitnessFunctionName, fitnessFunctionName, FITNESSFUNCTIONNAMELENGTH );
    }
}



/*
    sets the selection scheme used to select the parents from the candidate chromosomes. If the selectionScheme is NULL
    then the default selectFittest selection scheme is used.
*/
DLL_EXPORT void setCustomSelectionScheme ( struct parameters *params, void ( *selectionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **candidateChromos, int numParents, int numCandidateChromos ), char const *selectionSchemeName ) {

    if ( selectionScheme == NULL ) {
        params.selectionScheme = selectFittest;
        strncpy ( params.selectionSchemeName, "selectFittest", SELECTIONSCHEMENAMELENGTH );
    }
    else {
        params.selectionScheme = selectionScheme;
        strncpy ( params.selectionSchemeName, selectionSchemeName, SELECTIONSCHEMENAMELENGTH );
    }
}


/*
    sets the reproduction scheme used to select the parents from the candidate chromosomes. If the reproductionScheme is NULL
    then the default mutateRandomParent selection scheme is used.
*/

DLL_EXPORT void setCustomReproductionScheme ( struct parameters *params, void ( *reproductionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **children, int numParents, int numChildren ), char const *reproductionSchemeName ) {

    if ( reproductionScheme == NULL ) {
        params.reproductionScheme = mutateRandomParent;
        strncpy ( params.reproductionSchemeName, "mutateRandomParent", REPRODUCTIONSCHEMENAMELENGTH );
    }
    else {
        params.reproductionScheme = reproductionScheme;
        strncpy ( params.reproductionSchemeName, reproductionSchemeName, REPRODUCTIONSCHEMENAMELENGTH );
    }
}



/*
    sets the mutation type in params
*/
DLL_EXPORT void setMutationType ( struct parameters *params, char const *mutationType ) {

    if ( strncmp ( mutationType, "probabilistic", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params.mutationType = probabilisticMutation;
        strncpy ( params.mutationTypeName, "probabilistic", MUTATIONTYPENAMELENGTH );
    }

    else if ( strncmp ( mutationType, "point", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params.mutationType = pointMutation;
        strncpy ( params.mutationTypeName, "point", MUTATIONTYPENAMELENGTH );
    }


    else if ( strncmp ( mutationType, "onlyActive", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params.mutationType = probabilisticMutationOnlyActive;
        strncpy ( params.mutationTypeName, "onlyActive", MUTATIONTYPENAMELENGTH );
    }

    else if ( strncmp ( mutationType, "single", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params.mutationType = singleMutation;
        strncpy ( params.mutationTypeName, "single", MUTATIONTYPENAMELENGTH );
    }

    else {
        printf ( "\nWarning: mutation type '%s' is invalid. The mutation type must be 'probabilistic' or 'point'. The mutation type has been left unchanged as '%s'.\n", mutationType, params.mutationTypeName );
    }
}


/*
    Sets the update frequency in generations
*/
DLL_EXPORT void setUpdateFrequency ( struct parameters *params, int updateFrequency ) {

    if ( updateFrequency < 0 ) {
        printf ( "Warning: update frequency of %d is invalid. Update frequency must be >= 0. Update frequency is left unchanged as %d.\n", updateFrequency, params.updateFrequency );
    }
    else {
        params.updateFrequency = updateFrequency;
    }
}





/*
    chromosome function definitions
*/



/*
    Returns a pointer to an initialised chromosome with values obeying the given parameters.
*/
DLL_EXPORT struct chromosome *initialiseChromosomeFromChromosome ( struct chromosome *chromo ) {

    struct chromosome *chromoNew;
    int i;

    /* check that functionSet contains functions*/
    if ( chromo == NULL ) {
        printf ( "Error: cannot initialise chromosome from uninitialised chromosome.\nTerminating CGP-Library.\n" );
        exit ( 0 );
    }

    /* allocate memory for chromosome */
    chromoNew = ( struct chromosome* )std::malloc ( sizeof ( struct chromosome ) );

    /* allocate memory for nodes */
    chromoNew->nodes = ( struct node** )std::malloc ( chromo->numNodes * sizeof ( struct node* ) );

    /* allocate memory for outputNodes matrix */
    chromoNew->outputNodes = ( int* ) std::malloc ( chromo->numOutputs * sizeof ( int ) );

    /* allocate memory for active nodes matrix */
    chromoNew->activeNodes = ( int* ) std::malloc ( chromo->numNodes * sizeof ( int ) );

    /* allocate memory for chromosome outputValues */
    chromoNew->outputValues = ( double* ) std::malloc ( chromo->numOutputs * sizeof ( double ) );

    /* Initialise each of the chromosomes nodes */
    for ( i = 0; i < chromo->numNodes; i++ ) {
        chromoNew->nodes [ i ] = initialiseNode ( chromo->numInputs, chromo->numNodes, chromo->arity, chromo->functionSet->numFunctions, 0, 0, i );
        copyNode ( chromoNew->nodes [ i ], chromo->nodes [ i ] );
    }

    /* set each of the chromosomes outputs */
    for ( i = 0; i < chromo->numOutputs; i++ ) {
        chromoNew->outputNodes [ i ] = chromo->outputNodes [ i ];
    }

    /* set the number of inputs, nodes and outputs */
    chromoNew->numInputs = chromo->numInputs;
    chromoNew->numNodes = chromo->numNodes;
    chromoNew->numOutputs = chromo->numOutputs;
    chromoNew->arity = chromo->arity;


    /* copy over the chromsosme fitness */
    chromoNew->fitness = chromo->fitness;

    /* copy over the number of gnerations to find a solution */
    chromoNew->generation = chromo->generation;

    /* copy over the functionset */
    chromoNew->functionSet = ( struct functionSet* )std::malloc ( sizeof ( struct functionSet ) );
    copyFunctionSet ( chromoNew->functionSet, chromo->functionSet );

    /* set the active nodes in the newly generated chromosome */
    setChromosomeActiveNodes ( chromoNew );

    /* used internally by exicute chromosome */
    chromoNew->nodeInputsHold = ( double* ) std::malloc ( chromo->arity * sizeof ( double ) );

    return chromoNew;
}


/*
    used to access the chromosome outputs after executeChromosome
    has been called
*/
DLL_EXPORT double getChromosomeOutput ( struct chromosome *chromo, int output ) {

    if ( output < 0 or output > chromo->numOutputs ) {
        printf ( "Error: output less than or greater than the number of chromosome outputs. Called from getChromosomeOutput.\n" );
        exit ( 0 );
    }

    return chromo->outputValues [ output ];
}



/*
    used to access the chromosome node values after executeChromosome
    has been called
*/
DLL_EXPORT double getChromosomeNodeValue ( struct chromosome *chromo, int node ) {
    if ( node < 0 or node > chromo->numNodes ) {
        printf ( "Error: node less than or greater than the number of nodes  in chromosome. Called from getChromosomeNodeValue.\n" );
        exit ( 0 );
    }

    return chromo->nodes [ node ]->output;
}


DLL_EXPORT int compareChromosomesActiveNodes ( struct chromosome *chromoA, struct chromosome *chromoB ) {

    int i, j;

    /* ensure that the chromosomes don't point to NULL */
    if ( chromoA == NULL or chromoB == NULL ) {
        return 0;
    }

    /* Check the high level parameters */
    if ( chromoA->numInputs != chromoB->numInputs ) {
        return 0;
    }

    if ( chromoA->numNodes != chromoB->numNodes ) {
        return 0;
    }

    if ( chromoA->numOutputs != chromoB->numOutputs ) {
        return 0;
    }

    if ( chromoA->arity != chromoB->arity ) {
        return 0;
    }

    /* for each node*/
    for ( i = 0; i < chromoA->numNodes; i++ ) {

        /* if the node is active in both chromosomes */
        if ( chromoA->nodes [ i ]->active == 1 and chromoB->nodes [ i ]->active == 1 ) {

            /* Check the function genes */
            if ( chromoA->nodes [ i ]->function != chromoB->nodes [ i ]->function ) {
                return 0;
            }

            /* for each node input */
            for ( j = 0; j < chromoA->arity; j++ ) {

                /* Check the node inputs */
                if ( chromoA->nodes [ i ]->inputs [ j ] != chromoB->nodes [ i ]->inputs [ j ] ) {
                    return 0;
                }

                /* Check the connection weights inputs */
                if ( chromoA->nodes [ i ]->weights [ j ] != chromoB->nodes [ i ]->weights [ j ] ) {
                    return 0;
                }
            }
        }
        /* if the node is active in one chromosome */
        else if ( chromoA->nodes [ i ]->active != chromoB->nodes [ i ]->active ) {
            return 0;
        }

        /* The node is inactive in both chromosomes */
        else {
            /* do nothing */
        }
    }

    /* for all of the outputs */
    for ( i = 0; i < chromoA->numOutputs; i++ ) {

        /* Check the outputs */
        if ( chromoA->outputNodes [ i ] != chromoB->outputNodes [ i ] ) {
            return 0;
        }
    }

    return 1;
}



/*
    Mutates the given chromosome using the mutation method described in parameters
*/
DLL_EXPORT void mutateChromosome ( struct parameters *params, struct chromosome *chromo ) {

    params.mutationType ( params, chromo );

    setChromosomeActiveNodes ( chromo );
}




/*
    sets the fitness of the given chromosome
*/
DLL_EXPORT void setChromosomeFitness ( struct parameters *params, struct chromosome *chromo, struct dataSet *data ) {

    double fitness;

    setChromosomeActiveNodes ( chromo );

    resetChromosome ( chromo );

    fitness = params.fitnessFunction ( params, chromo, data );

    chromo->fitness = fitness;
}


/*
    reset the output values of all chromosome nodes to zero
*/
DLL_EXPORT void resetChromosome ( struct chromosome *chromo ) {

    int i;

    for ( i = 0; i < chromo->numNodes; i++ ) {
        chromo->nodes [ i ]->output = 0;
    }
}

/*
    Gets the number of chromosome inputs
*/
DLL_EXPORT int getNumChromosomeInputs ( struct chromosome *chromo ) {
    return chromo->numInputs;
}

/*
    Gets the number of chromosome nodes
*/
DLL_EXPORT int getNumChromosomeNodes ( struct chromosome *chromo ) {
    return chromo->numNodes;
}

/*
    Gets the number of chromosome active nodes
*/
DLL_EXPORT int getNumChromosomeActiveNodes ( struct chromosome *chromo ) {
    return chromo->numActiveNodes;
}

/*
    Gets the number of chromosome outputs
*/
DLL_EXPORT int getNumChromosomeOutputs ( struct chromosome *chromo ) {
    return chromo->numOutputs;
}

/*
    Gets the chromosome node arity
*/
DLL_EXPORT int getChromosomeNodeArity ( struct chromosome *chromo, int index ) {

    int chromoArity = chromo->arity;
    int maxArity = chromo->functionSet->maxNumInputs [ chromo->nodes [ index ]->function ];

    if ( maxArity == -1 ) {
        return chromoArity;
    }
    else if ( maxArity < chromoArity ) {
        return maxArity;
    }
    else {
        return chromoArity;
    }
}

/*
    Gets the chromosome fitness
*/
DLL_EXPORT double getChromosomeFitness ( struct chromosome *chromo ) {
    return chromo->fitness;
}

/*
    Gets the number of active connections in the given chromosome
*/
DLL_EXPORT int getNumChromosomeActiveConnections ( struct chromosome *chromo ) {

    int i;
    int complexity = 0;

    for ( i = 0; i < chromo->numActiveNodes; i++ ) {
        complexity += chromo->nodes [ chromo->activeNodes [ i ] ]->actArity;
    }

    return complexity;
}

/*
    Gets the number of generations required to find the given chromosome
*/
DLL_EXPORT int getChromosomeGenerations ( struct chromosome *chromo ) {
    return chromo->generation;
}



/*
    Dataset functions
*/


/*
    Initialises data structure and assigns values for given arrays
    arrays must take the form
    inputs[numSamples][numInputs]
    outputs[numSamples][numOutputs]
*/
DLL_EXPORT struct dataSet *initialiseDataSetFromArrays ( int numInputs, int numOutputs, int numSamples, double *inputs, double *outputs ) {

    int i, j;
    struct dataSet *data;

    /* initialise memory for data structure */
    data = ( struct dataSet* )std::malloc ( sizeof ( struct dataSet ) );

    data->numInputs = numInputs;
    data->numOutputs = numOutputs;
    data->numSamples = numSamples;

    data->inputData = ( double** ) std::malloc ( data->numSamples * sizeof ( double* ) );
    data->outputData = ( double** ) std::malloc ( data->numSamples * sizeof ( double* ) );

    for ( i = 0; i < data->numSamples; i++ ) {

        data->inputData [ i ] = ( double* ) std::malloc ( data->numInputs * sizeof ( double ) );
        data->outputData [ i ] = ( double* ) std::malloc ( data->numOutputs * sizeof ( double ) );

        for ( j = 0; j < data->numInputs; j++ ) {
            data->inputData [ i ] [ j ] = inputs [ ( i * data->numInputs ) + j ];
        }

        for ( j = 0; j < data->numOutputs; j++ ) {
            data->outputData [ i ] [ j ] = outputs [ ( i * data->numOutputs ) + j ];
        }
    }

    return data;
}


/*
    prints the given data structure to the screen
*/
DLL_EXPORT void printDataSet ( struct dataSet *data ) {

    int i, j;

    printf ( "DATA SET\n" );
    printf ( "Inputs: %d, ", data->numInputs );
    printf ( "Outputs: %d, ", data->numOutputs );
    printf ( "Samples: %d\n", data->numSamples );

    for ( i = 0; i < data->numSamples; i++ ) {

        for ( j = 0; j < data->numInputs; j++ ) {
            printf ( "%f ", data->inputData [ i ] [ j ] );
        }

        printf ( " : " );

        for ( j = 0; j < data->numOutputs; j++ ) {
            printf ( "%f ", data->outputData [ i ] [ j ] );
        }

        printf ( "\n" );
    }
}


/*
    saves dataset to file
*/
DLL_EXPORT void saveDataSet ( struct dataSet *data, char const *fileName ) {

    int i, j;
    FILE *fp;

    fp = fopen ( fileName, "w" );

    if ( fp == NULL ) {
        printf ( "Warning: cannot save data set to %s. Data set was not saved.\n", fileName );
        return;
    }

    fprintf ( fp, "%d,", data->numInputs );
    fprintf ( fp, "%d,", data->numOutputs );
    fprintf ( fp, "%d,", data->numSamples );
    fprintf ( fp, "\n" );


    for ( i = 0; i < data->numSamples; i++ ) {

        for ( j = 0; j < data->numInputs; j++ ) {
            fprintf ( fp, "%f,", data->inputData [ i ] [ j ] );
        }

        for ( j = 0; j < data->numOutputs; j++ ) {
            fprintf ( fp, "%f,", data->outputData [ i ] [ j ] );
        }

        fprintf ( fp, "\n" );
    }

    fclose ( fp );
}


/*
    returns the number of inputs for each sample in the given dataSet
*/
DLL_EXPORT int getNumDataSetInputs ( struct dataSet *data ) {
    return data->numInputs;
}


/*
    returns the number of outputs for each sample in the given dataSet
*/
DLL_EXPORT int getNumDataSetOutputs ( struct dataSet *data ) {
    return data->numOutputs;
}


/*
    returns the number of samples in the given dataSet
*/
DLL_EXPORT int getNumDataSetSamples ( struct dataSet *data ) {
    return data->numSamples;
}


/*
    returns the inputs of the given sample of the given dataSet
*/
DLL_EXPORT double *getDataSetSampleInputs ( struct dataSet *data, int sample ) {
    return data->inputData [ sample ];
}


/*
    returns the given input of the given sample of the given dataSet
*/
DLL_EXPORT double getDataSetSampleInput ( struct dataSet *data, int sample, int input ) {
    return data->inputData [ sample ] [ input ];
}


/*
    returns the outputs of the given sample of the given dataSet
*/
DLL_EXPORT double *getDataSetSampleOutputs ( struct dataSet *data, int sample ) {
    return data->outputData [ sample ];
}


/*
    returns the given output of the given sample of the given dataSet
*/
DLL_EXPORT double getDataSetSampleOutput ( struct dataSet *data, int sample, int output ) {
    return data->outputData [ sample ] [ output ];
}



/*
    Results Functions
*/


/*
    initialises a results structure
*/
struct results* initialiseResults ( struct parameters *params, int numRuns ) {

    struct results *rels;

    rels = ( struct results* )std::malloc ( sizeof ( struct results ) );
    rels->bestChromosomes = ( struct chromosome** )std::malloc ( numRuns * sizeof ( struct chromosome* ) );

    rels->numRuns = numRuns;

    /*
        Initialised chromosomes are returns from runCGP and stored in a results structure.
        Therefore they should not be initialised here.
    */

    return rels;
}


/*
    free a initialised results structure
*/
DLL_EXPORT void freeResults ( struct results *rels ) {

    int i;

    /* attempt to prevent user double freeing */
    if ( rels == NULL ) {
        printf ( "Warning: double freeing of results prevented.\n" );
        return;
    }

    for ( i = 0; i < rels->numRuns; i++ ) {
        freeChromosome ( rels->bestChromosomes [ i ] );
    }

    free ( rels->bestChromosomes );
    free ( rels );
}


/*
    saves results structure to file
*/
DLL_EXPORT void saveResults ( struct results *rels, char const *fileName ) {

    FILE *fp;
    int i;

    struct chromosome *chromoTemp;

    if ( rels == NULL ) {
        printf ( "Warning: cannot save uninitialised results structure. Results not saved.\n" );
        return;
    }

    fp = fopen ( fileName, "w" );

    if ( fp == NULL ) {
        printf ( "Warning: cannot open '%s' and so cannot save results to that file. Results not saved.\n", fileName );
        return;
    }

    fprintf ( fp, "Run,Fitness,Generations,Active Nodes\n" );

    for ( i = 0; i < rels->numRuns; i++ ) {

        chromoTemp = getChromosome ( rels, i );

        fprintf ( fp, "%d,%f,%d,%d\n", i, chromoTemp->fitness, chromoTemp->generation, chromoTemp->numActiveNodes );

        freeChromosome ( chromoTemp );
    }

    fclose ( fp );
}


/*
    Gets the number of chromosomes in the results structure
*/
DLL_EXPORT int getNumChromosomes ( struct results *rels ) {
    return rels->numRuns;
}


/*
    returns the average number of chromosome active nodes from repeated
    run results specified in rels.
*/
DLL_EXPORT double getAverageActiveNodes ( struct results *rels ) {

    int i;
    double avgActiveNodes = 0;
    struct chromosome *chromoTemp;

    for ( i = 0; i < getNumChromosomes ( rels ); i++ ) {

        chromoTemp = rels->bestChromosomes [ i ];

        avgActiveNodes += getNumChromosomeActiveNodes ( chromoTemp );
    }

    avgActiveNodes = avgActiveNodes / getNumChromosomes ( rels );

    return avgActiveNodes;
}


/*
    returns the median number of chromosome active nodes from repeated
    run results specified in rels.
*/
DLL_EXPORT double getMedianActiveNodes ( struct results *rels ) {

    int i;
    double medActiveNodes = 0;

    int *array = ( int* ) std::malloc ( getNumChromosomes ( rels ) * sizeof ( int ) );

    for ( i = 0; i < getNumChromosomes ( rels ); i++ ) {
        array [ i ] = getNumChromosomeActiveNodes ( rels->bestChromosomes [ i ] );
    }

    medActiveNodes = medianInt ( array, getNumChromosomes ( rels ) );

    free ( array );

    return medActiveNodes;
}



/*
    returns the average chromosome fitness from repeated
    run results specified in rels.
*/
DLL_EXPORT double getAverageFitness ( struct results *rels ) {

    int i;
    double avgFit = 0;
    struct chromosome *chromoTemp;


    for ( i = 0; i < getNumChromosomes ( rels ); i++ ) {

        chromoTemp = rels->bestChromosomes [ i ];

        avgFit += getChromosomeFitness ( chromoTemp );
    }

    avgFit = avgFit / getNumChromosomes ( rels );

    return avgFit;
}


/*
    returns the median chromosome fitness from repeated
    run results specified in rels.
*/
DLL_EXPORT double getMedianFitness ( struct results *rels ) {

    int i;
    double med = 0;

    double *array = ( double* ) std::malloc ( getNumChromosomes ( rels ) * sizeof ( double ) );

    for ( i = 0; i < getNumChromosomes ( rels ); i++ ) {
        array [ i ] = getChromosomeFitness ( rels->bestChromosomes [ i ] );
    }

    med = medianDouble ( array, getNumChromosomes ( rels ) );

    free ( array );

    return med;
}



/*
    returns the average number of generations used by each run  specified in rels.
*/
DLL_EXPORT double getAverageGenerations ( struct results *rels ) {

    int i;
    double avgGens = 0;
    struct chromosome *chromoTemp;

    for ( i = 0; i < getNumChromosomes ( rels ); i++ ) {

        chromoTemp = rels->bestChromosomes [ i ];

        avgGens += getChromosomeGenerations ( chromoTemp );
    }

    avgGens = avgGens / getNumChromosomes ( rels );

    return avgGens;
}


/*
    returns the median number of generations used by each run  specified in rels.
*/
DLL_EXPORT double getMedianGenerations ( struct results *rels ) {

    int i;
    double med = 0;

    int *array = ( int* ) std::malloc ( getNumChromosomes ( rels ) * sizeof ( int ) );

    for ( i = 0; i < getNumChromosomes ( rels ); i++ ) {
        array [ i ] = getChromosomeGenerations ( rels->bestChromosomes [ i ] );
    }

    med = medianInt ( array, getNumChromosomes ( rels ) );

    free ( array );

    return med;
}



/*
    returns a pointer to a copy of the best chromosomes found on the given run in rels.
*/
DLL_EXPORT struct chromosome* getChromosome ( struct results *rels, int run ) {

    struct chromosome *chromo;

    /* do some error checking */
    if ( rels == NULL ) {
        printf ( "Error: cannot get best chromosome from uninitialised results.\nTerminating CGP-Library.\n" );
        exit ( 0 );
    }

    chromo = initialiseChromosomeFromChromosome ( rels->bestChromosomes [ run ] );

    return chromo;
}



/*
    Conductions point mutation on the give chromosome. A predetermined
    number of chromosome genes are randomly selected and changed to
    a random valid allele. The number of mutations is the number of chromosome
    genes multiplied by the mutation rate. Each gene has equal probability
    of being selected.

    DO NOT USE WITH ANN
*/
static void pointMutation ( struct parameters *params, struct chromosome *chromo ) {

    int i;
    int numGenes;
    int numFunctionGenes, numInputGenes, numOutputGenes;
    int numGenesToMutate;
    int geneToMutate;
    int nodeIndex;
    int nodeInputIndex;

    /* get the number of each type of gene */
    numFunctionGenes = params.numNodes;
    numInputGenes = params.numNodes * params.arity;
    numOutputGenes = params.numOutputs;

    /* set the total number of chromosome genes */
    numGenes = numFunctionGenes + numInputGenes + numOutputGenes;

    /* calculate the number of genes to mutate */
    numGenesToMutate = ( int ) roundf ( numGenes * params.mutationRate );

    /* for the number of genes to mutate */
    for ( i = 0; i < numGenesToMutate; i++ ) {

        /* select a random gene */
        geneToMutate = randInt ( numGenes );

        /* mutate function gene */
        if ( geneToMutate < numFunctionGenes ) {

            nodeIndex = geneToMutate;

            chromo->nodes [ nodeIndex ]->function = getRandomFunction ( chromo->functionSet->numFunctions );
        }

        /* mutate node input gene */
        else if ( geneToMutate < numFunctionGenes + numInputGenes ) {

            nodeIndex = ( int ) ( ( geneToMutate - numFunctionGenes ) / chromo->arity );
            nodeInputIndex = ( geneToMutate - numFunctionGenes ) % chromo->arity;

            chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, nodeIndex, params.recurrentConnectionProbability );
        }

        /* mutate output gene */
        else {
            nodeIndex = geneToMutate - numFunctionGenes - numInputGenes;
            chromo->outputNodes [ nodeIndex ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params.shortcutConnections );
        }
    }
}


/*
    Conductions a single active mutation on the give chromosome.

    DO NOT USE WITH ANN
*/
static void singleMutation ( struct parameters *params, struct chromosome *chromo ) {

    int numFunctionGenes, numInputGenes, numOutputGenes;
    int numGenes;
    int geneToMutate;
    int nodeIndex;
    int nodeInputIndex;

    int mutatedActive = 0;
    int previousGeneValue;
    int newGeneValue;

    /* get the number of each type of gene */
    numFunctionGenes = params.numNodes;
    numInputGenes = params.numNodes * params.arity;
    numOutputGenes = params.numOutputs;

    /* set the total number of chromosome genes */
    numGenes = numFunctionGenes + numInputGenes + numOutputGenes;

    /* while active gene not mutated */
    while ( mutatedActive == 0 ) {

        /* select a random gene */
        geneToMutate = randInt ( numGenes );

        /* mutate function gene */
        if ( geneToMutate < numFunctionGenes ) {

            nodeIndex = geneToMutate;

            previousGeneValue = chromo->nodes [ nodeIndex ]->function;

            chromo->nodes [ nodeIndex ]->function = getRandomFunction ( chromo->functionSet->numFunctions );

            newGeneValue = chromo->nodes [ nodeIndex ]->function;

            if ( ( previousGeneValue != newGeneValue ) and ( chromo->nodes [ nodeIndex ]->active == 1 ) ) {
                mutatedActive = 1;
            }

        }

        /* mutate node input gene */
        else if ( geneToMutate < numFunctionGenes + numInputGenes ) {

            nodeIndex = ( int ) ( ( geneToMutate - numFunctionGenes ) / chromo->arity );
            nodeInputIndex = ( geneToMutate - numFunctionGenes ) % chromo->arity;

            previousGeneValue = chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ];

            chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, nodeIndex, params.recurrentConnectionProbability );

            newGeneValue = chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ];

            if ( ( previousGeneValue != newGeneValue ) and ( chromo->nodes [ nodeIndex ]->active == 1 ) ) {
                mutatedActive = 1;
            }
        }

        /* mutate output gene */
        else {
            nodeIndex = geneToMutate - numFunctionGenes - numInputGenes;

            previousGeneValue = chromo->outputNodes [ nodeIndex ];

            chromo->outputNodes [ nodeIndex ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params.shortcutConnections );

            newGeneValue = chromo->outputNodes [ nodeIndex ];

            if ( previousGeneValue != newGeneValue ) {
                mutatedActive = 1;
            }
        }
    }
}



/*
    Conductions probabilistic mutation on the active nodes in the given
    chromosome. Each chromosome gene is changed to a random valid allele
    with a probability specified in parameters.
*/
static void probabilisticMutationOnlyActive ( struct parameters *params, struct chromosome *chromo ) {

    int i, j;
    int activeNode;

    /* for every active node in the chromosome */
    for ( i = 0; i < chromo->numActiveNodes; i++ ) {

        activeNode = chromo->activeNodes [ i ];

        /* mutate the function gene */
        if ( randDecimal ( ) <= params.mutationRate ) {
            chromo->nodes [ activeNode ]->function = getRandomFunction ( chromo->functionSet->numFunctions );
        }

        /* for every input to each chromosome */
        for ( j = 0; j < params.arity; j++ ) {

            /* mutate the node input */
            if ( randDecimal ( ) <= params.mutationRate ) {
                chromo->nodes [ activeNode ]->inputs [ j ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, activeNode, params.recurrentConnectionProbability );
            }

            /* mutate the node connection weight */
            if ( randDecimal ( ) <= params.mutationRate ) {
                chromo->nodes [ activeNode ]->weights [ j ] = getRandomConnectionWeight ( params.connectionWeightRange );
            }
        }
    }

    /* for every chromosome output */
    for ( i = 0; i < params.numOutputs; i++ ) {

        /* mutate the chromosome output */
        if ( randDecimal ( ) <= params.mutationRate ) {
            chromo->outputNodes [ i ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params.shortcutConnections );
        }
    }
}


/*
    repetitively applies runCGP to obtain average behaviour
*/
DLL_EXPORT struct results* repeatCGP ( struct parameters *params, struct dataSet *data, int numGens, int numRuns ) {

    int i;
    struct results *rels;
    int updateFrequency = params.updateFrequency;

    /* set the update frequency so as to to so generational results */
    params.updateFrequency = 0;

    rels = initialiseResults ( params, numRuns );

    printf ( "Run\tFitness\t\tGenerations\tActive Nodes\n" );

    /* for each run */
#pragma omp parallel for default(none), shared(numRuns,rels,params,data,numGens), schedule(dynamic), num_threads(params.numThreads)
    for ( i = 0; i < numRuns; i++ ) {

        /* run cgp */
        rels->bestChromosomes [ i ] = runCGP ( params, data, numGens );

        printf ( "%d\t%f\t%d\t\t%d\n", i, rels->bestChromosomes [ i ]->fitness, rels->bestChromosomes [ i ]->generation, rels->bestChromosomes [ i ]->numActiveNodes );
    }

    printf ( "----------------------------------------------------\n" );
    printf ( "MEAN\t%f\t%f\t%f\n", getAverageFitness ( rels ), getAverageGenerations ( rels ), getAverageActiveNodes ( rels ) );
    printf ( "MEDIAN\t%f\t%f\t%f\n", getMedianFitness ( rels ), getMedianGenerations ( rels ), getMedianActiveNodes ( rels ) );
    printf ( "----------------------------------------------------\n\n" );

    /* restore the original value for the update frequency */
    params.updateFrequency = updateFrequency;

    return rels;
}


DLL_EXPORT struct chromosome* runCGP ( struct parameters *params, struct dataSet *data, int numGens ) {

    int i;
    int gen;

    /* bestChromo found using runCGP */
    struct chromosome *bestChromo;

    /* arrays of the parents and children */
    struct chromosome **parentChromos;
    struct chromosome **childrenChromos;

    /* storage for chromosomes used by selection scheme */
    struct chromosome **candidateChromos;
    int numCandidateChromos;

    /* error checking */
    if ( numGens < 0 ) {
        printf ( "Error: %d generations is invalid. The number of generations must be >= 0.\n Terminating CGP-Library.\n", numGens );
        exit ( 0 );
    }

    if ( data != NULL and params.numInputs != data->numInputs ) {
        printf ( "Error: The number of inputs specified in the dataSet (%d) does not match the number of inputs specified in the parameters (%d).\n", data->numInputs, params.numInputs );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    if ( data != NULL and params.numOutputs != data->numOutputs ) {
        printf ( "Error: The number of outputs specified in the dataSet (%d) does not match the number of outputs specified in the parameters (%d).\n", data->numOutputs, params.numOutputs );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    /* initialise parent chromosomes */
    parentChromos = ( struct chromosome** )std::malloc ( params.mu * sizeof ( struct chromosome* ) );

    for ( i = 0; i < params.mu; i++ ) {
        parentChromos [ i ] = initialiseChromosome ( params );
    }

    /* initialise children chromosomes */
    childrenChromos = ( struct chromosome** )std::malloc ( params.lambda * sizeof ( struct chromosome* ) );

    for ( i = 0; i < params.lambda; i++ ) {
        childrenChromos [ i ] = initialiseChromosome ( params );
    }

    /* intilise best chromosome */
    bestChromo = initialiseChromosome ( params );

    /* determine the size of the Candidate Chromos based on the evolutionary Strategy */
    if ( params.evolutionaryStrategy == '+' ) {
        numCandidateChromos = params.mu + params.lambda;
    }
    else if ( params.evolutionaryStrategy == ',' ) {
        numCandidateChromos = params.lambda;
    }
    else {
        printf ( "Error: the evolutionary strategy '%c' is not known.\nTerminating CGP-Library.\n", params.evolutionaryStrategy );
        exit ( 0 );
    }

    /* initialise the candidateChromos */
    candidateChromos = ( struct chromosome** )std::malloc ( numCandidateChromos * sizeof ( struct chromosome* ) );

    for ( i = 0; i < numCandidateChromos; i++ ) {
        candidateChromos [ i ] = initialiseChromosome ( params );
    }

    /* set fitness of the parents */
    for ( i = 0; i < params.mu; i++ ) {
        setChromosomeFitness ( params, parentChromos [ i ], data );
    }

    /* show the user whats going on */
    if ( params.updateFrequency != 0 ) {
        printf ( "\n-- Starting CGP --\n\n" );
        printf ( "Gen\tfitness\n" );
    }

    /* for each generation */
    for ( gen = 0; gen < numGens; gen++ ) {

        /* set fitness of the children of the population */
    #pragma omp parallel for default(none), shared(params, childrenChromos,data), schedule(dynamic), num_threads(params.numThreads)
        for ( i = 0; i < params.lambda; i++ ) {
            setChromosomeFitness ( params, childrenChromos [ i ], data );
        }

        /* get best chromosome */
        getBestChromosome ( parentChromos, childrenChromos, params.mu, params.lambda, bestChromo );

        /* check termination conditions */
        if ( getChromosomeFitness ( bestChromo ) <= params.targetFitness ) {

            if ( params.updateFrequency != 0 ) {
                printf ( "%d\t%f - Solution Found\n", gen, bestChromo->fitness );
            }

            break;
        }

        /* display progress to the user at the update frequency specified */
        if ( params.updateFrequency != 0 and ( gen % params.updateFrequency == 0 or gen >= numGens - 1 ) ) {
            printf ( "%d\t%f\n", gen, bestChromo->fitness );
        }

        /*
            Set the chromosomes which will be used by the selection scheme
            dependant upon the evolutionary strategy. i.e. '+' all are used
            by the selection scheme, ',' only the children are.
        */
        if ( params.evolutionaryStrategy == '+' ) {

            /*
                Note: the children are placed before the parents to
                ensure 'new blood' is always selected over old if the
                fitness are equal.
            */

            for ( i = 0; i < numCandidateChromos; i++ ) {

                if ( i < params.lambda ) {
                    copyChromosome ( candidateChromos [ i ], childrenChromos [ i ] );
                }
                else {
                    copyChromosome ( candidateChromos [ i ], parentChromos [ i - params.lambda ] );
                }
            }
        }
        else if ( params.evolutionaryStrategy == ',' ) {

            for ( i = 0; i < numCandidateChromos; i++ ) {
                copyChromosome ( candidateChromos [ i ], childrenChromos [ i ] );
            }
        }

        /* select the parents from the candidateChromos */
        params.selectionScheme ( params, parentChromos, candidateChromos, params.mu, numCandidateChromos );

        /* create the children from the parents */
        params.reproductionScheme ( params, parentChromos, childrenChromos, params.mu, params.lambda );
    }

    /* deal with formatting for displaying progress */
    if ( params.updateFrequency != 0 ) {
        printf ( "\n" );
    }

    /* copy the best best chromosome */
    bestChromo->generation = gen;
    /*copyChromosome(chromo, bestChromo);*/

    /* free parent chromosomes */
    for ( i = 0; i < params.mu; i++ ) {
        freeChromosome ( parentChromos [ i ] );
    }
    free ( parentChromos );

    /* free children chromosomes */
    for ( i = 0; i < params.lambda; i++ ) {
        freeChromosome ( childrenChromos [ i ] );
    }
    free ( childrenChromos );

    /* free the used chromosomes and population */
    for ( i = 0; i < numCandidateChromos; i++ ) {
        freeChromosome ( candidateChromos [ i ] );
    }
    free ( candidateChromos );

    return bestChromo;
}

/*
    returns a pointer to the fittest chromosome in the two arrays of chromosomes

    loops through parents and then the children in order for the children to always be selected over the parents
*/
static void getBestChromosome ( struct chromosome **parents, struct chromosome **children, int numParents, int numChildren, struct chromosome *best ) {

    int i;
    struct chromosome *bestChromoSoFar;

    bestChromoSoFar = parents [ 0 ];

    for ( i = 1; i < numParents; i++ ) {

        if ( parents [ i ]->fitness <= bestChromoSoFar->fitness ) {
            bestChromoSoFar = parents [ i ];
        }
    }

    for ( i = 0; i < numChildren; i++ ) {

        if ( children [ i ]->fitness <= bestChromoSoFar->fitness ) {
            bestChromoSoFar = children [ i ];
        }
    }

    copyChromosome ( best, bestChromoSoFar );
}



/*
    mutate Random parent reproduction method.
*/
static void mutateRandomParent ( struct parameters *params, struct chromosome **parents, struct chromosome **children, int numParents, int numChildren ) {

    int i;

    /* for each child */
    for ( i = 0; i < numChildren; i++ ) {

        /* set child as clone of random parent */
        copyChromosome ( children [ i ], parents [ randInt ( numParents ) ] );

        /* mutate newly cloned child */
        mutateChromosome ( params, children [ i ] );
    }
}

#endif