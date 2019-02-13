
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
#include <cstdio>
#include <cstdlib>

#include <experimental/fixed_capacity_vector> // https://github.com/gnzlbg/static_vector
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <absl/container/fixed_array.h>
#include <absl/container/inlined_vector.h>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

#include <frozen/unordered_map.h>
#include <frozen/string.h>


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
#if defined ( __GNUC__ )
#include <lehmer.hpp>       // https://github.com/degski/Sax/blob/master/lehmer.hpp
#else
#include <splitmix.hpp>     // https://github.com/degski/Sax/blob/master/splitmix.hpp
#endif
#endif

#ifndef nl
#define DEF_NL
#define nl '\n'
#endif

namespace cgp {

template<typename T>
using NodeArray = absl::FixedArray<T>;


namespace function {

// Node function defines in CGP-Library.
template<typename Real = float> Real f_add ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_sub ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_mul ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_divide ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_idiv ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_irem ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_negate ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_absolute ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_squareRoot ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_square ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_cube ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_power ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_exponential ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_sine ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_cosine ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_tangent ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_randFloat ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_constTwo ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_constOne ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_constZero ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_constPI ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_and ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_nand ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_or ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_nor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_xor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_xnor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_not ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_wire ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_sigmoid ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_gaussian ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_step ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_softsign ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
template<typename Real = float> Real f_hyperbolicTangent ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept;
}

template<typename Real = float>
using FunctionPointer = Real ( * ) ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ );


template<typename Real = float>
struct FunctionSet {

    private:

    struct FunctionData {
        FunctionPointer<Real> function;
        int maxNumInputs;
    };

    public:

    std::vector<frozen::string> functionNames;
    std::vector<FunctionPointer<Real>> function;
    std::vector<int> maxNumInputs;

    int numFunctions = 0;

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        ( addPresetNodeFunction ( args_ ), ... );
    }

    void addPresetNodeFunction ( const frozen::string & functionName_ ) {
        auto [ function, maxNumInputs ] { function_set.at ( functionName_ ) };
        addCustomNodeFunction ( functionName_, function, maxNumInputs );
    }

    void addCustomNodeFunction ( const frozen::string & functionName_, FunctionPointer<Real> function_, int maxNumInputs_ ) {
        functionNames.push_back ( functionName_ );
        function.push_back ( function_ );
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


template<typename Real = float>
struct DataSet {
    int numSamples;
    int numInputs;
    int numOutputs;
    Real **inputData;
    Real **outputData;
};


// Forward declarations.

template<typename Real = float>
struct Parameters;

template<typename Real = float>
struct Node;

template<typename Real = float>
struct Chromosome;

template<typename Real = float>
void probabilisticMutation ( const Parameters<Real> & params_, Chromosome<Real> & chromo_ ) noexcept;
template<typename Real>
Real supervisedLearning ( const Parameters<Real> & params_, Chromosome<Real> & chromo_, const DataSet<Real> & data_ );


template<typename Real>
struct Parameters {

    int mu;
    int lambda;
    char evolutionaryStrategy;
    private:
    Real mutationRate;
    public:
    std::bernoulli_distribution mutationDistribution;
    Real recurrentConnectionProbability;
    Real connectionWeightRange;
    int numInputs;
    int numNodes;
    int numOutputs;
    int arity;
    FunctionSet<Real> funcSet;
    Real targetFitness;
    int updateFrequency;
    int shortcutConnections;
    void ( *mutationType )( const Parameters & params_, Chromosome<Real> & chromo_ );
    std::string mutationTypeName;
    Real ( *fitnessFunction )( const Parameters & params_, Chromosome<Real> & chromo_, const DataSet<Real> & data_ );
    std::string fitnessFunctionName;
    //void ( *selectionScheme )( const Parameters & params_, Chromosome<Real> & *parents, Chromosome<Real> & *candidateChromos, int numParents, int numCandidateChromos );
    std::string selectionSchemeName;
    //void ( *reproductionScheme )( const Parameters & params_, Chromosome<Real> & *parents, Chromosome<Real> & *children, int numParents, int numChildren );
    std::string reproductionSchemeName;
    int numThreads;

    Parameters ( const int numInputs_, const int numNodes_, const int numOutputs_, const int arity_ ) noexcept :

        mu { 1 },
        lambda { 4 },
        evolutionaryStrategy { '+' },
        mutationRate { Real { 0.05 } },
        mutationDistribution { mutationRate },
        recurrentConnectionProbability { Real { 0 } },
        connectionWeightRange { Real { 1 } },
        numInputs { numInputs_ },
        numNodes { numNodes_ },
        numOutputs { numOutputs_ },
        arity { arity_ },
        targetFitness { Real { 0 } },
        updateFrequency { 1 },
        shortcutConnections { 1 },
        mutationType { probabilisticMutation },
        mutationTypeName { "probabilisticMutation" },
        fitnessFunction { supervisedLearning },
        fitnessFunctionName { "supervisedLearning" },
 //       selectionScheme { selectFittest },
        selectionSchemeName { "selectFittest" },
  //      reproductionScheme { mutateRandomParent },
        reproductionSchemeName { "mutateRandomParent" },
        numThreads { 1 } {

        assert ( numInputs > 0 );
        assert ( numNodes >= 0 );
        assert ( numOutputs > 0 );
        assert ( arity > 0 );
    }

    // Validate the current parameters.
    [[ nodiscard ]] bool validateParameters ( ) const noexcept {
        assert ( numInputs > 0 );
        assert ( numNodes >= 0 );
        assert ( numOutputs > 0 );
        assert ( arity > 0 );
        assert ( evolutionaryStrategy == '+' or evolutionaryStrategy == ',' );
    }

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        funcSet.addNodeFunction ( std::forward<Args> ( args_ ) ... );
        assert ( funcSet.numFunctions ( ) > 0 );
    }

    template<typename ... Args>
    void addCustomNodeFunction ( Args && ... args_ ) {
        funcSet.addCustomNodeFunction ( std::forward<Args> ( args_ ) ... );
    }

    // Mutation.

    void setMutationRate ( const Real mutationRate_ ) noexcept {
        assert ( mutationRate_ >= Real { 0 } and mutationRate_ <= Real { 1 } );
        mutationRate = mutationRate_;
        mutationDistribution = std::bernoulli_distribution { mutationRate };
    }

    [[ nodiscard ]] bool mutate ( ) const noexcept {
        return mutationDistribution ( Parameters::rng );
    }

    [[ nodiscard ]] int getRandomFunction ( ) const noexcept {
        return Parameters::randInt ( funcSet.numFunctions );
    }

    [[ nodiscard ]] Real getRandomConnectionWeight ( ) const noexcept {
        return std::uniform_real_distribution<Real> ( -connectionWeightRange, connectionWeightRange ) ( Parameters::rng );
    }

    [[ nodiscard ]] int getRandomNodeInput ( const Chromosome<Real> & chromo_, const int nodePosition_ ) const noexcept {
        return std::bernoulli_distribution ( recurrentConnectionProbability ) ( Parameters::rng ) ?
            Parameters::randInt ( chromo_.numNodes - nodePosition_ ) + nodePosition_ + chromo_.numInputs :
            Parameters::randInt ( chromo_.numInputs + nodePosition_ );
    }
    [[ nodiscard ]] int getRandomNodeInput ( const int nodePosition_ ) const noexcept {
        return std::bernoulli_distribution ( recurrentConnectionProbability ) ( Parameters::rng ) ?
            Parameters::randInt ( numNodes - nodePosition_ ) + nodePosition_ + numInputs :
            Parameters::randInt ( numInputs + nodePosition_ );
    }

    [[ nodiscard ]] int getRandomChromosomeOutput ( const Chromosome<Real> & chromo_ ) const noexcept {
        return shortcutConnections ? Parameters::randInt ( chromo_.numInputs + chromo_.numNodes ) : Parameters::randInt ( chromo_.numNodes ) + chromo_.numInputs;
    }
    [[ nodiscard ]] int getRandomChromosomeOutput ( ) const noexcept {
        return shortcutConnections ? Parameters::randInt ( numInputs + numNodes ) : Parameters::randInt ( numNodes ) + numInputs;
    }

    // Random generator.

    [[ nodiscard ]] static int randInt ( const int n_ ) noexcept {
        return std::uniform_int_distribution<int> ( 0, n_ - 1 ) ( Parameters::rng );
    }

    void seedRng ( const std::uint64_t s_ ) noexcept {
        rng.seed ( s_ );
    }

    #if M64
    #ifdef __GNUC__
        using Rng = mcg128_fast;
    #else
        using Rng = splitmix64;
    #endif
    #else
        using Rng = std::minstd_rand;
    #endif

    static Rng rng;

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
        std::printf ( "Connection weights range:\t\t+/- %f\n", connectionWeightRange );
        std::printf ( "Mutation Type:\t\t\t\t%s\n", mutationTypeName );
        std::printf ( "Mutation rate:\t\t\t\t%f\n", mutationRate );
        std::printf ( "Recurrent Connection Probability:\t%f\n", recurrentConnectionProbability );
        std::printf ( "Shortcut Connections:\t\t\t%d\n", shortcutConnections );
        std::printf ( "Fitness Function:\t\t\t%s\n", fitnessFunctionName );
        std::printf ( "Target Fitness:\t\t\t\t%f\n", targetFitness );
        std::printf ( "Selection scheme:\t\t\t%s\n", selectionSchemeName );
        std::printf ( "Reproduction scheme:\t\t\t%s\n", reproductionSchemeName );
        std::printf ( "Update frequency:\t\t\t%d\n", updateFrequency );
        std::printf ( "Threads:\t\t\t%d\n", numThreads );
        funcSet.print ( );
        std::printf ( "-----------------------------------------------------------\n\n" );
    }
};

#if M64
template<typename Real>
typename Parameters<Real>::Rng Parameters<Real>::rng { static_cast<std::uint64_t> ( std::random_device { } ( ) ) << 32 | static_cast<std::uint64_t> ( std::random_device { } ( ) ) };
#else
template<typename Real>
typename Parameters<Real>::Rng Parameters<Real>::rng { std::random_device { } ( ) };
#endif


template<typename Real>
struct Node {

    NodeArray<int> inputs;
    NodeArray<Real> weights;
    int function;
    bool active;
    Real output;
    int maxArity;
    int actArity;

    // Node ( ) { }
    Node ( const Parameters<Real> & params_, const int nodePosition_ ) :

        inputs { maxArity },
        weights { maxArity },
        function { params_.getRandomFunction ( ) },
        active { true },
        output { Real { 0 } },
        maxArity { params_.arity },
        actArity { params_.arity } {

        std::generate ( std::begin ( inputs ), std::end ( inputs ), params_.getRandomNodeInput ( nodePosition_ ) );
        std::generate ( std::begin ( weights ), std::end ( weights ), params_.getRandomConnectionWeight ( ) );
    }
};


template<typename Real>
struct Chromosome {

    int numInputs;
    int numOutputs;
    int numNodes;
    int numActiveNodes;
    int arity;
    int generation;
    std::vector<Node<Real>> nodes;
    std::vector<int> outputNodes;
    std::vector<int> activeNodes;
    std::vector<Real> outputValues;
    std::vector<Real> nodeInputsHold;
    const Parameters<Real> & params;
    Real fitness;

    // Chromosome ( ) { }
    Chromosome ( const Parameters<Real> & params_ ) :
        numInputs { params_.numInputs },
        numOutputs { params_.numOutputs },
        numNodes { params_.numNodes },
        numActiveNodes { numNodes },
        arity { params_.arity },
        generation { 0 },
        nodes { },
        outputNodes { },
        activeNodes { numActiveNodes },
        outputValues { numOutputs },
        nodeInputsHold { arity },
        params { params_ },
        fitness { Real { -1 } } {

        nodes.reserve ( numNodes );
        for ( int nodePosition = 0; nodePosition < numNodes; ++nodePosition )
            nodes.emplace_back ( params_, nodePosition );
        outputNodes.reserve ( numOutputs );
        std::generate_n ( std::back_inserter ( outputNodes ), numOutputs, params_.getRandomChromosomeOutput ( ) );

        setChromosomeActiveNodes ( );
    }


    // Executes the given chromosome.
    void execute ( const std::vector<Real> & inputs_ ) noexcept {
        // For all of the active nodes.
        for ( const int currentActiveNode : activeNodes ) {
            const int nodeArity = nodes [ currentActiveNode ].actArity;
            // For each of the active nodes inputs.
            for ( int i = 0; i < nodeArity; ++i ) {
                // Gather the nodes input locations.
                const int nodeInputLocation = nodes [ currentActiveNode ].inputs [ i ];
                nodeInputsHold [ i ] = nodeInputLocation < numInputs ? inputs_ [ nodeInputLocation ] : nodes [ nodeInputLocation - numInputs ].output;
            }
            // Get the functionality of the active node under evaluation.
            const int currentActiveNodeFunction = nodes [ currentActiveNode ].function;
            // calculate the output of the active node under evaluation.
            nodes [ currentActiveNode ].output = params.funcSet.functions [ currentActiveNodeFunction ] ( nodeInputsHold, nodes [ currentActiveNode ].weights );
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
        for ( int i = 0; i < numOutputs; ++i ) {
            outputValues [ i ] = outputNodes [ i ] < numInputs ? inputs_ [ outputNodes [ i ] ] : nodes [ outputNodes [ i ] - numInputs ].output;
        }
    }


    // Set the active nodes in the given chromosome.
    void setChromosomeActiveNodes ( ) noexcept {
        // Set the number of active nodes to zero.
        numActiveNodes = 0;
        // Reset the active nodes.
        for ( auto & node : nodes )
            node.active = false;
        // Start the recursive search for active nodes from
        // the output nodes for the number of output nodes.
        for ( auto & nodeIndex : outputNodes ) {
            // If the output connects to a chromosome input, skip.
            if ( nodeIndex < numInputs )
                continue;
            // Begin a recursive search for active nodes.
            recursivelySetActiveNodes ( nodeIndex );
        }
        // Place active nodes in order.
        std::sort ( std::begin ( activeNodes ), std::end ( activeNodes ) );
    }

    // Used by setActiveNodes to recursively search for active nodes.
    void recursivelySetActiveNodes ( int nodeIndex_ ) noexcept {
        nodeIndex_ -= numInputs;
        // If the given node is an input or has already been flagged as active, stop.
        if ( nodeIndex_ < 0 or nodes [ nodeIndex_ ].active )
            return;
        // Log the node as active.
        nodes [ nodeIndex_ ].active = true;
        activeNodes [ numActiveNodes ] = nodeIndex_;
        ++numActiveNodes;
        // Set the nodes actual arity.
        nodes [ nodeIndex_ ].actArity = getChromosomeNodeArity ( nodeIndex_ );
        // Recursively log all the nodes to which the current nodes connect as active.
        for ( int i = 0; i < nodes [ nodeIndex_ ].actArity; ++i )
            recursivelySetActiveNodes ( nodes [ nodeIndex_ ].inputs [ i ] );
    }

    // Gets the chromosome node arity.
    [[ nodiscard ]] int getChromosomeNodeArity ( const int index_ ) {
        const int functionArity = params.funcSet.maxNumInputs [ nodes [ index_ ].function ];
        return functionArity == -1 or arity < functionArity ? arity : functionArity;
    }
};

template<typename Real = float>
struct Results {
    int numRuns;
    Chromosome<Real> **bestChromosomes;
};


// Conductions probabilistic mutation on the given chromosome. Each
// chromosome gene is changed to a random valid allele with a
// probability specified in parameters.
template<typename Real>
void probabilisticMutation ( const Parameters<Real> & params_, Chromosome<Real> & chromo_ ) noexcept {
    int nodePosition = 0;
    for ( auto & node : chromo_.nodes ) {
        // mutate the function gene
        if ( params_.mutate ( ) )
            node.function = params_.getRandomFunction ( );
        for ( auto & input : node.inputs ) {
            if ( params_.mutate ( ) )
                input = params_.getRandomNodeInput ( chromo_, nodePosition );
        }
        for ( auto & weight : node.weights ) {
            if ( params_.mutate ( ) )
                weight = params_.getRandomConnectionWeight ( );
        }
        ++nodePosition;
    }
    for ( auto & output : chromo_.outputNodes ) {
        if ( params_.mutate ( ) )
            output = params_.getRandomChromosomeOutput ( chromo_ );
    }
}


// The default fitness function used by CGP-Library. simply assigns
// an error of the sum of the absolute differences between the target
// and actual outputs for all outputs over all samples.
template<typename Real>
Real supervisedLearning ( const Parameters<Real> & params_, Chromosome<Real> & chromo_, const DataSet<Real> & data_ ) {
    // error checking.
    if ( chromo_.numInputs != data_.numInputs )
        throw std::runtime_error ( "Error: the number of chromosome inputs must match the number of inputs specified in the dataSet." );
    if ( chromo_.numOutputs != data_.numOutputs )
        throw std::runtime_error ( "Error: the number of chromosome outputs must match the number of outputs specified in the dataSet." );
    Real error = Real { 0 };
    for ( int i = 0; i < data_.numSamples; ++i ) {
        // calculate the chromosome outputs for the set of inputs
        executeChromosome ( chromo_, data_.inputData [ i ] );
        // for each chromosome output
        for ( int j = 0; j < chromo_.numOutputs; ++j )
            error += std::abs ( chromo_.outputValues [ j ] - data_.outputData [ i, j ] );
    }
    return error;
}


// Returns the sum of the weighted inputs.
template<typename Real>
[[ nodiscard ]] Real sumWeigtedInputs ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::inner_product ( std::begin ( inputs_ ), std::end ( inputs_ ), std::begin ( connectionWeights_ ), Real { 0 } );
}


template<typename T>
[[ nodiscard ]] T median ( const std::vector<T> & vector_ ) {
    std::vector<T> copyVector { vector_ };
    auto median = std::next ( std::begin ( copyVector ), copyVector.size ( ) / 2 );
    std::nth_element ( std::begin ( copyVector ), median, std::end ( copyVector ) );
    return *median;
}


namespace function {

// Node function add. Returns the sum of all the inputs.
template<typename Real> Real f_add ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    Real sum = inputs_ [ 0 ];
    for ( i = 1; i < inputs_.size ( ); i++ ) {
        sum += inputs_ [ i ];
    }
    return sum;
}

// Node function sub. Returns the first input minus all remaining inputs_.
template<typename Real> Real f_sub ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    Real sum = inputs_ [ 0 ];
    for ( i = 1; i < inputs_.size ( ); i++ ) {
        sum -= inputs_ [ i ];
    }
    return sum;
}

// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> Real f_mul ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    Real multiplication = inputs_ [ 0 ];
    for ( i = 1; i < inputs_.size ( ); i++ ) {
        multiplication *= inputs_ [ i ];
    }
    return multiplication;
}

// Node function div. Returns the first input divided by the second input divided by the third input etc
template<typename Real> Real f_divide ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    Real divide = inputs_ [ 0 ];
    for ( i = 1; i < inputs_.size ( ); i++ ) {
        divide /= inputs_ [ i ];
    }
    return divide;
}

// Node function idiv.Returns the first input (cast to int) divided by the second input (cast to int),
// This function allows for integer arithmatic.
template<typename Real> Real f_idiv ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    if ( 0 != static_cast<int> ( inputs_ [ 1 ] ) )
        return static_cast<Real> ( static_cast<int> ( inputs_ [ 0 ] ) / static_cast<int> ( inputs_ [ 1 ] ) );
    return Real { 0 };
}

// Node function irem. Returns the remainder of the first input (cast to int) divided by the second input (cast to int),
// This function allows for integer arithmatic.
template<typename Real> Real f_irem ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    if ( 0 != static_cast<int> ( inputs_ [ 1 ] ) )
        return static_cast<Real> ( static_cast<int> ( inputs_ [ 0 ] ) % static_cast<int> ( inputs_ [ 1 ] ) );
    return Real { 0 };
}

// Node function abs. Returns the negation of the first input,
// This is useful if one doen't want to use the mathematically
// crazy sub function, then negate can be applied to add.
template<typename Real> Real f_negate ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return -inputs_ [ 0 ];
}

// Node function abs. Returns the absolute of the first input
template<typename Real> Real f_absolute ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::abs ( inputs_ [ 0 ] );
}

// Node function sqrt.  Returns the square root of the first input
template<typename Real> Real f_squareRoot ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::sqrt ( inputs_ [ 0 ] );
}

// Node function squ.  Returns the square of the first input
template<typename Real> Real f_square ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::pow ( inputs_ [ 0 ], 2 );
}

// Node function cub.  Returns the cube of the first input
template<typename Real> Real f_cube ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::pow ( inputs_ [ 0 ], 3 );
}

// Node function power.  Returns the first output to the power of the second
template<typename Real> Real f_power ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::pow ( inputs_ [ 0 ], inputs_ [ 1 ] );
}

// Node function exp.  Returns the exponential of the first input
template<typename Real> Real f_exponential ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::exp ( inputs_ [ 0 ] );
}

// Node function sin.  Returns the sine of the first input
template<typename Real> Real f_sine ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::sin ( inputs_ [ 0 ] );
}

// Node function cos.  Returns the cosine of the first input
template<typename Real> Real f_cosine ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::cos ( inputs_ [ 0 ] );
}

// Node function tan.  Returns the tangent of the first input
template<typename Real> Real f_tangent ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::tan ( inputs_ [ 0 ] );
}

// Node function one.  Always returns 1
template<typename Real> Real f_constTwo ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return 2.0;
}

// Node function one.  Always returns 1
template<typename Real> Real f_constOne ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return 1.0;
}

// Node function one.  Always returns 0
template<typename Real> Real f_constZero ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return 0.0;
}

// Node function one.  Always returns PI
template<typename Real> Real f_constPI ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return 3.141592653589793116;
}

// Node function rand.  Returns a random number between minus one and positive one
template<typename Real> Real f_randFloat ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::uniform_real_distribution<Real> ( -1.0, 1.0 ) ( Parameters<Real>::rng );
}

// Node function and. logical AND, returns '1' if all inputs_ are '1'
//    else, '0'
template<typename Real> Real f_and ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    for ( i = 0; i < inputs_.size ( ); i++ ) {
        if ( inputs_ [ i ] == 0.0 ) {
            return 0.0;
        }
    }
    return 1.0;
}

// Node function and. logical NAND, returns '0' if all inputs_ are '1'
//    else, '1'
template<typename Real> Real f_nand ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    for ( i = 0; i < inputs_.size ( ); i++ ) {
        if ( inputs_ [ i ] == 0.0 ) {
            return 1.0;
        }
    }
    return 0.0;
}

// Node function or. logical OR, returns '0' if all inputs_ are '0'
//    else, '1'
template<typename Real> Real f_or ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    for ( i = 0; i < inputs_.size ( ); i++ ) {
        if ( inputs_ [ i ] == 1.0 ) {
            return 1.0;
        }
    }
    return 0.0;
}

// Node function nor. logical NOR, returns '1' if all inputs_ are '0'
//    else, '0'
template<typename Real> Real f_nor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    for ( i = 0; i < inputs_.size ( ); i++ ) {
        if ( inputs_ [ i ] == 1.0 ) {
            return 0.0;
        }
    }
    return 1.0;
}

// Node function xor. logical XOR, returns '1' iff one of the inputs_ is '1'
//    else, '0'. AKA 'one hot'.
template<typename Real> Real f_xor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    int numOnes = 0;
    int out;
    for ( i = 0; i < inputs_.size ( ); i++ ) {
        if ( inputs_ [ i ] == 1 ) {
            numOnes++;
        }
        if ( numOnes > 1 ) {
            break;
        }
    }
    return numOnes == 1;
}

// Node function xnor. logical XNOR, returns '0' iff one of the inputs_ is '1'
//    else, '1'.
template<typename Real> Real f_xnor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    int i;
    int numOnes = 0;
    for ( i = 0; i < inputs_.size ( ); i++ ) {
        if ( inputs_ [ i ] == 1 ) {
            numOnes++;
        }
        if ( numOnes > 1 ) {
            break;
        }
    }
    return numOnes != 1;
}

// Node function not. logical NOT, returns '1' if first input is '0', else '1'
template<typename Real> Real f_not ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return inputs_ [ 0 ] == Real { 0 };
}

// Node function wire. simply acts as a wire returning the first input
template<typename Real> Real f_wire ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return inputs_ [ 0 ];
}

// Node function sigmoid. returns the sigmoid of the sum of weighted inputs_.
//    The specific sigmoid function used in the logistic function.
//    range: [0,1]
template<typename Real> Real f_sigmoid ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    Real weightedInputSum;
    Real out;
    weightedInputSum = sumWeigtedInputs ( inputs_, connectionWeights_ );
    out = 1 / ( 1 + exp ( -weightedInputSum ) );
    return out;
}

// Node function Gaussian. returns the Gaussian of the sum of weighted inputs_.
//    range: [0,1]
template<typename Real> Real f_gaussian ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    Real weightedInputSum;
    Real out;
    int centre = 0;
    int width = 1;
    weightedInputSum = sumWeigtedInputs ( inputs_, connectionWeights_ );
    out = exp ( -( pow ( weightedInputSum - centre, 2 ) ) / ( 2 * pow ( width, 2 ) ) );
    return out;
}

// Node function step. returns the step function of the sum of weighted inputs_.
//    range: [0,1]
template<typename Real> Real f_step ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return sumWeigtedInputs ( inputs_, connectionWeights_ ) >= Real { 0 };
}

// Node function step. returns the step function of the sum of weighted inputs_.
//    range: [-1,1]
template<typename Real> Real f_softsign ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    const Real weightedInputSum = sumWeigtedInputs ( inputs_, connectionWeights_ );
    return weightedInputSum / ( Real { 1 } + std::abs ( weightedInputSum ) );
}

// Node function tanh. returns the tanh function of the sum of weighted inputs_.
//    range: [-1,1]
template<typename Real> Real f_hyperbolicTangent ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return tanh ( sumWeigtedInputs ( inputs_, connectionWeights_ ) );
}

} // namespace function
} // namespace cgp

#if defined ( DEF_NL )
#undef nl
#undef DEF_NL
#endif
#undef M64
#undef M32
