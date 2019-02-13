
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

// https://stackoverflow.com/questions/18728257/back-emplacer-implementation-default-operator-vs-universal-reference-version

template<typename Container>
class back_emplace_iterator : public std::iterator< std::output_iterator_tag, void, void, void, void> {

    protected:

    Container * container;

    public:

    using container_type = Container ;

    explicit back_emplace_iterator ( Container & x ) noexcept : container ( & x ) { }

    template<typename T>
    using _not_self = std::enable_if_t<not(std::is_same_v<std::decay_t<T>, back_emplace_iterator>)>;

    template<typename T, typename = _not_self<T>>
    [[ maybe_unused ]] back_emplace_iterator<Container> & operator = ( T && t ) {
        container->emplace_back ( std::forward<T> ( t ) );
        return * this;
    }

    [[ maybe_unused ]] back_emplace_iterator & operator * ( ) { return * this; }
    [[ maybe_unused ]] back_emplace_iterator & operator ++ ( ) { return * this; }
    [[ maybe_unused ]] back_emplace_iterator & operator ++ ( int ) { return *this; }
};

template<typename Container>
[[ nodiscard ]] back_emplace_iterator<Container> back_emplacer ( Container & c ) {
    return back_emplace_iterator<Container> ( c );
}

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
    std::vector<std::vector<Real>> inputData;
    std::vector<std::vector<Real>> outputData;
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
        if ( not ( n_ ) )
            return 0;
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

    std::vector<int> inputs;
    std::vector<Real> weights;
    int function;
    bool active;
    Real output;
    int maxArity;
    int actArity;

    Node ( ) = delete;
    Node ( const Node & ) = default;
    Node ( Node && ) noexcept = default;
    Node ( const Parameters<Real> & params_, const int nodePosition_ ) :

        function { params_.getRandomFunction ( ) },
        active { true },
        output { Real { 0 } },
        maxArity { params_.arity },
        actArity { params_.arity } {

        inputs.reserve ( maxArity );
        std::generate_n ( back_emplacer ( inputs ), maxArity, [ & params_, nodePosition_ ] ( ) { return params_.getRandomNodeInput ( nodePosition_ ); } );
        weights.reserve ( maxArity );
        std::generate_n ( back_emplacer ( weights ), maxArity, [ & params_ ] ( ) { return params_.getRandomConnectionWeight ( ); } );
    }

    [[ maybe_unused ]] Node & operator = ( const Node & ) = default;
    [[ maybe_unused ]] Node & operator = ( Node && ) noexcept = default;

    [[ nodiscard ]] bool operator == ( const Node & rhs_ ) const noexcept {
        function == rhs_.function;
        inputs == rhs_.inputs;
        // weights == rhs_.weights; // for ANN
    }
    [[ nodiscard ]] bool operator != ( const Node & rhs_ ) const noexcept {
        return not ( operator == ( rhs_ ) );
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

    Chromosome ( ) = delete;
    Chromosome ( const Chromosome & ) = default;
    Chromosome ( Chromosome && ) noexcept = default;
    Chromosome ( const Parameters<Real> & params_ ) :

        numInputs { params_.numInputs },
        numOutputs { params_.numOutputs },
        numNodes { params_.numNodes },
        numActiveNodes { numNodes },
        arity { params_.arity },
        generation { 0 },
        activeNodes { numActiveNodes },
        outputValues { numOutputs },
        nodeInputsHold { arity },
        params { params_ },
        fitness { Real { -1 } } {

        nodes.reserve ( numNodes );
        std::generate_n ( back_emplacer ( nodes ), numNodes, [ & params_, this ] ( ) {
            return Node<Real> { params_, nodes.size ( ) };
        } );
        outputNodes.reserve ( numOutputs );
        std::generate_n ( back_emplacer ( outputNodes ), numOutputs, [ & params_ ] ( ) { return params_.getRandomChromosomeOutput ( ); } );

        setChromosomeActiveNodes ( );
    }

    [[ maybe_unused ]] Chromosome & operator = ( const Chromosome & ) = default;
    [[ maybe_unused ]] Chromosome & operator = ( Chromosome && ) noexcept = default;

    [[ nodiscard ]] bool operator == ( const Chromosome & rhs_ ) const noexcept {
        return
            numInputs == rhs_.numInputs and
            numOutputs == rhs_.numOutputs and
            numNodes == rhs_.numNodes and
            arity == rhs_.arity and
            nodes == rhs_.nodes and
            outputNodes == rhs_.outputNodes;
    }
    [[ nodiscard ]] bool operator != ( const Chromosome & rhs_ ) const noexcept {
        return not ( operator == ( rhs_ ) );
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
            nodes [ currentActiveNode ].output = params.funcSet.function [ currentActiveNodeFunction ] ( nodeInputsHold, nodes [ currentActiveNode ].weights );
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
        nodes [ nodeIndex_ ].actArity = getChromosomeNodeArity ( nodes [ nodeIndex_ ] );
        // Recursively log all the nodes to which the current nodes connect as active.
        for ( int i = 0; i < nodes [ nodeIndex_ ].actArity; ++i )
            recursivelySetActiveNodes ( nodes [ nodeIndex_ ].inputs [ i ] );
    }

    // Gets the chromosome node arity.
    [[ nodiscard ]] int getChromosomeNodeArity ( const Node<Real> & node_ ) {
        const int functionArity = params.funcSet.maxNumInputs [ node_.function ];
        return functionArity == -1 or arity < functionArity ? arity : functionArity;
    }

    void print ( const bool weights_ ) noexcept {
        // Set the active nodes in the given chromosome.
        setChromosomeActiveNodes ( );
        // For all the chromo inputs.
        int i = 0;
        for ( ; i < numInputs; ++i ) {
            std::printf ( "(%d):\tinput\n", i );
        }
        // For all the hidden nodes.
        for ( auto & node : nodes ) {
            // Print the node function.
            std::printf ( "(%d):\t%s\t", i, params.funcSet.functionNames [ node.function ] );
            // For the arity of the node.
            for ( int j = 0; j < getChromosomeNodeArity ( node ); ++j ) {
                // Print the node input information.
                if ( weights_ )
                    std::printf ( "%d,%+.1f\t", node.inputs [ j ], node.weights [ j ] );
                else
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
        chromo_.execute ( data_.inputData [ i ] );
        // for each chromosome output
        for ( int j = 0; j < chromo_.numOutputs; ++j )
            error += std::abs ( chromo_.outputValues [ j ] - data_.outputData [ i ][ j ] );
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
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 0 }, std::plus<Real> ( ) );
}

// Node function sub. Returns the first input minus all remaining inputs_.
template<typename Real> Real f_sub ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::accumulate ( std::next ( std::begin ( inputs_ ) ), std::end ( inputs_ ), inputs_ [ 0 ], std::minus<Real> ( ) );
}

// Node function mul. Returns the multiplication of all the inputs_.
template<typename Real> Real f_mul ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::accumulate ( std::begin ( inputs_ ), std::end ( inputs_ ), Real { 1 }, std::multiplies<Real> ( ) );
}

// Node function div. Returns the first input divided by the second input divided by the third input etc
template<typename Real> Real f_divide ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return std::accumulate ( std::next ( std::begin ( inputs_ ) ), std::end ( inputs_ ), inputs_ [ 0 ], std::divides<Real> ( ) );
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
    return inputs_ [ 0 ] * inputs_ [ 0 ];
}

// Node function cub.  Returns the cube of the first input
template<typename Real> Real f_cube ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    return inputs_ [ 0 ] * inputs_ [ 0 ] * inputs_ [ 0 ];
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
    for ( const auto i : inputs_ ) {
        if ( not ( i ) )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function and. logical NAND, returns '0' if all inputs_ are '1'
//    else, '1'
template<typename Real> Real f_nand ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( not ( i ) )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function or. logical OR, returns '0' if all inputs_ are '0'
//    else, '1'
template<typename Real> Real f_or ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( i )
            return Real { 1 };
    }
    return Real { 0 };
}

// Node function nor. logical NOR, returns '1' if all inputs_ are '0'
//    else, '0'
template<typename Real> Real f_nor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    for ( const auto i : inputs_ ) {
        if ( i )
            return Real { 0 };
    }
    return Real { 1 };
}

// Node function xor. logical XOR, returns '1' iff one of the inputs_ is '1'
//    else, '0'. AKA 'one hot'.
template<typename Real> Real f_xor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
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
template<typename Real> Real f_xnor ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
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
    return Real { 1 } / ( Real { 1 } + std::exp ( -sumWeigtedInputs ( inputs_, connectionWeights_ ) ) );
}

// Node function Gaussian. returns the Gaussian of the sum of weighted inputs_.
//    range: [0,1]
template<typename Real> Real f_gaussian ( const std::vector<Real> & inputs_, const std::vector<Real> & connectionWeights_ ) noexcept {
    constexpr int centre = 0, width = 1;
    return std::exp ( -( std::pow ( sumWeigtedInputs ( inputs_, connectionWeights_ ) - centre, Real { 2 } ) ) / ( Real { 2 } * std::pow ( width, 2 ) ) );
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
    return std::tanh ( sumWeigtedInputs ( inputs_, connectionWeights_ ) );
}

} // namespace function
} // namespace cgp

#if defined ( DEF_NL )
#undef nl
#undef DEF_NL
#endif
#undef M64
#undef M32

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
    struct functionSet *funcSet;
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
    struct functionSet *funcSet;
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

    params->numInputs = numInputs;
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

    params->numNodes = numNodes;
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

    params->numOutputs = numOutputs;
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

    params->arity = arity;
}


/*
    Sets the mu value in given parameters to the new given value. If mu value
    is invalid a warning is displayed and the mu value is left unchanged.
*/
DLL_EXPORT void setMu ( struct parameters *params, int mu ) {

    if ( mu > 0 ) {
        params->mu = mu;
    }
    else {
        printf ( "\nWarning: mu value '%d' is invalid. Mu value must have a value of one or greater. Mu value left unchanged as '%d'.\n", mu, params->mu );
    }
}


/*
    Sets the lambda value in given parameters to the new given value.
    If lambda value is invalid a warning is displayed and the lambda value
    is left unchanged.
*/
DLL_EXPORT void setLambda ( struct parameters *params, int lambda ) {

    if ( lambda > 0 ) {
        params->lambda = lambda;
    }
    else {
        printf ( "\nWarning: lambda value '%d' is invalid. Lambda value must have a value of one or greater. Lambda value left unchanged as '%d'.\n", lambda, params->lambda );
    }
}


/*
    Sets the evolutionary strategy given in parameters to '+' or ','.
    If an invalid option is given a warning is displayed and the evolutionary
    strategy is left unchanged.
*/
DLL_EXPORT void setEvolutionaryStrategy ( struct parameters *params, char evolutionaryStrategy ) {

    if ( evolutionaryStrategy == '+' or evolutionaryStrategy == ',' ) {
        params->evolutionaryStrategy = evolutionaryStrategy;
    }
    else {
        printf ( "\nWarning: the evolutionary strategy '%c' is invalid. The evolutionary strategy must be '+' or ','. The evolutionary strategy has been left unchanged as '%c'.\n", evolutionaryStrategy, params->evolutionaryStrategy );
    }
}


/*
    Sets the recurrent connection probability given in parameters. If an invalid
    value is given a warning is displayed and the value is left	unchanged.
*/
DLL_EXPORT void setRecurrentConnectionProbability ( struct parameters *params, double recurrentConnectionProbability ) {

    if ( recurrentConnectionProbability >= 0 and recurrentConnectionProbability <= 1 ) {
        params->recurrentConnectionProbability = recurrentConnectionProbability;
    }
    else {
        printf ( "\nWarning: recurrent connection probability '%f' is invalid. The recurrent connection probability must be in the range [0,1]. The recurrent connection probability has been left unchanged as '%f'.\n", recurrentConnectionProbability, params->recurrentConnectionProbability );
    }
}


/*
    Sets the whether shortcut connections are used. If an invalid
    value is given a warning is displayed and the value is left	unchanged.
*/
DLL_EXPORT void setShortcutConnections ( struct parameters *params, int shortcutConnections ) {

    if ( shortcutConnections == 0 or shortcutConnections == 1 ) {
        params->shortcutConnections = shortcutConnections;
    }
    else {
        printf ( "\nWarning: shortcut connection '%d' is invalid. The shortcut connections takes values 0 or 1. The shortcut connection has been left unchanged as '%d'.\n", shortcutConnections, params->shortcutConnections );
    }
}



/*
    sets the fitness function to the fitnessFunction passed. If the fitnessFunction is NULL
    then the default supervisedLearning fitness function is used.
*/
DLL_EXPORT void setCustomFitnessFunction ( struct parameters *params, double ( *fitnessFunction )( struct parameters *params, struct chromosome *chromo, struct dataSet *data ), char const *fitnessFunctionName ) {

    if ( fitnessFunction == NULL ) {
        params->fitnessFunction = supervisedLearning;
        strncpy ( params->fitnessFunctionName, "supervisedLearning", FITNESSFUNCTIONNAMELENGTH );
    }
    else {
        params->fitnessFunction = fitnessFunction;
        strncpy ( params->fitnessFunctionName, fitnessFunctionName, FITNESSFUNCTIONNAMELENGTH );
    }
}



/*
    sets the selection scheme used to select the parents from the candidate chromosomes. If the selectionScheme is NULL
    then the default selectFittest selection scheme is used.
*/
DLL_EXPORT void setCustomSelectionScheme ( struct parameters *params, void ( *selectionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **candidateChromos, int numParents, int numCandidateChromos ), char const *selectionSchemeName ) {

    if ( selectionScheme == NULL ) {
        params->selectionScheme = selectFittest;
        strncpy ( params->selectionSchemeName, "selectFittest", SELECTIONSCHEMENAMELENGTH );
    }
    else {
        params->selectionScheme = selectionScheme;
        strncpy ( params->selectionSchemeName, selectionSchemeName, SELECTIONSCHEMENAMELENGTH );
    }
}


/*
    sets the reproduction scheme used to select the parents from the candidate chromosomes. If the reproductionScheme is NULL
    then the default mutateRandomParent selection scheme is used.
*/

DLL_EXPORT void setCustomReproductionScheme ( struct parameters *params, void ( *reproductionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **children, int numParents, int numChildren ), char const *reproductionSchemeName ) {

    if ( reproductionScheme == NULL ) {
        params->reproductionScheme = mutateRandomParent;
        strncpy ( params->reproductionSchemeName, "mutateRandomParent", REPRODUCTIONSCHEMENAMELENGTH );
    }
    else {
        params->reproductionScheme = reproductionScheme;
        strncpy ( params->reproductionSchemeName, reproductionSchemeName, REPRODUCTIONSCHEMENAMELENGTH );
    }
}



/*
    sets the mutation type in params
*/
DLL_EXPORT void setMutationType ( struct parameters *params, char const *mutationType ) {

    if ( strncmp ( mutationType, "probabilistic", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params->mutationType = probabilisticMutation;
        strncpy ( params->mutationTypeName, "probabilistic", MUTATIONTYPENAMELENGTH );
    }

    else if ( strncmp ( mutationType, "point", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params->mutationType = pointMutation;
        strncpy ( params->mutationTypeName, "point", MUTATIONTYPENAMELENGTH );
    }

    else if ( strncmp ( mutationType, "pointANN", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params->mutationType = pointMutationANN;
        strncpy ( params->mutationTypeName, "pointANN", MUTATIONTYPENAMELENGTH );
    }

    else if ( strncmp ( mutationType, "onlyActive", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params->mutationType = probabilisticMutationOnlyActive;
        strncpy ( params->mutationTypeName, "onlyActive", MUTATIONTYPENAMELENGTH );
    }

    else if ( strncmp ( mutationType, "single", MUTATIONTYPENAMELENGTH ) == 0 ) {

        params->mutationType = singleMutation;
        strncpy ( params->mutationTypeName, "single", MUTATIONTYPENAMELENGTH );
    }

    else {
        printf ( "\nWarning: mutation type '%s' is invalid. The mutation type must be 'probabilistic' or 'point'. The mutation type has been left unchanged as '%s'.\n", mutationType, params->mutationTypeName );
    }
}


/*
    Sets the update frequency in generations
*/
DLL_EXPORT void setUpdateFrequency ( struct parameters *params, int updateFrequency ) {

    if ( updateFrequency < 0 ) {
        printf ( "Warning: update frequency of %d is invalid. Update frequency must be >= 0. Update frequency is left unchanged as %d.\n", updateFrequency, params->updateFrequency );
    }
    else {
        params->updateFrequency = updateFrequency;
    }
}





/*
    chromosome function definitions
*/


/*
    Reads in saved chromosomes
*/
DLL_EXPORT struct chromosome* initialiseChromosomeFromFile ( char const *file ) {

    int i, j;

    FILE *fp;
    struct chromosome *chromo;
    struct parameters *params;

    char *line, *record;
    char funcName [ FUNCTIONNAMELENGTH ];
    char buffer [ 1024 ];

    int numInputs, numNodes, numOutputs, arity;

    /* open the chromosome file */
    fp = fopen ( file, "r" );

    /* ensure that the file was opened correctly */
    if ( fp == NULL ) {
        printf ( "Warning: cannot open chromosome: '%s'. Chromosome was not open.\n", file );
        return NULL;
    }

    /* get num inputs */
    line = fgets ( buffer, sizeof ( buffer ), fp );
    if ( line == NULL ) {/*error*/ }
    record = strtok ( line, "," );
    record = strtok ( NULL, "," );
    numInputs = atoi ( record );

    /* get num nodes */
    line = fgets ( buffer, sizeof ( buffer ), fp );
    if ( line == NULL ) {/*error*/ }
    record = strtok ( line, "," );
    record = strtok ( NULL, "," );
    numNodes = atoi ( record );

    /* get num outputs */
    line = fgets ( buffer, sizeof ( buffer ), fp );
    if ( line == NULL ) {/*error*/ }
    record = strtok ( line, "," );
    record = strtok ( NULL, "," );
    numOutputs = atoi ( record );

    /* get arity */
    line = fgets ( buffer, sizeof ( buffer ), fp );
    if ( line == NULL ) {/*error*/ }
    record = strtok ( line, "," );
    record = strtok ( NULL, "," );
    arity = atoi ( record );

    /* initialise parameters  */
    params = initialiseParameters ( numInputs, numNodes, numOutputs, arity );

    /* get and set node functions */
    line = fgets ( buffer, sizeof ( buffer ), fp );
    if ( line == NULL ) {/*error*/ }
    record = strtok ( line, ",\n" );
    record = strtok ( NULL, ",\n" );

    /* for each function name */
    while ( record != NULL ) {

        strncpy ( funcName, record, FUNCTIONNAMELENGTH );

        /* can only load functions defined within CGP-Library */
        if ( addPresetFunctionToFunctionSet ( params, funcName ) == 0 ) {
            printf ( "Error: cannot load chromosome which contains custom node functions.\n" );
            printf ( "Terminating CGP-Library.\n" );
            freeParameters ( params );
            exit ( 0 );
        }

        record = strtok ( NULL, ",\n" );
    }

    /* initialise a chromosome beased on the parameters accociated with given chromosome */
    chromo = initialiseChromosome ( params );

    /* set the node parameters */
    for ( i = 0; i < numNodes; i++ ) {

        /* get the function gene */
        line = fgets ( buffer, sizeof ( buffer ), fp );
        record = strtok ( line, ",\n" );
        chromo->nodes [ i ]->function = atoi ( record );

        for ( j = 0; j < arity; j++ ) {
            line = fgets ( buffer, sizeof ( buffer ), fp );
            sscanf ( line, "%d,%lf", &chromo->nodes [ i ]->inputs [ j ], &chromo->nodes [ i ]->weights [ j ] );
        }
    }

    /* set the outputs */
    line = fgets ( buffer, sizeof ( buffer ), fp );
    record = strtok ( line, ",\n" );
    chromo->outputNodes [ 0 ] = atoi ( record );

    for ( i = 1; i < numOutputs; i++ ) {
        record = strtok ( NULL, ",\n" );
        chromo->outputNodes [ i ] = atoi ( record );
    }

    fclose ( fp );
    freeParameters ( params );

    /* set the active nodes in the copied chromosome */
    setChromosomeActiveNodes ( chromo );

    return chromo;
}


/*
    Returns a pointer to an initialised chromosome with values obeying the given parameters.
*/
DLL_EXPORT struct chromosome *initialiseChromosomeFromChromosome ( struct chromosome *chromo ) {

    struct chromosome *chromoNew;
    int i;

    /* check that funcSet contains functions*/
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
        chromoNew->nodes [ i ] = initialiseNode ( chromo->numInputs, chromo->numNodes, chromo->arity, chromo->funcSet->numFunctions, 0, 0, i );
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
    chromoNew->funcSet = ( struct functionSet* )std::malloc ( sizeof ( struct functionSet ) );
    copyFunctionSet ( chromoNew->funcSet, chromo->funcSet );

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


/*
    returns whether the specified node is active in the given chromosome
*/
DLL_EXPORT int isNodeActive ( struct chromosome *chromo, int node ) {

    if ( node < 0 or node > chromo->numNodes ) {
        printf ( "Error: node less than or greater than the number of nodes  in chromosome. Called from isNodeActive.\n" );
        exit ( 0 );
    }

    return chromo->nodes [ node ]->active;
}


/*
    Saves the given chromosome in a form which can be read in later
*/
DLL_EXPORT void saveChromosome ( struct chromosome *chromo, char const *fileName ) {

    int i, j;
    FILE *fp;

    /* create the chromosome file */
    fp = fopen ( fileName, "w" );

    /* ensure that the file was created correctly */
    if ( fp == NULL ) {
        printf ( "Warning: cannot save chromosome to '%s'. Chromosome was not saved.\n", fileName );
        return;
    }

    /* save meta information */
    fprintf ( fp, "numInputs,%d\n", chromo->numInputs );
    fprintf ( fp, "numNodes,%d\n", chromo->numNodes );
    fprintf ( fp, "numOutputs,%d\n", chromo->numOutputs );
    fprintf ( fp, "arity,%d\n", chromo->arity );

    fprintf ( fp, "functionSet" );

    for ( i = 0; i < chromo->funcSet->numFunctions; i++ ) {
        fprintf ( fp, ",%s", chromo->funcSet->functionNames [ i ] );
    }
    fprintf ( fp, "\n" );

    /* save the chromosome structure */
    for ( i = 0; i < chromo->numNodes; i++ ) {

        fprintf ( fp, "func %d\n", chromo->nodes [ i ]->function );

        for ( j = 0; j < chromo->arity; j++ ) {
            fprintf ( fp, "input %d, weights %f\n", chromo->nodes [ i ]->inputs [ j ], chromo->nodes [ i ]->weights [ j ] );
        }
    }

    for ( i = 0; i < chromo->numOutputs; i++ ) {
        fprintf ( fp, "output nodes %d,", chromo->outputNodes [ i ] );
    }

    fclose ( fp );
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


DLL_EXPORT int compareChromosomesActiveNodesANN ( struct chromosome *chromoA, struct chromosome *chromoB ) {

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

    params->mutationType ( params, chromo );

    setChromosomeActiveNodes ( chromo );
}


/*
    removes the inactive nodes from the given chromosome
*/
DLL_EXPORT void removeInactiveNodes ( struct chromosome *chromo ) {

    int i, j, k;

    int originalNumNodes = chromo->numNodes;

    /* set the active nodes */
    setChromosomeActiveNodes ( chromo );

    /* for all nodes */
    for ( i = 0; i < chromo->numNodes - 1; i++ ) {

        /* if the node is inactive */
        if ( chromo->nodes [ i ]->active == 0 ) {

            /* set the node to be the next node */
            for ( j = i; j < chromo->numNodes - 1; j++ ) {
                copyNode ( chromo->nodes [ j ], chromo->nodes [ j + 1 ] );
            }

            /* */
            for ( j = 0; j < chromo->numNodes; j++ ) {
                for ( k = 0; k < chromo->arity; k++ ) {

                    if ( chromo->nodes [ j ]->inputs [ k ] >= i + chromo->numInputs ) {
                        chromo->nodes [ j ]->inputs [ k ]--;
                    }
                }
            }

            /* for the number of chromosome outputs */
            for ( j = 0; j < chromo->numOutputs; j++ ) {

                if ( chromo->outputNodes [ j ] >= i + chromo->numInputs ) {
                    chromo->outputNodes [ j ]--;
                }
            }

            /* de-increment the number of nodes */
            chromo->numNodes--;

            /* made the newly assigned node be evaluated */
            i--;
        }
    }

    for ( i = chromo->numNodes; i < originalNumNodes; i++ ) {
        freeNode ( chromo->nodes [ i ] );
    }

    if ( chromo->nodes [ chromo->numNodes - 1 ]->active == 0 ) {
        freeNode ( chromo->nodes [ chromo->numNodes - 1 ] );
        chromo->numNodes--;
    }

    /* reallocate the memory associated with the chromosome */
    chromo->nodes = ( struct node** )realloc ( chromo->nodes, chromo->numNodes * sizeof ( struct node* ) );
    chromo->activeNodes = ( int* ) realloc ( chromo->activeNodes, chromo->numNodes * sizeof ( int ) );

    /* set the active nodes */
    setChromosomeActiveNodes ( chromo );
}


/*
    sets the fitness of the given chromosome
*/
DLL_EXPORT void setChromosomeFitness ( struct parameters *params, struct chromosome *chromo, struct dataSet *data ) {

    double fitness;

    setChromosomeActiveNodes ( chromo );

    resetChromosome ( chromo );

    fitness = params->fitnessFunction ( params, chromo, data );

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
    copies the contents of one chromosome to another. Provided the number of inputs, nodes, outputs and node arity are the same.
*/
DLL_EXPORT void copyChromosome ( struct chromosome *chromoDest, struct chromosome *chromoSrc ) {

    int i;

    /* error checking  */
    if ( chromoDest->numInputs != chromoSrc->numInputs ) {
        printf ( "Error: cannot copy a chromosome to a chromosome of different dimensions. The number of chromosome inputs do not match.\n" );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    if ( chromoDest->numNodes != chromoSrc->numNodes ) {
        printf ( "Error: cannot copy a chromosome to a chromosome of different dimensions. The number of chromosome nodes do not match.\n" );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    if ( chromoDest->numOutputs != chromoSrc->numOutputs ) {
        printf ( "Error: cannot copy a chromosome to a chromosome of different dimensions. The number of chromosome outputs do not match.\n" );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    if ( chromoDest->arity != chromoSrc->arity ) {
        printf ( "Error: cannot copy a chromosome to a chromosome of different dimensions. The arity of the chromosome nodes do not match.\n" );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    /* copy nodes and which are active */
    for ( i = 0; i < chromoSrc->numNodes; i++ ) {
        copyNode ( chromoDest->nodes [ i ], chromoSrc->nodes [ i ] );
        chromoDest->activeNodes [ i ] = chromoSrc->activeNodes [ i ];
    }

    /* copy functionset */
    copyFunctionSet ( chromoDest->funcSet, chromoSrc->funcSet );

    /* copy each of the chromosomes outputs */
    for ( i = 0; i < chromoSrc->numOutputs; i++ ) {
        chromoDest->outputNodes [ i ] = chromoSrc->outputNodes [ i ];
    }

    /* copy the number of active node */
    chromoDest->numActiveNodes = chromoSrc->numActiveNodes;

    /* copy the fitness */
    chromoDest->fitness = chromoSrc->fitness;

    /* copy generation */
    chromoDest->generation = chromoSrc->generation;
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
    int maxArity = chromo->funcSet->maxNumInputs [ chromo->nodes [ index ]->function ];

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
    set the active nodes in the given chromosome
*/
static void setChromosomeActiveNodes ( struct chromosome *chromo ) {

    int i;

    /* error checking */
    if ( chromo == NULL ) {
        printf ( "Error: chromosome has not been initialised and so the active nodes cannot be set.\n" );
        return;
    }

    /* set the number of active nodes to zero */
    chromo->numActiveNodes = 0;

    /* reset the active nodes */
    for ( i = 0; i < chromo->numNodes; i++ ) {
        chromo->nodes [ i ]->active = 0;
    }

    /* start the recursive search for active nodes from the output nodes for the number of output nodes */
    for ( i = 0; i < chromo->numOutputs; i++ ) {

        /* if the output connects to a chromosome input, skip */
        if ( chromo->outputNodes [ i ] < chromo->numInputs ) {
            continue;
        }

        /* begin a recursive search for active nodes */
        recursivelySetActiveNodes ( chromo, chromo->outputNodes [ i ] );
    }

    /* place active nodes in order */
    sortIntArray ( chromo->activeNodes, chromo->numActiveNodes );
}


/*
    used by setActiveNodes to recursively search for active nodes
*/
static void recursivelySetActiveNodes ( struct chromosome *chromo, int nodeIndex ) {

    int i;

    /* if the given node is an input, stop */
    if ( nodeIndex < chromo->numInputs ) {
        return;
    }

    /* if the given node has already been flagged as active */
    if ( chromo->nodes [ nodeIndex - chromo->numInputs ]->active == 1 ) {
        return;
    }

    /* log the node as active */
    chromo->nodes [ nodeIndex - chromo->numInputs ]->active = 1;
    chromo->activeNodes [ chromo->numActiveNodes ] = nodeIndex - chromo->numInputs;
    chromo->numActiveNodes++;

    /* set the nodes actual arity*/
    chromo->nodes [ nodeIndex - chromo->numInputs ]->actArity = getChromosomeNodeArity ( chromo, nodeIndex - chromo->numInputs );

    /* recursively log all the nodes to which the current nodes connect as active */
    for ( i = 0; i < chromo->nodes [ nodeIndex - chromo->numInputs ]->actArity; i++ ) {
        recursivelySetActiveNodes ( chromo, chromo->nodes [ nodeIndex - chromo->numInputs ]->inputs [ i ] );
    }
}


/*
    Sorts the given array of chromosomes by fitness, lowest to highest
    uses insertion sort (quickish and stable)
*/
static void sortChromosomeArray ( struct chromosome **chromoArray, int numChromos ) {

    int i, j;
    struct chromosome *chromoTmp;

    for ( i = 0; i < numChromos; i++ ) {
        for ( j = i; j < numChromos; j++ ) {

            if ( chromoArray [ i ]->fitness > chromoArray [ j ]->fitness ) {
                chromoTmp = chromoArray [ i ];
                chromoArray [ i ] = chromoArray [ j ];
                chromoArray [ j ] = chromoTmp;
            }
        }
    }
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
    Initialises data structure and assigns values of given file
*/
DLL_EXPORT struct dataSet *initialiseDataSetFromFile ( char const *file ) {

    int i;
    struct dataSet *data;
    FILE *fp;
    char *line, *record;
    char buffer [ 1024 ];
    int lineNum = -1;
    int col;

    /* attempt to open the given file */
    fp = fopen ( file, "r" );

    /* if the file cannot be found */
    if ( fp == NULL ) {
        printf ( "Error: file '%s' cannot be found.\nTerminating CGP-Library.\n", file );
        exit ( 0 );
    }

    /* initialise memory for data structure */
    data = ( struct dataSet* )std::malloc ( sizeof ( struct dataSet ) );

    /* for every line in the given file */
    while ( ( line = fgets ( buffer, sizeof ( buffer ), fp ) ) != NULL ) {

        /* deal with the first line containing meta data */
        if ( lineNum == -1 ) {

            sscanf ( line, "%d,%d,%d", &( data->numInputs ), &( data->numOutputs ), &( data->numSamples ) );

            data->inputData = ( double** ) std::malloc ( data->numSamples * sizeof ( double* ) );
            data->outputData = ( double** ) std::malloc ( data->numSamples * sizeof ( double* ) );

            for ( i = 0; i < data->numSamples; i++ ) {
                data->inputData [ i ] = ( double* ) std::malloc ( data->numInputs * sizeof ( double ) );
                data->outputData [ i ] = ( double* ) std::malloc ( data->numOutputs * sizeof ( double ) );
            }
        }
        /* the other lines contain input output pairs */
        else {

            /* get the first value on the given line */
            record = strtok ( line, " ,\n" );
            col = 0;

            /* until end of line */
            while ( record != NULL ) {

                /* if its an input value */
                if ( col < data->numInputs ) {
                    data->inputData [ lineNum ] [ col ] = atof ( record );
                }

                /* if its an output value */
                else {

                    data->outputData [ lineNum ] [ col - data->numInputs ] = atof ( record );
                }

                /* get the next value on the given line */
                record = strtok ( NULL, " ,\n" );

                /* increment the current col index */
                col++;
            }
        }

        /* increment the current line index */
        lineNum++;
    }

    fclose ( fp );

    return data;
}


/*
    frees given dataSet
*/
DLL_EXPORT void freeDataSet ( struct dataSet *data ) {

    int i;

    /* attempt to prevent user double freeing */
    if ( data == NULL ) {
        printf ( "Warning: double freeing of dataSet prevented.\n" );
        return;
    }

    for ( i = 0; i < data->numSamples; i++ ) {
        free ( data->inputData [ i ] );
        free ( data->outputData [ i ] );
    }

    free ( data->inputData );
    free ( data->outputData );
    free ( data );
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
    CGP Functions
*/


/*
    Other Functions
*/








/*
    Mutation Methods
*/



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
    numFunctionGenes = params->numNodes;
    numInputGenes = params->numNodes * params->arity;
    numOutputGenes = params->numOutputs;

    /* set the total number of chromosome genes */
    numGenes = numFunctionGenes + numInputGenes + numOutputGenes;

    /* calculate the number of genes to mutate */
    numGenesToMutate = ( int ) roundf ( numGenes * params->mutationRate );

    /* for the number of genes to mutate */
    for ( i = 0; i < numGenesToMutate; i++ ) {

        /* select a random gene */
        geneToMutate = randInt ( numGenes );

        /* mutate function gene */
        if ( geneToMutate < numFunctionGenes ) {

            nodeIndex = geneToMutate;

            chromo->nodes [ nodeIndex ]->function = getRandomFunction ( chromo->funcSet->numFunctions );
        }

        /* mutate node input gene */
        else if ( geneToMutate < numFunctionGenes + numInputGenes ) {

            nodeIndex = ( int ) ( ( geneToMutate - numFunctionGenes ) / chromo->arity );
            nodeInputIndex = ( geneToMutate - numFunctionGenes ) % chromo->arity;

            chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, nodeIndex, params->recurrentConnectionProbability );
        }

        /* mutate output gene */
        else {
            nodeIndex = geneToMutate - numFunctionGenes - numInputGenes;
            chromo->outputNodes [ nodeIndex ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params->shortcutConnections );
        }
    }
}


/*
    Same as pointMutation but also mutates weight genes. The reason this is separated is
    that point mutation should always mutate the same number of genes. When weight genes are not
    used many mutations will not do anything and so the number of actual mutations varies.
    - needs explaining better...
*/
static void pointMutationANN ( struct parameters *params, struct chromosome *chromo ) {

    int i;
    int numGenes;
    int numFunctionGenes, numInputGenes, numWeightGenes, numOutputGenes;
    int numGenesToMutate;
    int geneToMutate;
    int nodeIndex;
    int nodeInputIndex;

    /* get the number of each type of gene */
    numFunctionGenes = params->numNodes;
    numInputGenes = params->numNodes * params->arity;
    numWeightGenes = params->numNodes * params->arity;
    numOutputGenes = params->numOutputs;

    /* set the total number of chromosome genes */
    numGenes = numFunctionGenes + numInputGenes + numWeightGenes + numOutputGenes;

    /* calculate the number of genes to mutate */
    numGenesToMutate = ( int ) roundf ( numGenes * params->mutationRate );

    /* for the number of genes to mutate */
    for ( i = 0; i < numGenesToMutate; i++ ) {

        /* select a random gene */
        geneToMutate = randInt ( numGenes );

        /* mutate function gene */
        if ( geneToMutate < numFunctionGenes ) {

            nodeIndex = geneToMutate;

            chromo->nodes [ nodeIndex ]->function = getRandomFunction ( chromo->funcSet->numFunctions );
        }

        /* mutate node input gene */
        else if ( geneToMutate < numFunctionGenes + numInputGenes ) {

            nodeIndex = ( int ) ( ( geneToMutate - numFunctionGenes ) / chromo->arity );
            nodeInputIndex = ( geneToMutate - numFunctionGenes ) % chromo->arity;

            chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, nodeIndex, params->recurrentConnectionProbability );
        }

        /* mutate connection weight */
        else if ( geneToMutate < numFunctionGenes + numInputGenes + numWeightGenes ) {

            nodeIndex = ( int ) ( ( geneToMutate - numFunctionGenes - numInputGenes ) / chromo->arity );
            nodeInputIndex = ( geneToMutate - numFunctionGenes - numInputGenes ) % chromo->arity;

            chromo->nodes [ nodeIndex ]->weights [ nodeInputIndex ] = getRandomConnectionWeight ( params->connectionWeightRange );
        }

        /* mutate output gene */
        else {
            nodeIndex = geneToMutate - numFunctionGenes - numInputGenes - numWeightGenes;
            chromo->outputNodes [ nodeIndex ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params->shortcutConnections );
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
    numFunctionGenes = params->numNodes;
    numInputGenes = params->numNodes * params->arity;
    numOutputGenes = params->numOutputs;

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

            chromo->nodes [ nodeIndex ]->function = getRandomFunction ( chromo->funcSet->numFunctions );

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

            chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, nodeIndex, params->recurrentConnectionProbability );

            newGeneValue = chromo->nodes [ nodeIndex ]->inputs [ nodeInputIndex ];

            if ( ( previousGeneValue != newGeneValue ) and ( chromo->nodes [ nodeIndex ]->active == 1 ) ) {
                mutatedActive = 1;
            }
        }

        /* mutate output gene */
        else {
            nodeIndex = geneToMutate - numFunctionGenes - numInputGenes;

            previousGeneValue = chromo->outputNodes [ nodeIndex ];

            chromo->outputNodes [ nodeIndex ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params->shortcutConnections );

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
        if ( randDecimal ( ) <= params->mutationRate ) {
            chromo->nodes [ activeNode ]->function = getRandomFunction ( chromo->funcSet->numFunctions );
        }

        /* for every input to each chromosome */
        for ( j = 0; j < params->arity; j++ ) {

            /* mutate the node input */
            if ( randDecimal ( ) <= params->mutationRate ) {
                chromo->nodes [ activeNode ]->inputs [ j ] = getRandomNodeInput ( chromo->numInputs, chromo->numNodes, activeNode, params->recurrentConnectionProbability );
            }

            /* mutate the node connection weight */
            if ( randDecimal ( ) <= params->mutationRate ) {
                chromo->nodes [ activeNode ]->weights [ j ] = getRandomConnectionWeight ( params->connectionWeightRange );
            }
        }
    }

    /* for every chromosome output */
    for ( i = 0; i < params->numOutputs; i++ ) {

        /* mutate the chromosome output */
        if ( randDecimal ( ) <= params->mutationRate ) {
            chromo->outputNodes [ i ] = getRandomChromosomeOutput ( chromo->numInputs, chromo->numNodes, params->shortcutConnections );
        }
    }
}


/*
    repetitively applies runCGP to obtain average behaviour
*/
DLL_EXPORT struct results* repeatCGP ( struct parameters *params, struct dataSet *data, int numGens, int numRuns ) {

    int i;
    struct results *rels;
    int updateFrequency = params->updateFrequency;

    /* set the update frequency so as to to so generational results */
    params->updateFrequency = 0;

    rels = initialiseResults ( params, numRuns );

    printf ( "Run\tFitness\t\tGenerations\tActive Nodes\n" );

    /* for each run */
#pragma omp parallel for default(none), shared(numRuns,rels,params,data,numGens), schedule(dynamic), num_threads(params->numThreads)
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
    params->updateFrequency = updateFrequency;

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

    if ( data != NULL and params->numInputs != data->numInputs ) {
        printf ( "Error: The number of inputs specified in the dataSet (%d) does not match the number of inputs specified in the parameters (%d).\n", data->numInputs, params->numInputs );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    if ( data != NULL and params->numOutputs != data->numOutputs ) {
        printf ( "Error: The number of outputs specified in the dataSet (%d) does not match the number of outputs specified in the parameters (%d).\n", data->numOutputs, params->numOutputs );
        printf ( "Terminating CGP-Library.\n" );
        exit ( 0 );
    }

    /* initialise parent chromosomes */
    parentChromos = ( struct chromosome** )std::malloc ( params->mu * sizeof ( struct chromosome* ) );

    for ( i = 0; i < params->mu; i++ ) {
        parentChromos [ i ] = initialiseChromosome ( params );
    }

    /* initialise children chromosomes */
    childrenChromos = ( struct chromosome** )std::malloc ( params->lambda * sizeof ( struct chromosome* ) );

    for ( i = 0; i < params->lambda; i++ ) {
        childrenChromos [ i ] = initialiseChromosome ( params );
    }

    /* intilise best chromosome */
    bestChromo = initialiseChromosome ( params );

    /* determine the size of the Candidate Chromos based on the evolutionary Strategy */
    if ( params->evolutionaryStrategy == '+' ) {
        numCandidateChromos = params->mu + params->lambda;
    }
    else if ( params->evolutionaryStrategy == ',' ) {
        numCandidateChromos = params->lambda;
    }
    else {
        printf ( "Error: the evolutionary strategy '%c' is not known.\nTerminating CGP-Library.\n", params->evolutionaryStrategy );
        exit ( 0 );
    }

    /* initialise the candidateChromos */
    candidateChromos = ( struct chromosome** )std::malloc ( numCandidateChromos * sizeof ( struct chromosome* ) );

    for ( i = 0; i < numCandidateChromos; i++ ) {
        candidateChromos [ i ] = initialiseChromosome ( params );
    }

    /* set fitness of the parents */
    for ( i = 0; i < params->mu; i++ ) {
        setChromosomeFitness ( params, parentChromos [ i ], data );
    }

    /* show the user whats going on */
    if ( params->updateFrequency != 0 ) {
        printf ( "\n-- Starting CGP --\n\n" );
        printf ( "Gen\tfitness\n" );
    }

    /* for each generation */
    for ( gen = 0; gen < numGens; gen++ ) {

        /* set fitness of the children of the population */
    #pragma omp parallel for default(none), shared(params, childrenChromos,data), schedule(dynamic), num_threads(params->numThreads)
        for ( i = 0; i < params->lambda; i++ ) {
            setChromosomeFitness ( params, childrenChromos [ i ], data );
        }

        /* get best chromosome */
        getBestChromosome ( parentChromos, childrenChromos, params->mu, params->lambda, bestChromo );

        /* check termination conditions */
        if ( getChromosomeFitness ( bestChromo ) <= params->targetFitness ) {

            if ( params->updateFrequency != 0 ) {
                printf ( "%d\t%f - Solution Found\n", gen, bestChromo->fitness );
            }

            break;
        }

        /* display progress to the user at the update frequency specified */
        if ( params->updateFrequency != 0 and ( gen % params->updateFrequency == 0 or gen >= numGens - 1 ) ) {
            printf ( "%d\t%f\n", gen, bestChromo->fitness );
        }

        /*
            Set the chromosomes which will be used by the selection scheme
            dependant upon the evolutionary strategy. i.e. '+' all are used
            by the selection scheme, ',' only the children are.
        */
        if ( params->evolutionaryStrategy == '+' ) {

            /*
                Note: the children are placed before the parents to
                ensure 'new blood' is always selected over old if the
                fitness are equal.
            */

            for ( i = 0; i < numCandidateChromos; i++ ) {

                if ( i < params->lambda ) {
                    copyChromosome ( candidateChromos [ i ], childrenChromos [ i ] );
                }
                else {
                    copyChromosome ( candidateChromos [ i ], parentChromos [ i - params->lambda ] );
                }
            }
        }
        else if ( params->evolutionaryStrategy == ',' ) {

            for ( i = 0; i < numCandidateChromos; i++ ) {
                copyChromosome ( candidateChromos [ i ], childrenChromos [ i ] );
            }
        }

        /* select the parents from the candidateChromos */
        params->selectionScheme ( params, parentChromos, candidateChromos, params->mu, numCandidateChromos );

        /* create the children from the parents */
        params->reproductionScheme ( params, parentChromos, childrenChromos, params->mu, params->lambda );
    }

    /* deal with formatting for displaying progress */
    if ( params->updateFrequency != 0 ) {
        printf ( "\n" );
    }

    /* copy the best best chromosome */
    bestChromo->generation = gen;
    /*copyChromosome(chromo, bestChromo);*/

    /* free parent chromosomes */
    for ( i = 0; i < params->mu; i++ ) {
        freeChromosome ( parentChromos [ i ] );
    }
    free ( parentChromos );

    /* free children chromosomes */
    for ( i = 0; i < params->lambda; i++ ) {
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


/*
    Selection scheme which selects the fittest members of the population
    to be the parents.

    The candidateChromos contains the current children followed by the
    current parents. This means that using a stable sort to order
    candidateChromos results in children being selected over parents if
    their fitnesses are equal. A desirable property in CGP to facilitate
    neutral genetic drift.
*/
static void selectFittest ( struct parameters *params, struct chromosome **parents, struct chromosome **candidateChromos, int numParents, int numCandidateChromos ) {

    int i;

    sortChromosomeArray ( candidateChromos, numCandidateChromos );

    for ( i = 0; i < numParents; i++ ) {
        copyChromosome ( parents [ i ], candidateChromos [ i ] );
    }
}

#endif