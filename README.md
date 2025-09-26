# Quantum Computing: k-Coloring Problem with Range Oracle

## Overview

This repository presents an innovative solution to the **k-coloring problem** in quantum computing using Grover's algorithm combined with a novel **Range Oracle** approach. The k-coloring problem is a fundamental graph theory challenge that asks whether a graph can be colored using at most k colors such that no two adjacent vertices share the same color.

**Key Innovation**: This implementation introduces a breakthrough in complexity reduction by using a **greater-than oracle logic** to efficiently mark ancilla qubits for invalid color detection, eliminating the need for multiple individual color checks on each node.

The repository contains code implemented primarily in **Jupyter Notebooks** and **Python**, enabling users to simulate and analyze this advanced quantum algorithm approach.

## Revolutionary Approach: Range Oracle Solution

### The Problem with Traditional Methods
Traditional quantum approaches to the k-coloring problem require multiple conditional checks for each vertex-color combination, leading to:
- High circuit complexity from checking each invalid color individually
- Increased gate count due to repetitive validation operations
- Scalability limitations when dealing with larger color sets

### Our Breakthrough Solution
This repository implements a **Range Oracle** that leverages greater-than oracle logic to:
- **Dramatically reduce complexity** by using a single range comparison per node instead of multiple individual color checks
- **Eliminate redundant validation** through efficient greater-than oracle operations
- **Scale more effectively** to larger graph problems by reducing the number of required ancilla operations
- **Optimize quantum resource utilization** through streamlined circuit design

The key insight is using a greater-than oracle to mark ancilla qubits when `color >= k(problem)`, replacing the traditional approach of checking each invalid color value (k, k+1, k+2, etc.) separately with a single efficient range-based validation.

## Features

### Core Innovations
- **Range Oracle Implementation**: Novel greater-than oracle logic for efficient invalid color detection
- **Optimized Ancilla Marking**: Direct marking of constraint-violating states using mathematical comparisons
- **Grover's Algorithm Integration**: Enhanced amplitude amplification specifically tailored for the range oracle approach
- **Complexity Reduction**: Significant decrease in the number of required oracle operations per node

### Technical Components
- **Greater-Than Oracle**: Advanced quantum oracle using mathematical range comparisons instead of exhaustive enumeration
- **Efficient State Marking**: Streamlined approach to identify and mark invalid color configurations in a single operation
- **Modular Circuit Design**: Clean, reusable quantum circuit components optimized for the range oracle
- **Performance Analysis**: Demonstration of reduced circuit complexity compared to traditional multi-check approaches

## Key Technical Innovation

### Traditional Approach vs. Range Oracle
- **Traditional Method**: For each node with invalid colors {k, k+1, k+2, ..., 2^(ceil(log2(k)))-1}, create separate oracle checks
- **Range Oracle Method**: Single greater-than oracle check per node to detect any color >= k
- **Result**: Reduction from multiple individual checks to one efficient range comparison per node

### Implementation Details
The breakthrough is implemented in the `invalid_color_greater_than` function which:
1. Uses binary representation of the color threshold
2. Applies controlled operations based on bit patterns
3. Marks the ancilla qubit when the node's color value exceeds the valid range
4. Eliminates the need for separate validation of each invalid color value

## Authors and Acknowledgments

The foundational quantum computing functions in this repository build upon the excellent work of **Oscar-Belletti** and **JSRivero**. However, the core innovation—the **Range Oracle approach with greater-than logic for complexity reduction**—represents original research and development.

This breakthrough in using range-based comparisons to mark ancilla qubits instead of performing multiple individual color checks is a novel contribution to quantum graph coloring algorithms.

We acknowledge Oscar-Belletti and JSRivero for their foundational contributions to quantum computing research that enabled this advanced implementation.

## Repository Structure

### Language Composition
- **Jupyter Notebook (73%)**: Main implementation showcasing the Range Oracle approach
- **Python (27%)**: Supporting functions and optimized quantum circuit modules

### Key Files
- **`Oracles.py`**: Core implementation of the greater-than oracle logic
- **`problem.py`**: Problem structure and the new `invalid_color_greater_than` function
- **Implementation Notebooks**: Jupyter notebooks demonstrating the Range Oracle solution
- **Performance Comparisons**: Analysis showing complexity reduction achievements

## How to Use

### Prerequisites
- **Python 3.8+**
- **Qiskit**: Latest version for quantum circuit simulation
- **NumPy**: For mathematical operations and performance analysis

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/larcangeli/Quantum-computing-k-coloring-problem-with-Range-Oracle.git
   cd Quantum-computing-k-coloring-problem-with-Range-Oracle
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Range Oracle Implementation
Launch Jupyter Notebook to explore the breakthrough solution:
```bash
jupyter notebook
```

Open the notebooks to:
- See the Range Oracle implementation in action
- Compare the new approach with traditional multi-check methods
- Analyze the efficiency gains from the greater-than oracle logic

## Technical Innovation Details

### Range Oracle Mechanism
The core innovation uses **greater-than oracle logic** to:
1. **Encode color constraints** as a single mathematical range comparison per node
2. **Mark ancilla qubits** directly when `color >= k(problem)` is detected
3. **Eliminate iterative individual checks** by processing the range condition in one operation
4. **Reduce quantum circuit complexity** through optimized logical operations

### Complexity Improvement
- **Traditional Approach**: Requires separate oracle operations for each invalid color value per node
- **Range Oracle Approach**: Single greater-than oracle operation per node for all invalid colors
- **Result**: Significant reduction in the number of oracle operations and circuit complexity

### Grover's Algorithm Enhancement
The Range Oracle integrates seamlessly with Grover's algorithm to:
- Amplify valid k-coloring solutions more efficiently
- Reduce the total number of oracle operations required
- Improve convergence to optimal solutions through streamlined invalid state detection

## Performance Benefits

### Quantifiable Improvements
- **Oracle Operation Reduction**: Eliminates multiple individual color checks per node
- **Ancilla Efficiency**: Streamlined ancilla qubit utilization through single range comparisons
- **Scalability**: Better performance on larger graphs and higher color counts
- **Resource Optimization**: More efficient use of quantum computing resources

## Future Enhancements

- **Hardware Validation**: Testing the Range Oracle approach on real quantum devices
- **Extended Range Operations**: Implementing additional range-based quantum operations for other constraints
- **Multi-Constraint Optimization**: Extending the approach to other graph optimization problems
- **Hybrid Classical-Quantum**: Integrating classical preprocessing for even greater efficiency

## References

- **Foundational Work**: Oscar-Belletti and JSRivero for base quantum computing functions
- **Qiskit Framework**: [Qiskit Documentation](https://qiskit.org/documentation/) for quantum programming tools
- **Graph Coloring Theory**: Classical approaches that inspired this quantum innovation

## License

This project is open-source and available under the MIT License. The Range Oracle innovation is freely available for research and educational purposes.
