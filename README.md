# Quantum Computing: k-Coloring Problem with Less-than and Greater-than Oracle

## Overview

This repository explores solving the **k-coloring problem** in quantum computing using Grover's algorithm and **less-than + greater-than oracle**. The k-coloring problem is a well-known graph theory problem that aims to assign colors to vertices of a graph such that no two adjacent vertices share the same color, and the total number of colors used does not exceed k.

Quantum computing offers a novel approach to solving this problem by leveraging Grover's algorithm for amplitude amplification combined with a less-than oracle for efficient state marking.

The repository contains code implemented primarily in **Jupyter Notebooks** and **Python**, enabling users to simulate and analyze quantum algorithms for solving the k-coloring problem.


## Features

### Key Components
- **Grover's Algorithm**: A quantum search algorithm used for amplitude amplification, allowing us to amplify the probability of valid solutions to the k-coloring problem.
- **Less-Than/Greater-than Oracle**: A quantum oracle designed to mark states where a certain condition is fulfilled (e.g., states corresponding to vertex-color combinations that satisfy the coloring constraints).
- **Quantum Circuit Design**: Implementation of modular quantum circuits, including oracles and Grover iterations.
- **State Analysis**: Tools to analyze and visualize the amplitude distribution after applying the oracle.


## Authors and Acknowledgments

The majority of the functions and quantum circuit designs in this repository are based on the work of **Oscar-Belletti** and **JSRivero**, whose contributions to quantum computing research have significantly shaped this project.

Their code served as a foundation for implementing the less-than oracle and other related components, adapted and extended for the k-coloring problem in this repository.

We sincerely thank Oscar-Belletti and JSRivero for their innovative contributions to quantum computing, which have been instrumental to this project.


## Repository Structure

### Language Composition
- **Jupyter Notebook (90.6%)**: The main implementation and simulation environment.
- **Python (9.4%)**: Supporting functions and modules for quantum circuit design.

### Files and Directories
- **`notebooks/`**: Contains Jupyter Notebooks for solving and simulating the k-coloring problem.
- **`functions/`**: Python files implementing reusable quantum functions (e.g., Grover's algorithm, less-than oracle).
- **`README.md`**: Documentation for the repository.


## How to Use

### Prerequisites
- **Python 3.8+**
- **Qiskit**: A quantum computing library for designing and simulating quantum circuits.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/larcangeli/Quantum-computing-k-coloring-problem-with-a-less-than-oracle.git
   cd Quantum-computing-k-coloring-problem-with-a-less-than-oracle
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks
Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/
```
Open the notebook files to simulate and analyze the quantum k-coloring problem.


## Quantum Concepts Used

### Grover's Algorithm
Grover's algorithm is leveraged to amplify the probability of valid solutions to the k-coloring problem. Specifically, it uses:
- **Amplitude amplification**: Boosting the amplitudes of states that satisfy the coloring constraints.
- **Iterative searches**: Running multiple iterations of Grover's algorithm to refine the marked solutions.

### Oracle
The oracle is used to mark states that fulfill specific conditions. It uses modular quantum circuit design with:
- **Controlled operations**: Multi-controlled gates to enforce conditions.
- **Bit manipulation**: X gates to temporarily scramble states for conditional checks.


## Future Work
- **Greater-Than Oracle**: Extend the implementation to include a greater-than oracle for different types of graph constraints.
- **Optimization**: Reduce circuit depth and gate count for scalability to larger graphs.
- **Experimental Validation**: Test on real quantum hardware to evaluate performance and fidelity.

## References
- **Oscar-Belletti** and **JSRivero**: Original authors of the foundational quantum functions used in this repository.
- **Qiskit Documentation**: [Qiskit](https://qiskit.org/documentation/) for detailed tutorials and guides on quantum programming.


## License
This project is open-source and available under the MIT License. Feel free to use, modify, and share the code!
