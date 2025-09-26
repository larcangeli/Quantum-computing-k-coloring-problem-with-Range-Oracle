from math import sqrt, pi, ceil, log2, floor
from statistics import fmean

import configparser

import networkx as nx
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator

from qiskit.qasm3 import dump, dumps

import pickle
from alive_progress import alive_bar
from time import strftime

from problem import decompose, diffusion, make_problem
from optimization import deduplicate

from simple import simple_system
from minimal import minimal_system
from balanced import balanced_system
from original import original_system

from histogram import plot_simulation_data


def make_circuit(graph, color_number, method, data, grover_iterations=-1):
    qubit_count, init, compose = method

    n = graph.order()
    k = color_number
    qn = ceil(log2(k))  # qubits per node

    if grover_iterations == -1:
        all_colorings = 2 ** (qn * len(conf["graph"]))
        correct_colors = len(cpu_color_graph(conf["graph"], conf["k"]))

        grover_iterations = floor(pi/4 * sqrt(all_colorings / correct_colors))
    data["grover_iterations"] = grover_iterations
    print("Grover iterations:", grover_iterations)
    # inv_col = 2 ** qn - k

    def node_qubits(i):
        return list(range(qn*i, qn*(i+1)))

    problem = make_problem(graph, k)

    num_qubits = qubit_count(problem)

    qc = QuantumCircuit(QuantumRegister(num_qubits),
                        ClassicalRegister(n * qn, name="creg"))

    for i in range(qn*n):
        qc.h(i)

    qc.x(num_qubits - 1)
    qc.h(num_qubits - 1)

    init(qc, problem)

    for i_grv in range(grover_iterations):
        problem = make_problem(graph, k)  # new history
        qv = compose(qc, problem)
        qc.mcx(qv, num_qubits - 1)
        decompose(problem)
        diffusion(qc, problem)

    for i in range(qn*n):
        qc.measure(i, i)

    return qc


def cpu_color_graph(graph, k, node=0, coloring=[]):
    if node == graph.order():
        return [coloring]

    def admissible_colors():
        return set(range(k)) - set(
            [coloring[i] for i in graph.adj[node] if i < len(coloring)])

    valids = []
    for color in admissible_colors():
        valids += cpu_color_graph(graph, k, node+1, coloring + [color])
    return valids


def configure(args_kw, data):
    config_f = configparser.ConfigParser()
    config_f.read("config.ini")

    def parse(v):
        if v.lower() in ["yes", "true", "on"]:
            return True
        if v.lower() in ["no", "false", "off"]:
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    conf = {}
    conf.update([(key, parse(value))
                 for key, value in config_f["graph"].items()])
    conf.update([(key, parse(value))
                 for key, value in config_f["options"].items()])
    conf.update([(key, value)
                 for key, value in args_kw.items()
                 if value is not None])

    if "figsize" not in conf:
        conf["figsize"] = (conf["histogram_size_w"], conf["histogram_size_h"])
    if "graph" not in conf:
        if conf["generate"] == "complete":
            conf["graph"] = nx.complete_graph(conf["k"])
            print("Generating complete graph of", conf["k"], "colors")
        elif conf["generate"] == "random":
            conf["graph"] = gen_graph(
                conf.get("nodes", conf["k"]),
                conf["k"])
            print("Generating random graph with", conf["nodes"],
                  "nodes with chromatic number:", conf["k"])

    systems = {"simple": simple_system,
               "minimal": minimal_system,
               "balanced": balanced_system,
               "original": original_system}

    data["system"] = conf["system"]
    conf["system"] = systems[conf["system"]]
    return conf


def run_circ(qc, conf, data):
    global results
    
    if conf["run"] == "local":  # local sim
        if conf["quantum_sim"]:
            data["noise_model"] = conf["quantum_sim"]
            print("Noise model:", conf["quantum_sim"])
            service = QiskitRuntimeService()
            qbackend = service.get_backend(conf["quantum_sim"])
            noise_model = NoiseModel.from_backend(qbackend)
            backend = AerSimulator(noise_model=noise_model)
        else:
            backend = AerSimulator()
        print("Simulating circuit locally")
        job = backend.run(qc)
        results = job.result()
        data["results"] = results
        return results.get_counts()
    elif conf["run"] == "online" or conf["run"] == "quantum":
        service = QiskitRuntimeService()
        if conf["run"] == "online":
            print("Simulating circuit on online servers")
            backend = service.get_backend(conf["online_sim"])
            shots = conf["online_shots"]
        else:
            qp = conf["quantum_sim"]
            if not qp:
                qpc = service.least_busy(simulator=False,
                                         min_num_qubits=qc.num_qubits)
                qp = qpc.name
            print("Running on quantum computer", qp)
            data["quantum_sim"] = qp
            backend = service.get_backend(qp)
            shots = conf["quantum_shots"]
        pass_manager = generate_preset_pass_manager(optimization_level=3,
                                                    backend=backend)
        transpiled = pass_manager.run(qc)
        data["transpiled"] = transpiled
        sampler = Sampler(backend)
        job = sampler.run([transpiled], shots=shots)
        results = job.result()[0]
        data["results"] = results.data.creg
        return results.data.creg.get_counts()
    else:
        print("Not executing the circuit")
        return None


def plot_graph(graph):
    p = nx.draw(graph)
    plt.show()
    plt.close(p)


def plot_figures(measures, figsize):
    fig = plot_histogram(measures, figsize=figsize)
    fig.tight_layout()
    fig.savefig("measures_new.png")
    # plt.show()
    plt.close(fig)


def plot_circuit(qc):
    p = qc.draw(output="mpl")
    plt.show()
    plt.close(p)


graphs = []


def main(k=None, graph=None, run=None, grover_iterations=None,
         online_sim=None, quantum_sim=None, local_shots=None,
         online_shots=None, quantum_shots=None, print_circuit=None,
         figsize=None, generate=None, nodes=None, print_graph=None,
         system=None, deduplicate_opt=None):
    kwargs = locals().copy()  # keyword arguments as a dict

    global conf, measures
    data = {}
    conf = configure(kwargs, data)

    graphs.append(conf["graph"])

    drawings = []

    if conf["print_graph"]:
        drawings.append(lambda: plot_graph(conf["graph"]))

    data["graph"] = nx.from_edgelist(conf["graph"].edges())
    print(conf["graph"].edges())
    qc = make_circuit(conf["graph"], conf["k"],
                      conf["system"], data,
                      conf["grover_iterations"])

    data["circuit"] = qc
    data["depth"] = qc.depth()
    data["width"] = qc.num_qubits

    print("depth:", qc.depth())
    print("width:", qc.num_qubits)
    # qc.draw(output="mpl")
    if conf["deduplicate_opt"]:
        print("Deduplicating gates...")
        qc = deduplicate(qc)
        data["opt_depth"] = qc.depth()
        print("  After optimization:")
        print("depth:", qc.depth())
        print("width:", qc.num_qubits)
    data["dxw"] = qc.depth() * qc.num_qubits
    print(" -> depth * width:", qc.depth() * qc.num_qubits)
    if print_circuit:
        drawings.append(lambda: plot_circuit(qc))

    measures = run_circ(qc, conf, data)

    if measures is None:
        while drawings:
            drawings.pop()()
        return data

    plot_figures(measures, conf["figsize"])

    n = conf["graph"].order()
    qn = ceil(log2(conf["k"]))

    (all_colorings, correct_colors, measures_of_correct,
     measures_of_incorrect) = interpret_measures(conf, measures)

    data["cpu_sol_num"] = correct_colors
    data["colorings_num"] = all_colorings
    print("Number of solutions(cpu):", correct_colors,
          "/", all_colorings)
    if correct_colors != 0:
        data["opt_grover"] = pi / 4 * sqrt(all_colorings / correct_colors)
        data["random_guess_chance"] = correct_colors / all_colorings
        data["correct_chance"] = (sum(measures_of_correct)
                                  / sum(measures.values()))
        data["avg_prob_corr"] = (fmean(measures_of_correct)
                                 / sum(measures.values()))
        data["avg_prob_inc"] = (fmean(measures_of_incorrect)
                                / sum(measures.values()))
        print("Optimal grover iterations number:",
              pi / 4 * sqrt(all_colorings / correct_colors))
        print("Random guess chance of being correct:",
              correct_colors / all_colorings)
        print("Chance of getting a correct result:",
              sum(measures_of_correct)/sum(measures.values()))
        print("Average \"probability\" of individual correct outcomes:",
              fmean(measures_of_correct)/sum(measures.values()))
        print("Average \"probability\" of individual incorrect outcomes:",
              fmean(measures_of_incorrect)/sum(measures.values()))

    while drawings:
        drawings.pop()()
    return data


def interpret_measures(conf, measures):
    qn = ceil(log2(conf["k"]))
    all_colorings = 2 ** (qn * len(conf["graph"]))
    correct_colors = cpu_color_graph(conf["graph"], conf["k"])

    def coloring(code):  # parse the measured colors into ints
        for i in reversed(list(range(int(len(code)/qn)))):
            yield int("".join(reversed(code[i*qn:(i+1)*qn])), 2)

    measures_of_correct = []
    measures_of_incorrect = []

    for code, count in measures.items():
        if list(coloring(code)) in correct_colors:
            measures_of_correct.append(count)
        else:
            measures_of_incorrect.append(count)

    for i in range(len(correct_colors) - len(measures_of_correct)):
        measures_of_correct.append(0)

    for i in range(2 ** (qn * len(conf["graph"]))
                   - len(correct_colors) - len(measures_of_incorrect)):
        measures_of_incorrect.append(0)

    return (all_colorings, len(correct_colors),
            measures_of_correct, measures_of_incorrect)


# generate graph with specific chromatic number
def gen_graph(nodes, colors, max_tries=100000, p=0.2):
    if nodes == colors:
        return nx.complete_graph(nodes)
    print("p =", p)
    for i in range(max_tries):
        g = nx.gnp_random_graph(nodes, p)
        if (
                nx.is_connected(g)
                and len(cpu_color_graph(g, colors - 1)) == 0
                and len(cpu_color_graph(g, colors)) != 0
        ):
            return g


# generate graph with low number of grover iterations required
def cheat_graph(n, k, amt=20):
    graphs = [gen_graph(n, k) for i in range(amt)]
    return min(graphs,
               key=lambda g: floor(pi/4 * sqrt(2 ** (ceil(log2(k)) * len(g))
                                               / len(cpu_color_graph(g, k)))))


def backends():
    service = QiskitRuntimeService()
    return list(map(lambda x: x.name(), service.backends()))


def simulation(graph, k, iterations=5):
    with open("log" + strftime("%Y_%m_%d__%H_%M") + ".pkl", "wb") as logfile:
        with alive_bar(4 * 4 * iterations) as bar:
            for system in ["simple", "minimal", "balanced", "original"]:
                for noise_model in ['ibm_sherbrooke', 'ibm_brisbane',
                                    'ibm_kyoto', 'ibm_osaka']:
                    for _ in range(iterations):
                        data = main(graph=graph, k=k,
                                    system=system,
                                    quantum_sim=noise_model)
                        bar()
                        pickle.dump(data, logfile)
                        logfile.flush()


class Dummy:
    pass

class UnpicklerIgnoreErrors(pickle._Unpickler):
    def find_class(self, module, name):
        try:
            if (name == "CircuitInstruction" or name == "QuantumCircuit"):
                return Dummy
            return super().find_class(module, name)
        except AttributeError:
            return Dummy
    


def loadlog(filename="log.pkl"):
    objects = []
    with open(filename, "rb") as logfile:
        unpk = UnpicklerIgnoreErrors(logfile)
        while True:
            try:
                objects.append(pickle.load(logfile))
            except EOFError:
                break
    return objects


def format_data(results):
    noise_models = ['ibm_sherbrooke', 'ibm_brisbane',
                    'ibm_kyoto', 'ibm_osaka']
    dic = {x: {} for x in noise_models}
    for system in ["simple", "minimal", "balanced", "original"]:
        for noise_model in noise_models:
            dic[noise_model][system] = []
    for o in results:
        dic[o["noise_model"]][o["system"]].append(o["correct_chance"])
    return dic


def plot_simulation(filename):
    sim = loadlog(filename)
    data = format_data(sim)
    random_guess_chance = sim[0]["random_guess_chance"]
    plot_simulation_data(data, random_guess_chance)


def simulation_details(filename, k):
    sim = loadlog(filename)
    n = len(sim[0]["graph"].nodes)
    chunk = int(len(sim)/4)
    data = {}
    for i in range(4):
        s = sim[i*chunk]
        system = s["system"]
        w = s["width"]
        d = s["depth"]
        anc = s["width"] - ceil(log2(k)) * n - 1
        data[system] = {"width": w,
                        "depth": d,
                        "complexity": w * d,
                        "ancillae": anc}
    return data


def qasm2():
    
    names = {"simple": "minimum_depth",
             "minimal": "minimum_width",
             "balanced": "balanced",
             "original": "original"}

    with alive_bar(4 * 6 * 4) as bar:
        for system in ["simple", "minimal", "balanced", "original"]:
            for n in range(3, 9):
                for k in range(2, 5):
                    if k > n:
                        break
                    print("system:", system, "n:", n, "k:", k)
                    with open("qasm/" + "example_graph_" + names[system]
                              + "_n" + str(n)
                              + "_k" + str(k) + ".qasm", "w") as f:
                        dump(main(generate="random",
                                  nodes=n, k=k, system=system,
                                  run="")["circuit"],
                             f)
                        bar()



def qasm3(log, k):
    graph = log[0]["graph"]
    n = len(graph.nodes())
    for i in [0, 20, 40, 60]:
        system = log[i]["system"]
        print("system:", system)
        with open("qasm/noisy_simulation/" + system
                  + "_n" + str(n) +  "_k" + str(k) + ".qasm", "w") as f:
            c = main(graph=graph, k=k, system=system, run="")["circuit"]
            print(c.qasm(), file=f)
