# ===----------------------------------------------------------------------===//
#
#                         Numbskull - Factor Graphs
#                       --------------------------------
#                               GML for ALSA
#
# distributed.py
#
# Stephen Bach
# Modified by: Benjamin Chaddha
# 
# This module contains the implementation of a distributed system 
# for running Gibbs sampling on factor graphs
# ===----------------------------------------------------------------------===//\

from __future__ import print_function
import zmq
import sys
import time
import argparse
import gibbs
import numpy as np






def send_array(socket, A, flags=0, copy=True, track=False):
    """Send a numpy array with metadata over a socket."""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def recv_array(socket, flags=0, copy=True, track=False):
    """Receive a numpy array over a socket."""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)

    try:
        A = np.frombuffer(buf, dtype=md['dtype'])
    except:
        A = np.frombuffer(buf, dtype=eval(md['dtype']))

    return A.reshape(md['shape'])


def server(argv=None):
    """Run the server for the distributed Gibbs sampling system."""
    parser = argparse.ArgumentParser(
        description="Run Gibbs worker",
        epilog="")

    parser.add_argument("directory",
                        metavar="DIRECTORY",
                        nargs="?",
                        help="specify directory of factor graph files",
                        default="",
                        type=str)
    parser.add_argument("-p", "--port",
                        metavar="PORT",
                        help="port",
                        default=5556,
                        type=int)
    parser.add_argument("-m", "--meta",
                        metavar="META_FILE",
                        dest="meta",
                        default="graph.meta",
                        type=str,
                        help="meta file")
    parser.add_argument("-w", "--weight",
                        metavar="WEIGHTS_FILE",
                        dest="weight",
                        default="graph.weights",
                        type=str,
                        help="weight file")
    parser.add_argument("-v", "--variable",
                        metavar="VARIABLES_FILE",
                        dest="variable",
                        default="graph.variables",
                        type=str,
                        help="variable file")
    parser.add_argument("-f", "--factor",
                        metavar="FACTORS_FILE",
                        dest="factor",
                        default="graph.factors",
                        type=str,
                        help="factor file")
    parser.add_argument("-b", "--burn",
                        metavar="NUM_BURN_STEPS",
                        dest="burn",
                        default=0,
                        type=int,
                        help="number of burn-in sweeps")
    parser.add_argument("-l", "--learn",
                        metavar="NUM_LEARN_STEPS",
                        dest="learn",
                        default=0,
                        type=int,
                        help="number of learning sweeps")
    parser.add_argument("-e", "--epoch",
                        metavar="NUM_LEARNING_EPOCHS",
                        dest="epoch",
                        default=0,
                        type=int,
                        help="number of learning epochs")
    parser.add_argument("-i", "--inference",
                        metavar="NUM_INFERENCE_STEPS",
                        dest="inference",
                        default=0,
                        type=int,
                        help="number of inference sweeps")
    parser.add_argument("-q", "--quiet",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        help="quiet")
    parser.add_argument("--verbose",
                        dest="verbose",
                        default=False,
                        action="store_true",
                        help="verbose")

    print("Running server...")

    arg = parser.parse_args(argv[1:])

    if arg.directory == "":
        fg = None
    else:
        var_copies = 1
        weight_copies = 1
        (meta, weight, variable, factor,
         fstart, fmap, vstart, vmap, equalPredicate) = \
            gibbs.load(arg.directory, arg.meta, arg.weight, arg.variable,
                       arg.factor, not arg.quiet, not arg.verbose)
        fg_args = (weight, variable, factor, fstart, fmap, vstart,
                   vmap, equalPredicate, var_copies, weight_copies)
        fg = gibbs.FactorGraph(*fg_args)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % arg.port)

    num_clients = 0

    while True:
        # Wait for next request from client
        message = socket.recv()
        if message == "HELLO":  # Initial message from client
            print("Received HELLO.")
            socket.send("CLIENT_ID", zmq.SNDMORE)
            socket.send_json("%d" % num_clients)
            num_clients += 1
        elif message == 'R_FACTOR_GRAPH':  # Request for factor graph
            client_id = socket.recv_json()
            print("Received factor graph request from client #%d." % client_id)
            socket.send("FACTOR_GRAPH", zmq.SNDMORE)
            socket.send_json(len(fg_args), zmq.SNDMORE)
            for a in fg_args:
                is_array = (type(a) == np.ndarray)
                socket.send_json(is_array, zmq.SNDMORE)
                if is_array:
                    send_array(socket, a, zmq.SNDMORE)
                else:
                    socket.send_json(a, zmq.SNDMORE)
            socket.send("DONE")
        elif message == "READY":  # Client ready
            print("Received ready.")
            socket.send("BURN", zmq.SNDMORE)
            socket.send_json(arg.burn)
        elif message == 'DONE_BURN' or message == 'DONE_LEARN':
            if message == 'DONE_BURN':  # Done burning
                epochs = 0
            else:  # Done learning
                epochs = socket.recv_json()
                fg.wv += recv_array(socket)

            if epochs < arg.epoch:
                socket.send("LEARN", zmq.SNDMORE)
                socket.send_json(arg.learn, zmq.SNDMORE)
                socket.send_json(0.001, zmq.SNDMORE)
                send_array(socket, fg.wv)
            else:
                socket.send("INFERENCE", zmq.SNDMORE)
                socket.send_json(arg.inference, zmq.SNDMORE)
                send_array(socket, fg.wv)
        elif message == 'DONE_INFERENCE':  # Client done with inference
            data = recv_array(socket)
            socket.send("EXIT")
        else:
            print("Message (%s) cannot be interpreted." % message,
                  file=sys.stderr)
            socket.send("EXIT")

    return


def client(argv=None):
    """Run the client for the distributed Gibbs sampling system."""
    parser = argparse.ArgumentParser(
        description="Run Gibbs worker",
        epilog="")

    parser.add_argument("directory",
                        metavar="DIRECTORY",
                        nargs="?",
                        help="specify directory of factor graph files",
                        default="",
                        type=str)
    parser.add_argument("-p", "--port",
                        metavar="PORT",
                        help="port",
                        default=5556,
                        type=int)
    parser.add_argument("-m", "--meta",
                        metavar="META_FILE",
                        dest="meta",
                        default="graph.meta",
                        type=str,
                        help="meta file")
    parser.add_argument("-w", "--weight",
                        metavar="WEIGHTS_FILE",
                        dest="weight",
                        default="graph.weights",
                        type=str,
                        help="weight file")
    parser.add_argument("-v", "--variable",
                        metavar="VARIABLES_FILE",
                        dest="variable",
                        default="graph.variables",
                        type=str,
                        help="variable file")
    parser.add_argument("-f", "--factor",
                        metavar="FACTORS_FILE",
                        dest="factor",
                        default="graph.factors",
                        type=str,
                        help="factor file")
    parser.add_argument("-q", "--quiet",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        help="quiet")
    parser.add_argument("--verbose",
                        dest="verbose",
                        default=False,
                        action="store_true",
                        help="verbose")

    print(argv)
    arg = parser.parse_args(argv[1:])

    print("Running client...")
    print(arg.directory)

    if arg.directory == "":
        fg = None
    else:
        var_copies = 1
        weight_copies = 1
        (meta, weight, variable, factor,
         fstart, fmap, vstart, vmap, equalPredicate) = \
            gibbs.load(arg.directory, arg.meta, arg.weight, arg.variable,
                       arg.factor, not arg.quiet, not arg.verbose)
        fg = gibbs.FactorGraph(weight, variable, factor, fstart, fmap, vstart,
                               vmap, equalPredicate, var_copies, weight_copies)

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%s" % arg.port)

    # hello message
    print("Sent HELLO.")
    socket.send("HELLO")
    message = socket.recv()
    assert(message == "CLIENT_ID")
    message = socket.recv_json()
    client_id = int(message)
    print("Received id #%d.\n" % client_id)

    # request factor graph if not loaded
    if fg is None:
        socket.send("R_FACTOR_GRAPH", zmq.SNDMORE)
        socket.send_json(client_id)

        message = socket.recv()
        assert(message == "FACTOR_GRAPH")

        length = socket.recv_json()
        fg_args = [None, ] * length
        for i in range(length):
            is_array = socket.recv_json()
            if is_array:
                fg_args[i] = recv_array(socket)
            else:
                fg_args[i] = socket.recv_json()
        assert(socket.recv() == "DONE")

        fg = gibbs.FactorGraph(*fg_args)

    # Send "ready"
    socket.send("READY")

    learning_epochs = 0
    while True:
        message = socket.recv()
        if message == 'BURN':  # request for burn-in
            print("Received request for burn-in.")
            burn = socket.recv_json()
            print("Burning", burn, "sweeps.")
            fg.gibbs(burn, 0, 0)
            socket.send("DONE_BURN")
        elif message == 'LEARN':  # Request for learning
            print("Received request for learning.")
            sweeps = socket.recv_json()
            step = socket.recv_json()

            fg.wv = recv_array(socket)
            wv = fg.wv

            fg.learn(sweeps, step, 0, 0)

            dw = fg.wv - wv
            socket.send("DONE_LEARNING", zmq.SNDMORE)
            learning_epochs += 1
            socket.send_json(learning_epochs, zmq.SNDMORE)
            send_array(socket, dw)
        elif message == 'INFERENCE':  # Request for inference
            print("Received request for inference.")
            inference = socket.recv_json()
            fg.wv = recv_array(socket)
            print("Inference:", inference, "sweeps.")
            fg.clear()
            fg.gibbs(inference, 0, 0)
            socket.send("DONE_INFERENCE", zmq.SNDMORE)
            send_array(socket, fg.count)
        elif message == 'EXIT':  # Exit
            print("Exit")
            break
        else:
            print("Message cannot be interpreted.", file=sys.stderr)
            break


def main(argv=None):
    """Main entry point for the distributed Gibbs sampling system."""
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print("Usage: ./distributed.py [server/client]", file=sys.stderr)
    elif argv[0].lower() == "server" or argv[0].lower() == "s":
        server(argv)
    elif argv[0].lower() == "client" or argv[0].lower() == "c":
        client(argv)
    else:
        print("Error:", argv[0], "is not a valid choice.", file=sys.stderr)

if __name__ == "__main__":
    main()

