#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: ISOMAP.PY
Date: Monday, November 23 2009
Description: Implements ISOMAP
"""

import os, sys, getopt, pdb, string
import matplotlib as mpl
mpl.use('gtkagg')

import numpy as np
import fpconst
from mds import mds
import pylab

def list_components(adj):
    # Now rows may be duplicate, create g x n matrix where g is number
    # of unique groups and n is number of sensors
    groups = {}

    for row in adj:
        groups[str(row)] = row

    return [np.nonzero(row)[0] for row in groups.values()]

def flatidx(idx):
    adds = np.arange(0, len(idx) ** 2, len(idx))
    return np.ravel(np.transpose(idx) + adds)


def slice_matrix(m,i,j):
    """
    i is an array of row indices
    j is an array of column indices
    m is a matrix where we want to slice out rows i and cols j
    """
    return np.take(np.take(m,i,0),j,1)


def cluster_graph(d, fnc = 'k', size = 7, graph = 'adjacency'):
    """
    An essential part of isomap is the construction of a graph. We
    can use this graph for other kinds of structural analysis.
    """

    # TODO: do sparse clustering

    # the put operation is destructive to d
    ld = d.copy()

    if fnc == 'k':
        if not type(size) == int:
            raise TypeError, "Size must be an integer"
        else:
            # sort to find initial k clusters
            idx = np.argsort(ld)

            # transform row indices for square d to row indices for flat d
            np.put(ld, flatidx(idx[:,size:]), fpconst.PosInf)

    elif fnc == 'epsilon':
        if not type(size) == float:
            raise TypeError, "Size must be a float"
        else:
            ld = np.where(np.less(ld, size), ld, fpconst.PosInf)
    else:
        raise ValueError, "Unknown fnc type"

    # ensure that the result is symmetric
    ld = np.minimum(ld, np.transpose(ld))

    if graph == 'adjacency':
        return np.less(ld, fpconst.PosInf) # boolean matrix. use astype(int) for 0-1 matrix.
    elif graph == 'distance':
        """ Return the shortest paths. """
        return shortest_paths(ld)
    else:
        raise ValueError, "Unknown graph type!"

def shortest_paths(adj, alg = "Floyd1"):
    (n,m) = adj.shape
    assert n == m
    if alg == "Floyd1":
        for k in range(n):
            adj = np.minimum( adj, np.add.outer(adj[:,k],adj[k,:]) )
        return adj
    elif alg == "Floyd2": # variant taken from original Isomap implementation
        for k in range(n):
            adj = np.minimum(adj, np.tile(adj[:,k].reshape(-1,1),(1,n)) + np.tile(adj[k,:],(n,1)))
        return adj
    else:
        raise Error, "Not implemented"

def isomap(d, fnc = 'k', size = 7, dimensions = 2):
    """ Compute isomap instead of mds. Currently neighborhoods of type k are supported. """
    if not len(d.shape) == 2:
        raise ValueError, "d must be a square matrix"

    # the put operation is destructive to d
    ld = cluster_graph(d, fnc = fnc, size = size, graph = 'distance')
    adj = cluster_graph(d, fnc = fnc, size = size, graph = 'adjacency')

    # shortest paths will find connected components
    tmp = np.less(ld, fpconst.PosInf) # 0-1
    groups = list_components(tmp)

    # now do classical mds on largest connected component
    groups.sort(lambda x,y: cmp(len(x),len(y)), reverse=True)
    dg = slice_matrix(ld, groups[0], groups[0])

    Y,eigs = mds(dg, dimensions)

    return (Y, eigs, adj)

def norm(vec):
    return np.sqrt(np.sum(vec**2))

def square_points(size):
    nsensors = size ** 2
    return np.array([(i / size, i % size) for i in range(nsensors)])

def test():

    points = square_points(10)

    distance = np.zeros((100,100))
    for (i, pointi) in enumerate(points):
        for (j, pointj) in enumerate(points):
            distance[i,j] = norm(pointi - pointj)

    Y, eigs, adj = isomap(distance)

    pylab.figure(1)
    pylab.plot(Y[:,0],Y[:,1],'.')

    pylab.figure(2)
    pylab.plot(points[:,0], points[:,1], '.')

    pylab.show()

def main():

    def usage():
	print sys.argv[0] + "[-h] [-d]"

    try:
        (options, args) = getopt.getopt(sys.argv[1:], 'dh', ['help','debug'])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)

    for o, a in options:
        if o in ('-h', '--help'):
            usage()
            sys.exit()
	elif o in ('-d', '--debug'):
	    pdb.set_trace()

    test()

if __name__ == "__main__":
    main()
