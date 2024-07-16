# ===----------------------------------------------------------------------===//
#
#                         Numbskull - Factor Graphs
#                       --------------------------------
#                               GML for ALSA
#
# learning.py
#
# Stephen Bach
# Modified by: Benjamin Chaddha
# 
# ===----------------------------------------------------------------------===//\

from __future__ import print_function, absolute_import
import numba
from numba import jit
import numpy as np
import math
import random
from numbskull.inference import draw_sample, eval_factor

@jit(nopython=True, cache=True, nogil=True)
# This file contains the implementation of the `learnthread` function, which performs stochastic gradient descent for each variable.

def learnthread(shardID, nshards, step, regularization, reg_param, truncation,
                var_copy, weight_copy, weight,
                variable, factor, fmap,
                vmap, factor_index, Z, fids, var_value, var_value_evid,
                weight_value, learn_non_evidence):
    """Perform stochastic gradient descent for each variable."""
    # Identify start and end variable
    nvar = variable.shape[0]
    start = (shardID * nvar) // nshards
    end = ((shardID + 1) * nvar) // nshards
    for var_samp in range(start, end):  # Sample and perform stochastic gradient descent for each variable
        if variable[var_samp]["isEvidence"] == 4:
            # This variable is not owned by this machine
            continue
        sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                       var_copy, weight_copy, weight, variable,
                       factor, fmap, vmap,
                       factor_index, Z[shardID], fids[shardID], var_value,
                       var_value_evid, weight_value, learn_non_evidence)


@jit(nopython=True, cache=True, nogil=True)
def get_factor_id_range(variable, vmap, var_samp, val):
    """TODO."""
    varval_off = val
    if variable[var_samp]["dataType"] == 0:
        varval_off = 0
    vtf = vmap[variable[var_samp]["vtf_offset"] + varval_off]
    start = vtf["factor_index_offset"]
    end = start + vtf["factor_index_length"]
    return (start, end)


@jit(nopython=True, cache=True, nogil=True)
def sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                   var_copy, weight_copy, weight, variable, factor, fmap,
                   vmap, factor_index, Z, fids, var_value, var_value_evid,
                   weight_value, learn_non_evidence):
    """TODO."""
    # Stochastic gradient descent approximates all samples using a single sample to adjust the parameter theta, which is faster to compute and results in the vicinity of the optimum.
    # The var_samp parameter is an index, and truncation means truncation.
    # If learn_non_evidence, sample twice.
    # The method corresponds to expectation-conjugate descent.
    if variable[var_samp]["isEvidence"] != 1:  # If it is not an observed variable, it needs to be sampled
        evidence = draw_sample(var_samp, var_copy, weight_copy,
                               weight, variable, factor,
                               fmap, vmap, factor_index, Z,
                               var_value_evid, weight_value)
        # If evidence, store the initial value in a tmp variable
    # then sample and compute the gradient.
    else:
        evidence = variable[var_samp]["initialValue"]  # If it is an observed variable, directly take out the initial value

    var_value_evid[var_copy][var_samp] = evidence  # var_value_evid should store the variables that are evidence nodes
    # Sample the variable
    proposal = draw_sample(var_samp, var_copy, weight_copy, weight,
                           variable, factor, fmap, vmap,
                           factor_index, Z, var_value, weight_value)

    var_value[var_copy][var_samp] = proposal
    if not learn_non_evidence and variable[var_samp]["isEvidence"] != 1:
        return
    # Compute the gradient and update the weights
    # Iterate over corresponding factors

    range_fids = get_factor_id_range(variable, vmap, var_samp, evidence)  # get_factor_id_range returns the ID range of the factor
    # TODO: is it possible to avoid copying around fids
    if evidence != proposal:
        range_prop = get_factor_id_range(variable, vmap, var_samp, proposal)
        s1 = range_fids[1] - range_fids[0]
        s2 = range_prop[1] - range_prop[0]
        s = s1 + s2
        fids[:s1] = factor_index[range_fids[0]:range_fids[1]]
        fids[s1:s] = factor_index[range_prop[0]:range_prop[1]]
        fids[:s].sort()
    else:
        s = range_fids[1] - range_fids[0]
        fids[:s] = factor_index[range_fids[0]:range_fids[1]]

    truncate = random.random() < 1.0 / truncation if regularization == 1 else False  # random() returns a random float number in the range [0,1)
    # go over all factor ids, ignoring dupes
    last_fid = -1  # numba 0.28 would complain if this were None
    for factor_id in fids[:s]:  # Iterate over all factors
        if factor_id == last_fid:
            continue
        last_fid = factor_id
        weight_id = factor[factor_id]["weightId"]
        if weight[weight_id]["isFixed"]:    # If the weight is fixed, no need to update
            continue
        # Compute Gradient
        p0 = eval_factor(factor_id, var_samp,
                         evidence, var_copy,
                         variable, factor, fmap,
                         var_value_evid)
        p1 = eval_factor(factor_id, var_samp,
                         proposal, var_copy,
                         variable, factor, fmap,
                         var_value)
        # If parameterized
        if weight[factor[factor_id]['weightId']]['parameterize']:
            x = fmap[factor[factor_id]["ftv_offset"]]['x']      # Find x based on the offset
            theta = fmap[factor[factor_id]["ftv_offset"]]['theta']
            gradient1 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (x - weight[factor[factor_id]['weightId']]['b'])
            gradient2 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (-weight[factor[factor_id]['weightId']]['a'])
            weight[factor[factor_id]['weightId']]['a'] -= step * gradient1  # update a
            weight[factor[factor_id]['weightId']]['b'] -= step * gradient2  # Update b
            a = weight[factor[factor_id]['weightId']]['a']
            b = weight[factor[factor_id]['weightId']]['b']
            if regularization == 2:  # Whether regularization is needed
                a *= (1.0 / (1.0 + reg_param * step))
                a -= step * gradient1
                b *= (1.0 / (1.0 + reg_param * step))
                b -= step * gradient2
            elif regularization == 1:
                # Truncated Gradient
                # "Sparse Online Learning via Truncated Gradient"
                #  Langford et al. 2009
                a -= step * gradient1
                b -= step * gradient2
                if truncate:
                    l1delta = reg_param * step * truncation
                    a = max(0, a - l1delta) if a > 0 else min(0, a + l1delta)
                    b = max(0, b - l1delta) if b > 0 else min(0, b + l1delta)
            else:
                a -= step * gradient1
                b -= step * gradient2

            w = theta * a * (x - b)
        else:  # If not parameterized
            gradient = (p1 - p0) * factor[factor_id]["featureValue"]
            # Update weight
            w = weight_value[weight_copy][weight_id]
            if regularization == 2:
                w *= (1.0 / (1.0 + reg_param * step))
                w -= step * gradient
            elif regularization == 1:
                # Truncated Gradient
                # "Sparse Online Learning via Truncated Gradient"
                #  Langford et al. 2009
                w -= step * gradient
                if truncate:
                    l1delta = reg_param * step * truncation
                    w = max(0, w - l1delta) if w > 0 else min(0, w + l1delta)
            else:
                w -= step * gradient
        weight_value[weight_copy][weight_id] = w
        weight[factor[factor_id]['weightId']]['initialValue'] = w
