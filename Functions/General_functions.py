import timeit
from itertools import cycle

import scipy
import numpy as np
import matplotlib.pyplot as plt
import _tkinter
import datetime
import os
from scipy.special import erfc, erf
import sys
import inspect
import glob
import random
import scipy.integrate as integ
from PyAstronomy.pyaC import zerocross1d
import pandas as pd
import scipy.constants as const

def retrieve_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def find_nearest_index_in_array(array, value, sorted_array = False):
    '''
    A function that returns the nearest index in an array to the input value. If the array is sorted, a much faster
    method (~1000 times faster) can be used by setting sorted_array = True
    :param array: THe array to be searched
    :param value: The value to search for
    :keyword sorted_array = False: If True, a bisection method is used that is 1000 times faster in finding the index
    :return: index
    '''
    array = np.asarray(array)
    if sorted_array:
        n = len(array)
        if (value < array[0]):
            return 0
        elif (value > array[-1]):
            return n-1
        jl = 0  # Initialize lower
        ju = n - 1  # and upper limits.
        while (ju - jl > 1):  # If we are not yet done,
            jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl = jm  # and replace either the lower limit
            else:
                ju = jm  # or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):  # edge cases at bottom
            return 0
        elif (value == array[n - 1]):  # and top
            return n - 1
        else:
            m = (array[jl] + array[ju]) / 2.0
            if value - m < 0:
                return jl
            else:
                return ju
    else:
        index = (np.abs(array - value)).argmin()
        return index

def input_loop_until_correct_value_given(variable=False, variable_type = float, input_message = False):
    '''
    Reduces having the user input a variable value with error handling until a correct value type is given to a single line.
    Variable can be defined by the function instead, meaning no input for variable given.
    :keyword variable = False: Variable to be updated, if wanted for clarity.
    :keyword variable_type = float (int, float, etc): The type of the input variable.
    :keyword input_message = False (True | False): If False, it uses a simple input message with the variable name.
    :return: input variable.
    '''
    input_message = input_message if input_message else '\nInput value for {} variable: '.format(retrieve_variable_name(variable))
    while True:
        try:
            variable = variable_type(input(input_message))
            break
        except Exception as error:
            print('\WARNGING! {} error occured. Incorrect input. Input variable with type {}.'.format(error, variable_type))
    return variable


def indeces_for_list_items_containing_object(search_list, object):
    indices = [index for index, element in enumerate(search_list) if object in element]
    return indices


def display_bmp_image(bmp_image, *args):
    bmp_image.show() if 'skip' not in args else 5
    return

def divide_dic_entries_by_val(dic, val=0, sum_of_entries = False, return_val = False):
    '''
    Takes dictionary of any dimension (can be dic of dic of dic...) and divides each entry by a given value (or by sum
    of all entries if sum_of_entries = True) and returns the new dictionary. Can also return the value used to divide
    each entry
    :param dic:
    :keyword val = 0: Input value to divide each entry by. Overridden if sum_of_entries = True
    :keyword sum_of_entries = False (True|False): If True, overrides val and divides each entry by total sum of entries
    :keyword return_val = False (True|False): If True, return gives new_dic, division_val
    :return: dic if sum_of_entries = False else dic, val
    '''
    if sum_of_entries: #Choice to replace val by sum of all dictionary entries.
        val = sum_of_dictionary_entries(dic)
        if val == 0:
            print('Warning! Sum of {} entries is 0. Skipping division process'.format(retrieve_variable_name(dic)))
            return dic
    dims = find_num_dimensions_of_dic(dic)
    all_keys, dim_lengths, all_vals, new_dic = [],[],[], dic
    for i in range(dims):
        if i == 0:
            temp = []
            keys = list(dic.keys())
            all_keys.append(keys)
            dim_lengths.append(len(keys))
            for key in keys:
                temp.append(dic[key])
        else:
            new_temp = []
            for dictionary in temp:
                keys = list(dictionary.keys())
                all_keys.append(keys)
                for key in keys:
                    new_temp.append(dictionary[key])
            dim_lengths.append(int(len(new_temp)/len(temp)))
            temp = new_temp
    all_vals = np.asarray(temp)
    new_all_vals = all_vals/float(val)
    #creating new dictionary with reduced values
    for i in range(dims):
        start_point = 1 if dims >> 1 else 0
        for j in range(dims-(i+2)):
            start_point += dim_lengths[j]*dim_lengths[j-1] if j >> 0 else dim_lengths[j]
        dim_product = 1
        for j in range(i + 2, dims + 1):
            dim_product *= dim_lengths[-j]
        if i == 0: #First loop, unpack most embedded dictionaries and assign new values to them.
            dics_list = []
            n=0
            for j in range(start_point, start_point+dim_product):
                temp_dic = {}
                for k in range(len(all_keys[j])):
                    temp_dic[all_keys[j][k]] = new_all_vals[n]
                    n+=1
                if dims == 1:
                    final_dic = temp_dic
                else:
                    dics_list.append(temp_dic)

        elif i != dims-1:
            new_dics_list = []
            n = 0
            for j in range(start_point, start_point+dim_product):
                temp_dic = {}
                for k in range(len(all_keys[j])):
                    temp_dic[all_keys[j][k]] = dics_list[n]
                    n+=1
                new_dics_list.append(temp_dic)
            dics_list = new_dics_list
        else: #final loop
            final_dic = {}
            n = 0
            for key in all_keys[0]:
                final_dic[key] = dics_list[n]
                n+=1
    if return_val:
        return final_dic, val
    else:
        return final_dic

def find_num_dimensions_of_dic(dic):
    '''
    Put in a dictionary (can be a dic of dic of dics etc), returns the number of dimensions of that dictionary. This
    is achieved by seeing if there are any more keys down the line of the first key entry of each dimension, so it's
    important the multidimensional dictionaries represent consistent matrices (there is a value for every single spot in
    the matrix)
    :param dic:
    :return: dimensions (int)
    '''
    dims = 0
    new_dic = dic
    while True:
        try:
            keys = list(new_dic.keys())
        except TypeError and AttributeError:
            break
        dims += 1
        new_dic = new_dic[keys[0]]
    return dims

def sum_of_dictionary_entries(dic):
    '''
    Returns the sum of all entries of a dictionary of any dimension (can be a dic of dic of dics n times)
    :param dic:
    :return: sum of dictionary entries
    '''
    dims = find_num_dimensions_of_dic(dic)
    for i in range(dims):
        if i == 0:
            temp = []
            keys = list(dic.keys())
            for key in keys:
                temp.append(dic[key])
        else:
            new_temp = []
            for dictionary in temp:
                keys = list(dictionary.keys())
                for key in keys:
                    new_temp.append(dictionary[key])
            temp = new_temp
    return np.sum(temp)

def find_value_in_dic_return_keys(dic, val=0, find_max = False, find_min = False, return_val = False):
    '''
    A function that takes a dictionary of arbitrary dimensions and then returns the keys corresponding to the given
    value in the dictionary and returns them as a list in order of the dimension order. Each dimension doesn't have to
    be the same size, but the keys to each dictionary corresponding to a dimension must be the same. `Dimension' refers
    to the number of layers of dictionaries within dictionaries, from 1 up to n. For example, a 2 dimensional dic with
    dimensions of 2 x 3 must look like {'a': {'x': 1, 'y':2}, 'b':{'x':3, 'y':4}}. Returns ['b', 'y']
    :param dic:
    :keyword val = 0: Value to be found. If using find_max or find_min, this value is overriden.
    :keyword find_max = False (True|False) Find the maximum value of the dictionary. Overrides val
    :keyword find_min = False (True|False) Find the minimum value in the dictionary. Overrides val
    :keyword return_val = False (True|False) also return the maximum value found
    :return: [keys] if return_max = False else [keys], max_value
    '''
    dims = find_num_dimensions_of_dic(dic)

    #Turn each dimension of keys into a list within all keys, and find the dimension lengths. Also
    #get all values as single list (all_vals). Index of each position with dimension is found by dividing out product
    #of all remaining dimension lengths and finding remainder. Then, subtract everything but remainder from the index to
    #form the new index. Repeat until 1 dimension left.
    all_keys, dim_lengths, all_vals, new_dic = [], [], [], dic
    for i in range(dims):
        keys = list(new_dic.keys())
        all_keys.append(keys)
        dim_lengths.append(len(keys))
        if i == 0:
            temp = []
            for key in keys:
                temp.append(dic[key])

        else:
            new_temp = []
            for dictionary in temp:
                for key in keys:
                    new_temp.append(dictionary[key])
            temp = new_temp
        new_dic = new_dic[keys[0]]

    all_vals = temp
    val = max(all_vals) if find_max else val
    val = min(all_vals) if find_min else val
    val_index = find_nearest_index_in_array(all_vals, val)
    index, max_index_keys = val_index, []
    for i in range(dims):
        if i == dims-1:
            max_index_keys.append(all_keys[i][index])
        else:
            if i == dims - 2:
                dim_product = dim_lengths[-1]
            else:
                dim_product = 1
                for j in range(i+1, dims):
                    dim_product*= dim_lengths[j]
            pos_in_dimension = int(index/dim_product)
            max_index_keys.append(all_keys[i][pos_in_dimension])
            index = index - dim_product*int(index/dim_product)
    if return_val:
        return max_index_keys, val
    else:
        return max_index_keys

def convert_multi_dimension_dic_to_single_dimension(dic, display_time_taken = 'n'):
    dims = find_num_dimensions_of_dic(dic)
    if dims >= 6:
        print('\nWARNING! convert_multi_dimension_dic_to_single_dimension function can only handle dictionary of dimension up to 5.')
        print('input dictionary has dimension {}. Returning original dictionary'.format(dims))
        return dic
    start_time = timeit.default_timer()
    final_dic = {}
    if dims >= 2:
        first_keys = list(dic.keys())
        for first_key in first_keys:
            second_keys = list(dic[first_key].keys())
            for second_key in second_keys:
                if dims == 2:
                    final_dic[str(first_key) + ',' + str(second_key)] = dic[first_key][second_key]
                else:
                    third_keys = list(dic[first_key][second_key].keys())
                    for third_key in third_keys:
                        if dims == 3:
                            final_dic[str(first_key)+','+str(second_key)+','+str(third_key)] = dic[first_key][second_key][third_key]
                        else:
                            fourth_keys = list(dic[first_key][second_key][third_key].keys())
                            for fourth_key in fourth_keys:
                                if dims == 4:
                                    final_dic[str(first_key)+','+str(second_key)+','+str(third_key)+','+str(fourth_key)] = dic[first_key][second_key][third_key][fourth_key]
                                else:
                                    fifth_keys = list(dic[first_key][second_key][third_key][fourth_key].keys())
                                    for fifth_key in fifth_keys:
                                        final_dic[str(first_key)+','+str(second_key)+','+str(third_key)+','+str(fourth_key)+','+str(fifth_key)] = dic[first_key][second_key][third_key][fourth_key][fifth_key]
    end_time = timeit.default_timer()
    print('Dictionary conversion from {} dimensions to single dimension took {} seconds'.format(dims, end_time-start_time)) if display_time_taken == 'y' else 0
    return final_dic

def sum_max_values_in_dic_until_threshold_reached(dic, val=0.95, return_sum = False):
    '''
    Takes dictionary of dimension up to 5, converts it into a single dimension dictionary (using existing function) where
    the keys of each dimension are combined as a string with comma separation (so, if the dic was {'a':{1:0, 2:,0.3}},
    the new dic would be {'a,1':0, 'a,2':0.3}. Then, it finds the maximum value in this dictionary, removes it, and stores
    the value with its corresponding key in a new dictionary. This process is repeated until the sum of entries in the new
    dictionary exceeds a given value. This is useful for finding all values that correspond to a 95 % probability range
    if the dic contains probability densities, for example.
    :param dic: The dictionary (DIM <= 5)
    :param val: The value to sum up to.
    :keyword return_sum = False (True|false): If true, return gives the total sum of the returned dictionary (adds values until
    val is met or exceeded, could go over).
    :return: max_values_dic if return_sum = False else max_values_dic, sum(max_values_dic)
    '''
    single_dim_dic = convert_multi_dimension_dic_to_single_dimension(dic)
    max_values_dic = {}
    summed_value = 0
    while summed_value <= val:
        max_keys, max_value = find_value_in_dic_return_keys(single_dim_dic, find_max=True, return_val=True)
        max_values_dic[max_keys[0]] = max_value
        summed_value += max_value
        single_dim_dic.pop(max_keys[0])
    if return_sum:
        return max_values_dic, summed_value
    else:
        return max_values_dic

def swap_key_positions_in_dic(dic, key_pos_initial, key_pos_final, string_seperator = ',', swap = 'y'):
    '''
    Takes 1D dic and swaps the position of names in the key string that forms it. Can choose to not perform swap, so that
    key is placed in desired position and pushes everything down after that.
    :param dic:
    :param key_pos_initial:
    :param key_pos_final:
    :keyword string_seperator = ',': The string that separates values in the dictionary key. E.g. if the key is 'a:b',
    string separator = ':'
    :keyword swap = 'y' (y/n)
    :return: dic with new keys
    '''
    for key in list(dic.keys()):
        dic_val = dic[key]
        dic.pop(key)
        key = key.split(string_seperator)
        new_key = key.copy()
        new_key[key_pos_final] = key[key_pos_initial]
        if swap == 'y':
            new_key[key_pos_initial] = key[key_pos_final]
        else:
            passed = 'n'
            for j in range(key_pos_final+1, len(key)):
                if j-1 == key_pos_initial:
                    new_key[j] = key[j]
                    passed = 'y'
                else:
                    new_key[j] = key[j-1] if passed == 'n' else key[j]
        final_new_key = ''
        for i in range(len(new_key)-1):
            final_new_key += new_key[i]+string_seperator
        final_new_key += new_key[-1]
        dic[final_new_key] = dic_val
    return dic

def user_check(*query):
    '''
    :param query:
    :return: y/n
    '''
    if len(query) == 0:
        query = 'Are you willing to proceed? y/n: '
    else:
        if 'y/n:' in query[0]:
            query = query[0]
        else:
            query = query[0] + ' y/n: '
        pass
    while True:
        check = input(query).lower()
        if check == 'y' or check == 'n':
            return check
        else:
            print('Please type "y" or "n", you duffer.')


def close_figure_with_keyboard_or_mouse(*args):
    '''
    :return: function requires plt.close(fig) after use
    '''
    try:
        plt.waitforbuttonpress()
    except _tkinter.TclError:
        pass
    return


def linear_function(x, m, c):
    '''
    :return: m*x + C
    '''
    return m * x + c

def any_power_x(x,m, exp, c):
    '''
    :return: m*x**exp + C
    '''
    return m*x**exp + c

def exponential(x, A, B, C):
    '''
    :return: A * np.exp(B*x) + C
    '''
    return A * np.exp(B * x) + C

def exponential_asymptote(x, A, B, C):
    '''
    :return: A * (1 - np.exp(-B*x)) + C
    '''
    return A*(1-np.exp(-B*x)) + C

def generalised_exponential_asymptote(x, A, B, K, C):
    '''
    :return: A*(1-np.exp(-B*(x+K))) + C
    '''
    return A*(1-np.exp(-B*(x+K))) + C


def derivative_of_exponential(x, A, B):
    '''
    A function that is the derivative of the function A * np.exp(B * x) + C
    :return: A * B * np.exp(B * x)
    '''
    return A*B*np.exp(B*x)

def langmuir_maxwellian_integral(U, A, B, V, V_p):
    return A * U * np.sqrt(U**2 + 2*const.e * (V-V_p) / const.m_e) * B * np.exp(-B * U**2)

def curve_fit_langmuir_maxwellian_integral(V, A, B, V_p):
    return [integ.quad(langmuir_maxwellian_integral, np.sqrt(abs(2 * const.e * (x - V_p) / const.m_e)) if x <= V_p else 0, 1e8, args = (A, B, x, V_p))[0] for x in V]

def curve_fit_langmuir_double_maxwellian_integral(V, A_1, B_1, A_2, B_2, V_p):
    return [integ.quad(langmuir_maxwellian_integral, np.sqrt(abs(2 * const.e * (x - V_p) / const.m_e)) if x <= V_p else 0, 1e8, args = (A_1, B_1, x, V_p))[0] +
            integ.quad(langmuir_maxwellian_integral, np.sqrt(abs(2 * const.e * (x - V_p) / const.m_e)) if x <= V_p else 0, 1e8, args = (A_2, B_2, x, V_p))[0] for x in V]

def curve_fit_ln_langmuir_double_maxwellian_integral(V, A_1, B_1, A_2, B_2, V_p):
    return np.log([integ.quad(langmuir_maxwellian_integral, np.sqrt(abs(2 * const.e * (x - V_p) / const.m_e)) if x <= V_p else 0, 1e8, args = (A_1, B_1, x, V_p))[0] +
            integ.quad(langmuir_maxwellian_integral, np.sqrt(abs(2 * const.e * (x - V_p) / const.m_e)) if x <= V_p else 0, 1e8, args = (A_2, B_2, x, V_p))[0] for x in V])

def maxwellian_integral(U, T_e, n_e, r_p, l_p, V):
    '''
    The Maxwellian integral from Langmuir OML with infinite sheath approximation. To be used directly with integ.quad, for example,
    to create an ideal characteristic from known input parameters
    :param U: represents electron velocity
    :param T_e: Electron temp
    :param n_e: electron Density
    :param V: Voltage difference from plasma potential
    :return: General form of current collected by a cylindrical probe when V <= V_p
    '''
    A = 4*np.pi*r_p*l_p*n_e*const.e
    A_exp = const.m_e/(2*const.k*T_e)
    return A * U * np.sqrt(U ** 2 + 2 * const.e * V / const.m_e) * (A_exp / np.pi) * np.exp(-A_exp * U ** 2)

def drifting_maxwellian_integral(theta, U, T_e, n_e, u_0, r_p, l_p, V):
    '''
    The drifting maxwellian integral from Langmuir OML with infinite sheath approximation. This is used with integ.dblquad,
    which requres that the inner integral be placed second in the function definition. This is confusing though, since when calling
    #integ.dblquad, you have to put the limits for the inner integral first and then the limits for the outer integral second (as function
    #outputs themselves).
    :param theta: Free param, angle between drift velocity and surface section of probe being considered.
    :param U: Free param representing electron velocity
    :param T_e: Electron temp
    :param n_e: Electron density
    :param u_0: Drift velocity of electrons (m/s)
    :param V: Voltage difference from plasma potential
    :return: Integral sum of current for a given probe potential.
    '''
    A = 2 * r_p * l_p * n_e * const.e

    A_exp = const.m_e / (2*const.k * T_e)

    return A * U * np.sqrt(U**2 + 2 * const.e * V / const.m_e) * (A_exp/np.pi) * np.exp(-A_exp*(U**2 + u_0**2 + 2*u_0*U*np.cos(theta)))

def langmuir_exponential(x, A, B, C):
    return (A/np.sqrt(B))*np.exp(B*x) + C

def langmuir_exponential_varying_start(x, A, B, C, D):
    return (A/np.sqrt(B))*np.exp(B*(x+D)) + C


def maxwellian_langmuir_finite_sheath(x, A, B, r_s, C):
    r_p = 50.0e-6
    print('\nUsing probe radius of {} m in Maxwellian fit.'.format(r_p))
    return A * (r_s * (1 - erfc(np.sqrt(r_p ** 2 * (B * x) / (r_s ** 2 - r_p ** 2)))) + erfc(
        np.sqrt(r_s ** 2 * (B * x) / (r_s ** 2 - r_p ** 2)))) + C


def maxwellian_infinite_sheath_Langmuir(x, A, B, C):
    return (A / np.sqrt(B)) * ((2.0 / np.sqrt(np.pi)) * (np.sqrt(B * x)) + np.exp(B * x) * erfc(np.sqrt(B * x))) + C


def maxwellian_infinite_sheath_Langmuir_approximation_greater_than_2(x, A, B, C):
    return (A / np.sqrt(B)) * (2 / np.sqrt(np.pi)) * np.sqrt(1.0 + B * x) + C


def maxwellian_infinite_sheath_Chen_approximation(x, A, B, C):
    return (A / np.sqrt(B)) * np.sqrt(1.0 + 4 * (B * x / np.pi)) + C


def isotropic_monoenergetic_infinite_sheath(x, A, V_c, C):
    return A * (np.sqrt(x / V_c) + (1.0 + x / V_c) * np.arcsin(np.sqrt(V_c / (x + V_c)))) + C


def double_maxwellian_full_fit_chen(x, A_1, B_1, A_2, B_2):
    out = np.empty_like(x)
    mask = x <= 0
    out[mask] = (A_1 / np.sqrt(B_1)) * np.exp(B_1 * x[mask]) + (A_2 / np.sqrt(B_2)) * np.exp(B_2 * x[mask])
    out[~mask] = (A_1 / np.sqrt(B_1)) * np.sqrt(1.0 + 4.0 * B_1 * x[~mask] / np.pi) + (A_2 / np.sqrt(B_2)) * np.sqrt(
        1.0 + 4.0 * B_2 * x[~mask] / np.pi)
    return out


def double_maxwellian_full_fit_langmuir(x, A_1, B_1, A_2, B_2):
    out = np.empty_like(x)
    mask = x <= 0
    out[mask] = (A_1 / np.sqrt(B_1)) * np.exp(B_1 * x[mask]) + (A_2 / np.sqrt(B_2)) * np.exp(B_2 * x[mask])
    out[~mask] = (A_1 / np.sqrt(B_1)) * (
                (2.0 / np.sqrt(np.pi)) * (np.sqrt(B_1 * x[~mask])) + np.exp(B_1 * x[~mask]) * erfc(
            np.sqrt(B_1 * x[~mask]))) \
                 + (A_2 / np.sqrt(B_2)) * (
                             (2.0 / np.sqrt(np.pi)) * (np.sqrt(B_2 * x[~mask])) + np.exp(B_2 * x[~mask]) * erfc(
                         np.sqrt(B_2 * x[~mask])))
    return out


def single_maxwellian_full_fit_chen(x, A, B):
    out = np.empty_like(x)
    mask = x <= 0  # This is some kind of method for fitting different functions with curve fit about a defined point.
    out[mask] = (A / np.sqrt(B)) * np.exp(B * (x[mask]))
    out[~mask] = (A / np.sqrt(B)) * np.sqrt(1.0 + 4.0 * B * (x[~mask]) / np.pi)
    return out

#IF USING SQRT(v), THEN i IS INDEPENDENT OF T_e
def laframboise_radial_ion_theory(x, A, D):
    return A * D * np.sqrt(x)  # one constant is ion current at plasma potential, the other is the constant in the laframboise function which we can limit

def laframboise_radial_ion_theory_plus_constant(x, A, D, F):
    return A * D * np.sqrt(x) + F


def laframboise_radial_ion_theory_variable_V_p(x, A, C, D):
    return A* D * np.sqrt(x + C)

def laframboise_radial_ion_theory_variable_V_p_plus_constant(x, A, C, D, F):
    return A * D * np.sqrt(x + C) + F

def laframboise_radial_ion_theory_variable_power(x, A, B, D, E):
    return (A / np.sqrt(B)) * D * (B * x) ** E

def cylinder_SA(radius, height):
    return 2 * np.pi * radius * height + 2 * np.pi * radius ** 2


def get_variable_name_as_string(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def user_select_points_from_figure(x_axis, y_axis, fullscreen = 'n', **kwargs):
    '''
    :param x_axis:
    :param y_axis:
    :keyword fullscreen = 'n' (y/n)
    :param kwargs: num_points | message | plot_type (empty/semilogy/semilogx/loglog) | linetype | x_label | y_label | title
    :return: points [[x_1,y_1], [x_2,y_2], etc]
    '''
    num_points = int(kwargs['num_points']) if 'num_points' in kwargs else 1
    if 'message' in kwargs:
        print(kwargs['message'])
    else:
        print('\nUser select {} point/s from the figure.'.format(num_points))
    while True:
        plt.close() #just in case one is still open from something else.
        if 'plot_type' not in kwargs:
            plt.plot(x_axis, y_axis, kwargs['line_type']) if 'line_type' in kwargs else plt.plot(x_axis, y_axis)
        elif kwargs['plot_type'].lower().replace(' ', '') == 'semilogy':
            plt.semilogy(x_axis, y_axis, kwargs['line_type'], base=np.e) if 'line_type' in kwargs else plt.semilogy(x_axis, y_axis, base=np.e)
        elif kwargs['plot_type'].lower().replace(' ', '') == 'semilogx':
            plt.semilogx(x_axis, y_axis, kwargs['line_type'], base=np.e) if 'line_type' in kwargs else plt.semilogy(x_axis, y_axis, base=np.e)
        elif kwargs['plot_type'].lower().replace(' ', '') == 'loglog':
            plt.loglog(x_axis, y_axis, kwargs['line_type'], basex=np.e, basey=np.e) if 'line_type' in kwargs else plt.semilogy(x_axis, y_axis,basex=np.e, basey=np.e)
        print('\nUse left mouse to seelct points, right mouse to remove last point, middle mouse button to finish after selecting points')
        plt.xlabel(kwargs['x_label']) if 'x_label' in kwargs else 0
        plt.ylabel(kwargs['y_label']) if 'y_label' in kwargs else 0
        plt.title(kwargs['title']) if 'title' in kwargs else 0
        plt.get_current_fig_manager().full_screen_toggle() if fullscreen == 'y' else 0
        points = plt.ginput(num_points, timeout=0)
        plt.close()
        fuckup = user_check('\nDid you fuck up the points? y/n: ')
        if fuckup.lower().replace(' ', '') == 'n':
            break
    return points


def finish_fullscreen_with_tight_layout(**kwargs):
    '''
    Finishes a plot in fullscreen mode with tight layout.
    :param kwargs: fig_object (can pass a figure object if needed)
    :return:
    '''
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    plt.close(kwargs['fig_object']) if 'fig_object' in kwargs else plt.close()
    return

def clip_array(array, remove_val, **kwargs):
    '''
    Function that takes an array and removes remove_val vals from the start, end, or both ends of the array
    :param array: array to be dealt with
    :param remove_val: Value to be removed from chosen sides. remove_val = 0 returns unchanged array
    :param kwargs: remove_from (MUST) = 'left'/'right'/'both'
    :return: clipped array
    '''
    kwargs['remove_from'] == 'left' if remove_val == 0 else kwargs['remove_from']
    while True:
        if kwargs['remove_from'] == 'both':
            array = array[remove_val:-remove_val+1]
            break
        elif kwargs['remove_from'] == 'left':
            array = array[remove_val:]
            break
        elif kwargs['remove_from'] == 'right':
            array = array[:-remove_val+1]
            break
        else:
            kwargs['remove_from'] == input('\nPlease choose which side of the array to clip (left/right/both): ')
    return array


def find_intersection_point_of_two_lines(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# Given a set of datasets which have multiple y values per x value, return a list of datasets that have 1 averaged y value per x value.
# Excellent for returning a datset from staircase of increasing x values, like in the langmuir probe pulses
def remove_staircase_aspect(number_of_points_per_step, *args):
    """
    :param number_of_points_per_step:
    :param args: you need to put your staircase arrays here as args, so the function can handle a variable number of them
    :return: final_datasets, a list of the inputted datasets
    """
    final_datasets = []
    for dataset in args:
        i, final_dataset = 0, []
        while len(final_dataset) < len(dataset) / number_of_points_per_step:
            add = 0
            for j in range(number_of_points_per_step):
                add += dataset[i + j] if i+j < len(dataset) else 0
            final_dataset.append(add / number_of_points_per_step)
            i += number_of_points_per_step
        final_datasets.append(np.asarray(final_dataset))
    return final_datasets


# Takes inputs and adds them to the wanted dictionary
def convert_datavals_to_dictionary(*args, print_additions = 'n', print_addition_values = 'n', **kwargs):
    '''
    A function that takes variables and either creates a dictionary which has the variable names as keys mapping to the variable values or otherwise
    adds such entries to an existing dictionary.
    :param args: The variables to be added
    :keyword print_additions = 'n' (y/n): print what variables are being added to the dictionary
    :keyword print_addition_values = 'n' (y/n): Print each variable added name along with its value
    :param kwargs: dictionary_to_add_to (Optional: dictionary to add entries too.
    :return: New dictionary or updated old dictionary
    '''
    name_dic = kwargs['dictionary_to_add_to'] if 'dictionary_to_add_to' in kwargs else {}
    names = []
    for variable in args:
        name_dic[retrieve_variable_name(variable)] = variable
        names.append(retrieve_variable_name(variable))
    print('\nAdding variables named {} to {} dictionary'.format(names, retrieve_variable_name(kwargs['dictionary_to_add_to']) if 'dictionary_to_add_to' in kwargs else 'created')) if print_additions == 'y' else 0
    if print_addition_values == 'y':
        for name in names:
            print('{} = {}'.format(name, name_dic[name]))
    return name_dic


def search_down_array_for_exact_point(x_array, y_array, **kwargs):
    '''
    A function that searches up and down a y_array for the first time it crosses a certain value. It then gives you the
    corresponding value of the same point in the x_array
    :param x_array:
    :param y_array:
    :param kwargs: y = (float) MUST (the value to be searched for)
    :param kwargs: search_direction = 'left/right' MUST (left = start from right and search left)
    :return: x_value (exact float value of point in x, even if not corresponding to an index value (interpolates for o)), y_value
    '''
    val = float(kwargs['y'])
    while True:
        if kwargs['search_direction'] == 'left':
            for i in range(len(x_array)):
                if y_array[-i - 1] - val <= 0:
                    x, y = np.asarray([x_array[-i - 1], x_array[-i]]), np.asarray([y_array[-i - 1]-val, y_array[-i]-val])
                    break
            break
        elif kwargs['search_direction'] == 'right':
            for i in range(len(x_array)):
                if y_array[i] - val <= 0:
                    x, y = np.asarray([x_array[i - 1], x_array[i]]), np.asarray([y_array[i - 1]-val, y_array[i]-val])
                    break
            break
        else:
            kwargs['search_direction'] = input('\nPlease enter "left" or "right" for search_direction: ')
    x_val = zerocross1d(x, y)
    return x_val[0], val,  # y_val will be contained by the value you entered to search for


def slope_and_constant_of_line_from_two_points(points):
    gradient = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    constant = points[1][1] - gradient * points[1][0]
    return gradient, constant


def save_dictionary_as_CSV_in_specified_folder(save_dic, save_path, file_name = False, append_to_existing = False):
    '''
    A function that converts a dictionary to a pandas dataframe and saves it as a CSV to the savepath, creating that savepath
    if the directory doesn't exist yet. If no filename given, it uses the name of the dictionary. There is also an option
    to append the datafile to an existing csv. Can also directly save dataframes
    :param save_dic: The dictionary to save
    :param save_path: The path to save it to, not including the name
    :keyword file_name = False: If given a name, determines the name of the saved file. Otherwise, the function uses the name of the dic.
    :param append_to_existing = False: If true, appends to an existing dictionary.
    :return:
    '''
    if type(save_dic) == dict:
        try:
            df = pd.DataFrame(save_dic)
        except ValueError:
            df = pd.DataFrame(save_dic, index=['value:'])
    else:
        df = save_dic
    file_name = file_name or retrieve_variable_name(save_dic)
    file_name = file_name + '.csv' if '.csv' not in file_name else file_name
    if append_to_existing:
        if os.path.isfile(os.path.join(save_path, file_name)) == True:
            print('\nAppending {} dictionary to existing {}'.format(retrieve_variable_name(save_dic), file_name))
            df.to_csv(os.path.join(save_path, file_name), mode='a')
        else:
            print('\nCreating {} file.'.format(os.path.join(save_path, file_name)))
            df.to_csv(os.path.join(save_path, file_name))
    else:
        print('\n{} file already exists. Overwriting.'.format(file_name)) if os.path.isfile(os.path.join(save_path, file_name)) == True else print('\nCreating {} file.'.format(os.path.join(save_path, file_name)))
        df.to_csv(os.path.join(save_path, file_name))
    return


def create_directory_if_it_doesnt_exist(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print('\nCreating directory:', path)
    return

def swap_array_names(array1, array2, *args, **kwargs):
    if kwargs['swap_names'].lower().replace(' ', '') == 'y':
        print('Swapping {} with {}'.format(retrieve_variable_name(array1), retrieve_variable_name(array2))) if 'multiple' not in args else True
        return array2, array1
    else:
        print('NOT swapping {} with {}'.format(retrieve_variable_name(array1), retrieve_variable_name(array2))) if 'multiple' not in args else True
        return array1, array2

def flip_array_if_needed(array, **kwargs):
    '''
    A function that can either flip or not flip an array and also add offsets.
    :param array:
    :param kwargs: flip_array ('y/n')
    :param kwargs: add_current_offset (optional float) offset to be added to array after flipping
    :param kwargs: add_array_min (optional 'y/n') adds the minimum of the array after flipping such that the array min value is 0 (good for log plots)
    :return: new array
    '''
    if kwargs['flip_array'].lower().replace(' ', '') == 'y':
        if 'add_current_offset' in kwargs:
            return -array + kwargs['add_current_offset']
        elif 'add_array_min' in kwargs:
            return -array - min(array)
        return -array
    else:
        if 'add_current_offsest' in kwargs:
            return array + kwargs['add_current_offset']
        return array

def cycle_through_datasets_and_remove(x_dataset, y_dataset, *args, **kwargs):
    '''
    Takes input datasets as a list of datasets and allows you to remove particular datasets from it
    :param x_dataset: list of datasets
    :param y_dataset: list of datasets
    :param args: plot (see final product)
    :param kwargs: labels (optional, must be list of strings) | x_label (optional) | y_label (optional) | title (optional) | fullscreen (y/n)
    :return: x_dataset, y_dataset
    '''
    while True:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colours = cycle(prop_cycle.by_key()['color'])
        for i in range(len(x_dataset)):
            lab = str(kwargs['labels'][i]) if 'labels' in kwargs else int(i)
            plt.plot(x_dataset[i], y_dataset[i], label=lab, color = next(colours))
        plt.legend()
        plt.xlabel(kwargs['x_label']) if 'x_label' in kwargs else 0
        plt.ylabel(kwargs['y_label']) if 'y_label' in kwargs else 0
        plt.title(kwargs['title']) if 'title' in kwargs else plt.title('select dataset integer to remove')
        plt.tight_layout()
        mng = plt.get_current_fig_manager() if kwargs['fullscreen'] == 'y' else 0
        mng.window.state('zoomed') if kwargs['fullscreen'] == 'y' else 0
        close_figure_with_keyboard_or_mouse()
        remove_val = input('\nType integer of dataset to remove, or "n": ')
        if remove_val.lower().replace(' ', '') == 'n':
            break
        else:
            x_dataset.pop(int(remove_val)), y_dataset.pop(int(remove_val))
    if 'plot' in args:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colours = cycle(prop_cycle.by_key()['color'])
        for i in range(len(x_dataset)):
            lab = kwargs['labels'][i] if 'labels' in kwargs else int(i)
            plt.plot(x_dataset[i], y_dataset[i], label=lab, color=next(colours))
        plt.legend()
        plt.xlabel(kwargs['x_label']) if 'x_label' in kwargs else 0
        plt.ylabel(kwargs['y_label']) if 'y_label' in kwargs else 0
        plt.title(kwargs['title']) if 'title' in kwargs else plt.title('select dataset integer to remove')
        plt.tight_layout()
        mng = plt.get_current_fig_manager() if kwargs['fullscreen'] == 'y' else 0
        mng.window.state('zoomed') if kwargs['fullscreen'] == 'y' else 0
        close_figure_with_keyboard_or_mouse()
    return x_dataset, y_dataset

#Performs a notch filter on each current dataset, which removes a specific frequency
def notch_filter(time_data, signal_data, notch_freq, quality_factor, *args, **kwargs):
    '''
    A function that removes a specific frequency (with a narrow domain) from your time dataset.
    :param time_data: (must contain at least the time between 0 and 1)
    :param signal_data:
    :param notch_freq: The freq to be removed
    :param quality_factor: A tradeoff between the bandwidth of removed signal and the reduction in decibels of the notch freq
    :param args: skip | plot | plot_filter (plots the filter (transfer function) that is multiplied against the signal
    :param kwargs: suptitle (for figure)
    :return: output_signal (signal with frequency removed
    '''
    if 'skip' in args:
        return signal_data
    #First, we need to recover the sample frequency. This is how many samples you have in time per second.
    time_start, time_end = find_nearest_index_in_array(time_data, 0), find_nearest_index_in_array(time_data, 1)
    sample_freq = time_end-time_start #This is the sample frequency of the dataset (the number of datapoints per second)

    #Compute the notch filter numerator and denomiator (the function that is multiplied against the input signal to creat the output signal)
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, sample_freq)

    #compute the magnitude response of the filter
    freq, h = scipy.signal.freqz(b_notch, a_notch, fs=sample_freq)
    if 'plot_filter' in args:
        fig = plt.figure(figsize=(8,6))
        plt.plot(freq, 20*np.log10(abs(h)), 'r', label = 'Bandpass filter', linewidth = '2')
        plt.xlabel('Frequency (Hz)')
        plt.xscale('log')
        plt.ylabel('Magnitude [dB]')
        plt.title('Notch filter')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close(fig)

    #Now normally we would create a time vector, but we already technically have one. However, i'm going to shift it, possibly, so that it starts at 0 and goes upwards?
    num_secs = time_data[-1] - time_data[0]
    n = np.linspace(0, num_secs, len(time_data))
    output_signal = scipy.signal.filtfilt(b_notch, a_notch, signal_data)

    if 'plot' in args:
        fig2 = plt.figure(figsize=(8,6))
        plt.subplot(211)
        plt.plot(time_data, signal_data, color ='r', linewidth = 2)
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.title('Original signal')

        plt.subplot(212)
        plt.plot(time_data, output_signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.title('Filtered signal')
        plt.subplots_adjust(hspace = 0.5)
        plt.suptitle(kwargs['suptitle']) if 'suptitle' in kwargs else 0
        finish_fullscreen_with_tight_layout(fig_object=fig2)

    return  output_signal

def differentiate_array(array, h, *args, method = 'central', dtve = 1, **kwargs):
    '''
    Function that differentiates using either the forward, backward, or central difference approximation
    :param array:
    :param h (dx for dy/dx)
    :param args: 'skip'
    :param DEF_KWARG method: 'forward/backward/central'
    :param DEF_KWARGs: dtve = 1 (1/2) (do first or second derivative. If second derivative, uses central difference).
    :param kwargs: left_val | right_val (preset the left and right value of the differentiated array)
    :return: derivative of array
    '''
    if 'skip' in args:
        return array
    derivative = np.zeros(len(array))
    if int(dtve) == 1:
        for i in range(len(array)):
            if i == 0: #forward difference for first point
                derivative[0] = float(kwargs['left_val']) if 'left_val' in kwargs else (array[1]-array[0])/h
            elif i == len(array)-1: #backward difference for last point
                derivative[-1] = float(kwargs['right_val']) if 'right_val' in kwargs else (array[-1]-array[-2])/h
            else:
                if method == 'central':
                    derivative[i] = (array[i+1]-array[i-1])/(2*h)
                elif method == 'forward':
                    derivative[i] = (array[i+1] - array[i])/h
                elif method == 'backward':
                    derivative[i] = (array[i]-array[i-1])/h
    if dtve == 2:
        for i in range(len(array)):
            if i == 0:
                derivative[0] = float(kwargs['left_val']) if 'left_val' in kwargs else (2*array[i]-5*array[i+1] + 4*array[i+2]-array[i+3])/h**3
            elif i == len(array) - 1:
                derivative[-1] = float(kwargs['right_val']) if 'right_val' in kwargs else (2*array[i] - 5*array[i-1] + 4*array[i-2] - array[i-3])/h**3
            else:
                derivative[i] = (array[i-1] - 2*array[i] + array[i+1])/h**2
    return derivative

def avg_section_of_array(array, left_index = 0, right_index = -1):
    '''
    Returns average of array between left and right index
    :param array: preferably numpy
    :param DEF_KWARG left_index: int
    :param DEF_KWARG right_index: int
    :return: np.mean(array[left_index:right_index])
    '''
    return np.mean(array[left_index:right_index])

def distribution_function_sum(place, voltage_array, distribution_function_array):
    '''
    Performs the sum within the finite difference method for calculating the distribution function from the current to a cylindrical probe.
    :param place: This is annoying with python. The 4th distribution function position, for example, occurs between the 4th and 5th voltage places. so
    place should only start at the third voltage position (to get f_2(5/2))
    :param voltage_array:
    :param distribution_function_array:
    :return: sum from j = 1 up to j = place - 1
    '''
    sum = 0
    if place == 1:
        return sum
    for j in range(1, place): #needs to start at 1 and go upwards
        sum += distribution_function_array[j-1]*( (2.0/3.0)*(const.e*voltage_array[place] - const.e*voltage_array[j-1])**(1.5) -
                        (2.0/3.0)*(const.e*voltage_array[place] - const.e * voltage_array[j])**(1.5))
    return sum

def finite_difference_dist_func_from_cyl_probe(voltage_array, current_array, **kwargs):
    '''
    The function that performs the finite difference calculation to get the distribution function from the current to a
    cylindrical probe without using derivatives. From 'Method to find the electron distribution function from cylindrical probe data'.
    :param voltage_array:
    :param current_array:
    :param kwargs: probe_radius | probe_length
    :return: finite_diff_voltage, f_2_q_phi
    '''
    finite_diff_voltage = []
    f_2_q_phi = []
    for k in range(1,len(current_array)):
        finite_diff_voltage.append((voltage_array[k - 1] + voltage_array[k]) / 2.0)
        if k == 1:
            f_2_q_phi.append(0)
        elif k == 2:
            top = current_array[1]*const.m_e/(2*np.pi*kwargs['probe_radius']*kwargs['probe_length']*const.e)*np.sqrt(const.m_e/2.0)
            bottom = (2.0/3.0)*(const.e*voltage_array[1]-const.e*voltage_array[0])**(1.5)
            f_2_q_phi.append(top/bottom)
        else:
            top_left_val = current_array[k]*const.m_e/(2*np.pi*kwargs['probe_radius']*kwargs['probe_length']*const.e)*np.sqrt(const.m_e/2.0)
            bottom_val = (2.0/3.0) * (const.e * voltage_array[k] - const.e*voltage_array[k-1])**(1.5)
            f_2_q_phi.append( (top_left_val - distribution_function_sum(k, voltage_array, np.array(f_2_q_phi)))/bottom_val)
    return -np.array(finite_diff_voltage), np.array(f_2_q_phi)

def force_arrays_to_same_length(array_list, remove_first_point_from = 'end'):
    '''
    Function that takes list of arrays and reduces all array lengths to be the same as the shortest one.
    :param array_list:
    :param DEF_KWARG remove_first_point_from = 'end' ('end'/'start'): begin removing points from start or end of array.
    :return: new_array_list
    '''
    len_dic, new_array_list = {len(array): array for array in array_list}, []
    min_length = min(list(len_dic.keys()))
    if len(len_dic) > 1:
        print('Arrays in {} not the same length. Reducing to {}'.format(retrieve_variable_name(array_list), min_length))
    for array in array_list:
        i = 0
        while len(array) > min_length:
            if type(array) is tuple == True: #check if numpy array or not
                if remove_first_point_from == 'end':
                    array.pop(-1) if i%2 == 0 else array.pop(0)
                else:
                    array.pop(0) if i%2 == 0 else array.pop(-1)
            else:
                if remove_first_point_from == 'end':
                    array = np.delete(array, -1) if i%2 == 0 else np.delete(array,0)
                else:
                    array=np.delete(array, 0) if i%2 == 0 else np.delete(array, -1)
            i+=1
        new_array_list.append(array)
    return new_array_list

def remove_array_from_list_if_nans(array_list, additional_array_list = False, print_removal_message = 'y'):
    '''
    Function that searches list of arrays and removes any that has nans
    :param array_list:
    :arg *additional_array_lists (array lists to have corresponding arrays removed if nans in array from first array list
    :keyword print_removal_message = 'y' (print a message if NaN array detected)
    :return: new_array_list (alone if additional_array_lists empty), list of additional array lists. (will have to be unpacked unless only 1. eg list of additional array lists[1] for second additional array list
    '''
    new_array_list, new_additional_array_list = [], []
    for i in range(len(array_list)):
        if np.isnan(np.array(array_list[i])).any():
            print('{} contains array {} with NaN. Removing from list.'.format(retrieve_variable_name(array_list), i)) if print_removal_message == 'y' else False
            print('Simultaneously removing array {} from {}'.format(i,retrieve_variable_name(additional_array_list))) if additional_array_list else 0
        else:
            new_array_list.append(array_list[i])
            new_additional_array_list.append(additional_array_list[i]) if additional_array_list else True

    if additional_array_list:
        return new_array_list, new_additional_array_list
    else:
        return new_array_list

def remove_nans_from_array_if_below_tol(array, *args, tolerance = 0.01, array_no = False, print_warning = 'y'):
    '''
    A function that checks an array for NaNs and removes them if the number is below a percentage of the total array values.
    It can also take an additional array and removes entries from both arrays if the entry in either of the arrays is NaN.
    :param array: must be numpy.array()
    :args additional arrays are put in this way.
    :keyword tolerance = 0.01: as fraction of 1 (1% = 0.01)
    :keyword array_no = False: Optional extra to print the array number if repeating nan cleaning
    :keyword print_warning = 'y': Prints the Nan warning if applicable
    :return: array, list of additional arrays if additional array input else array (if only 1 additional array, additional array returned as single array)
    '''
    new_array, new_additional_array_list = [],[]
    num_nans = {}
    nan_arrays = []
    for i in range(len(array)):
        if np.isnan(array[i]):
            num_nans[i] = True
            nan_arrays.append(retrieve_variable_name(array)) if retrieve_variable_name(array) not in nan_arrays else 0
        else:
            for extra_array in args:
                if np.isnan(extra_array[i]):
                    num_nans[i] = True
                    nan_arrays.append(retrieve_variable_name(extra_array)) if retrieve_variable_name(extra_array) not in nan_arrays else 0
    if len(num_nans) == 0:
        new_array = array
        for extra_array in args:
            new_additional_array_list.append(extra_array)
    else:
        if len(num_nans) > len(array)*tolerance:
            print('\nWARNING! {} Nans found in{} files {}, above tolerance of {}%. Removing from arrays.'.format(len(num_nans), ' {}th dataset for'.format(array_no) if array_no else '',  nan_arrays, tolerance*100))
        else:
            print('\nWARNING! {} NaNs out of {} found in{} files {} below total tolerance of {} %. Removing NaNs from {} {}'.format(len(num_nans), ' {}th dataset for'.format(array_no) if array_no else '', len(array), nan_arrays, tolerance*100, retrieve_variable_name(array), nan_arrays)) if print_warning == 'y' else 0
        for n in range(len(array)):
            if n not in list(num_nans.keys()):
                new_array.append(array[n])
        for extra_array in args:
            new_additional_array = []
            for m in range(len(extra_array)):
                if m not in list(num_nans.keys()):
                    new_additional_array.append(extra_array[m])
            new_additional_array_list.append(np.array(new_additional_array))

    new_array = np.asarray(new_array)
    for i in range(len(new_additional_array_list)):
        new_additional_array_list[i] = np.asarray(new_additional_array_list[i])
    if len(new_additional_array_list) == 0:
        return new_array
    elif len(new_additional_array_list) == 1:
        return new_array, new_additional_array_list[0]
    else:
        return new_array, new_additional_array_list

def create_ideal_characteristic_drifting_secondary(*args, n_ec = 1e15, n_eh = 1e13, n_i = 1e15, T_ec_ev = 3, T_eh_ev=10, u_D_ev=10, m_i = 6.6335e-26, r_p = 50.0e-6, l_p = 15.0e-3, V_p = 0, voltage_array = 0, min_voltage = -100, max_voltage=100, num_steps=1001, print_messages = 'n'):
    '''
    Creates an ideal characteristic assuming maxwellian bulk electrons, drifting maxwellian secondary electrons, and stationary ions following Laframboise's theory
    :param args: 'skip'|'plot'
    :param n_ec:
    :param n_eh:
    :param n_i:
    :param T_ec_ev:
    :param T_eh_ev:
    :param u_D_eV: This is the secondary drift velocity, which should be given in eV at first.
    :param m_i:
    :param r_p:
    :param l_p:
    :param V_p:
    :param voltage_array:
    :param min_voltage:
    :param max_voltage:
    :param num_steps:
    :param print_messages = 'n' (y|n): Print the messages made during the calculations.
    :return:
    '''
    if 'skip' in args:
        return 0,0,0,0,0,0,0
    T_ec, T_eh = T_ec_ev*const.e/const.k, T_eh_ev*const.e/const.k
    voltage = np.linspace(min_voltage, max_voltage, num_steps) if len(voltage_array) <= 1 else voltage_array
    voltage_p = voltage - V_p
    ion_voltage = voltage[voltage<=V_p]
    #Creating complicated constants
    I_i = const.e*n_i*2*np.pi*r_p*l_p*np.sqrt(const.k*T_ec/(2*np.pi*m_i))
    eta_pre_c, eta_pre_h = const.e/(const.k*T_ec), const.e/(const.k*T_eh)
    u_D, u_Th = np.sqrt(2*u_D_ev*const.e/const.m_e), np.sqrt(2*T_eh*const.e/const.k)
    bulk_current, secondary_current, total_current = np.zeros(len(voltage)), np.zeros(len(voltage)), np.zeros(len(voltage))
    start_time = timeit.default_timer()
    print('\nBeginning double integral calculation of maxwellian plus drifting maxwellian calculation with {} points'.format(len(voltage))) if print_messages == 'y' else 0
    for i in range(len(voltage)):
        if i % int(len(voltage)/10) == 0 and print_messages == 'y':
            print('Calculating from voltage position {} out of {}.'.format(i, len(voltage)))
        if voltage_p[i] <= 0:
            u_1 = np.sqrt(np.abs(2 * const.e * voltage_p[i] / const.m_e))
            bulk_current[i], variances = integ.quad(maxwellian_integral, u_1, 1e8, args=(T_ec, n_ec, r_p, l_p, voltage_p[i]))
            secondary_current[i], variances = integ.dblquad(drifting_maxwellian_integral, u_1, 3e8, lambda x: 0, lambda x: np.pi, args=(T_eh, n_eh, u_D, r_p, l_p, voltage_p[i]))
        else:
            bulk_current[i], variances = integ.quad(maxwellian_integral, 0, 1e8, args=(T_ec, n_ec, r_p, l_p, voltage_p[i]))
            secondary_current[i], variances = integ.dblquad(drifting_maxwellian_integral, 0, 3e8, lambda x: 0, lambda x: np.pi, args=(T_eh, n_eh, u_D, r_p, l_p, voltage_p[i]))
    end_time = timeit.default_timer()
    print('Total calculation time took {} s'.format(end_time-start_time)) if print_messages == 'y' else 0
    ion_current = -I_i*1.15*(-eta_pre_c*voltage_p[voltage_p<=0])**0.5 #1.15, 1/2 = A, B from laframboise function set to T_i = 0
    total_current[voltage_p<=0] = bulk_current[voltage_p<=0] + secondary_current[voltage_p<=0] + ion_current
    total_current[voltage_p>0] = bulk_current[voltage_p>0] + secondary_current[voltage_p>0]
    V_f = voltage[find_nearest_index_in_array(total_current, 0)]

    if 'plot' in args:
        fig_cic, ax_cic = plt.subplots(nrows=2, ncols=2)

        ax_cic[0, 0].plot(voltage, bulk_current)
        ax_cic[0, 0].set_xlabel('Experimental Voltage (V)')
        ax_cic[0, 0].set_ylabel('Current (A)')
        ax_cic[0, 0].set_title('Bulk electron current T_ec = {}, n_ec = {:.1e}'.format(T_ec_ev, n_ec))
        ax_cic[0, 0].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[0, 0].axhline(y=0, color='grey', linestyle='--')

        ax_cic[0, 1].plot(voltage, secondary_current)
        ax_cic[0, 1].set_xlabel('Experimental Voltage (V)')
        ax_cic[0, 1].set_ylabel('Current (A)')
        ax_cic[0, 1].set_title('secondary electron current T_eh = {}, n_eh = {:.1e}, u_D/u_Th = {:.2f}'.format(T_eh_ev, n_eh, u_D/u_Th))
        ax_cic[0, 1].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[0, 1].axhline(y=0, color='grey', linestyle='--')

        ax_cic[1, 0].plot(ion_voltage, ion_current)
        ax_cic[1, 0].set_xlabel('Experimental Voltage (V)')
        ax_cic[1, 0].set_ylabel('Current (A)')
        ax_cic[1, 0].set_title('ion current, n_i = {:.1e}'.format(n_i))
        ax_cic[1, 0].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[1, 0].axhline(y=0, color='grey', linestyle='--')

        ax_cic[1, 1].plot(voltage, bulk_current, 'b--', label='bulk')
        ax_cic[1, 1].plot(voltage, secondary_current, 'r--', label='2ndry')
        ax_cic[1, 1].plot(ion_voltage, ion_current, 'g--', label='ion')
        ax_cic[1, 1].plot(voltage, total_current, 'k', label='total')
        ax_cic[1, 1].legend()
        ax_cic[1, 1].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[1, 1].axhline(y=0, color='grey', linestyle='--')
        ax_cic[1, 1].axvline(x=V_f, color = 'green', linestyle = '--')
        ax_cic[1, 1].set_title('All currents')
        ax_cic[1, 1].set_xlabel('Experimental voltage (V)')
        ax_cic[1, 1].set_ylabel('Current (A)')

        fig_cic.suptitle('Ideal i-V with V_p = {:.2f}, V_f = {:.2f}'.format(V_p, V_f))
        fig_cic.tight_layout()
        plt.show()

    return total_current, bulk_current, secondary_current, ion_current, voltage, ion_voltage


def create_ideal_characteristic(*args, n_ec = 1e15, n_eh=1e14, n_i=1e15, T_ec_ev=3, T_eh_ev=10, m_i=6.6335e-26, r_p=50.0e-6, l_p=15.0e-3, V_p=0, voltage_array = 0, min_voltage = -100, max_voltage = 100, num_steps = 1001):
    '''
    A function that creates an ideal two-temp maxwellian i-V curve based on the given inputs. To only have a single maxwellian, set n_ec or n_eh or n_i to 0.
    Uses Langmuir's integral function to calculate electrons and Laframboise theory with T_i = 0 for ions.
    All parameters set to keyword arguments for ease of reading
    :param n_ec: cold electron density m^-3
    :param n_eh: hot electron density m^-3
    :param n_i: ion density m^-3
    :param T_ec_ev: cold electron temp in eV
    :param T_eh_ev: hot electron temp in eV
    :param m_i = m_Ar: ion mass kg
    :param r_p: probe radius m
    :param l_p: probe length m
    :param V_p: plasma potential estimate
    :param args: skip|plot
    :keyword voltage_array = False: You can either input your actual voltage array being used, or the function can create a voltage array
    :keyword min_voltage = -100: min voltage for created voltage array
    :keyword max_voltage = 100:  max voltage for created voltage array
    :keyword num_steps = 1001: number of points to be used if creating voltage array
    :return total current, bulk current (only cold electrons), secondary_current (only hot electrons),
    :return: ion current, voltage (original, not changed to V_p = 0), ion_voltage (voltage array for ion current)
    '''
    if 'skip' in args:
        return 0, 0, 0, 0, 0, 0
    T_ec, T_eh = T_ec_ev*const.e/const.k, T_eh_ev*const.e/const.k
    voltage = np.linspace(min_voltage, max_voltage, num_steps) if len(voltage_array) <= 1 else voltage_array
    voltage_p = voltage - V_p
    ion_voltage = voltage[voltage<=V_p]
    #Creating complicated constants
    I_i = const.e*n_i*2*np.pi*r_p*l_p*np.sqrt(const.k*T_ec/(2*np.pi*m_i))
    eta_pre_c, eta_pre_h = const.e/(const.k*T_ec), const.e/(const.k*T_eh)
    bulk_current, secondary_current, total_current = np.zeros(len(voltage)), np.zeros(len(voltage)), np.zeros(len(voltage))
    for i in range(len(voltage)):
        if voltage_p[i] <= 0:
            u_0 = np.sqrt(np.abs(2*const.e*voltage_p[i]/const.m_e))
            bulk_current[i], variances = integ.quad(maxwellian_integral, u_0, 1e8, args = (T_ec, n_ec, r_p, l_p, voltage_p[i]))
            secondary_current[i], variances = integ.quad(maxwellian_integral, u_0, 1e8, args = (T_eh, n_eh, r_p, l_p, voltage_p[i]))
        else:
            bulk_current[i], variances = integ.quad(maxwellian_integral, 0, 1e8, args=(T_ec, n_ec, r_p, l_p, voltage_p[i]))
            secondary_current[i], variances = integ.quad(maxwellian_integral, 0, 1e8, args=(T_eh, n_eh, r_p, l_p, voltage_p[i]))
    ion_current = -I_i*1.15*(-eta_pre_c*voltage_p[voltage_p<=0])**0.5 #1.15, 1/2 = A, B from laframboise function set to T_i = 0
    total_current[voltage_p<=0] = bulk_current[voltage_p<=0] + secondary_current[voltage_p<=0] + ion_current
    total_current[voltage_p>0] = bulk_current[voltage_p>0] + secondary_current[voltage_p>0]
    V_f = voltage[find_nearest_index_in_array(total_current, 0)]

    if 'plot' in args:
        fig_cic, ax_cic = plt.subplots(nrows=2, ncols=2)

        ax_cic[0, 0].plot(voltage, bulk_current)
        ax_cic[0, 0].set_xlabel('Experimental Voltage (V)')
        ax_cic[0, 0].set_ylabel('Current (A)')
        ax_cic[0, 0].set_title('Bulk electron current T_ec = {}, n_ec = {:.1e}'.format(T_ec_ev, n_ec))
        ax_cic[0, 0].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[0, 0].axhline(y=0, color='grey', linestyle='--')

        ax_cic[0, 1].plot(voltage, secondary_current)
        ax_cic[0, 1].set_xlabel('Experimental Voltage (V)')
        ax_cic[0, 1].set_ylabel('Current (A)')
        ax_cic[0, 1].set_title('secondary electron current T_eh = {}, n_eh = {:.1e}'.format(T_eh_ev, n_eh))
        ax_cic[0, 1].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[0, 1].axhline(y=0, color='grey', linestyle='--')

        ax_cic[1, 0].plot(ion_voltage, ion_current)
        ax_cic[1, 0].set_xlabel('Experimental Voltage (V)')
        ax_cic[1, 0].set_ylabel('Current (A)')
        ax_cic[1, 0].set_title('ion current, n_i = {:.1e}'.format(n_i))
        ax_cic[1, 0].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[1, 0].axhline(y=0, color='grey', linestyle='--')

        ax_cic[1, 1].plot(voltage, bulk_current, 'b--', label='bulk')
        ax_cic[1, 1].plot(voltage, secondary_current, 'r--', label='2ndry')
        ax_cic[1, 1].plot(ion_voltage, ion_current, 'g--', label='ion')
        ax_cic[1, 1].plot(voltage, total_current, 'k', label='total')
        ax_cic[1, 1].legend()
        ax_cic[1, 1].axvline(x=V_p, color='grey', linestyle='--')
        ax_cic[1, 1].axhline(y=0, color='grey', linestyle='--')
        ax_cic[1, 1].axvline(x=V_f, color = 'green', linestyle = '--')
        ax_cic[1, 1].set_title('All currents')
        ax_cic[1, 1].set_xlabel('Experimental voltage (V)')
        ax_cic[1, 1].set_ylabel('Current (A)')

        fig_cic.suptitle('Ideal i-V with V_p = {:.2f}, V_f = {:.2f}'.format(V_p, V_f))
        fig_cic.tight_layout()
        plt.show()

    return total_current, bulk_current, secondary_current, ion_current, voltage, ion_voltage

def shift_array_by_value(array, value, plot_diff = False, print_message = 'n'):
    '''
    A function that shifts an array by a given value, so that array[i] ---> array[i]-value
    :param array:
    :param value:
    :keyword print_message = 'n': Prints a message stating the array and value variable names as well as the shifting value.
    :keyword plot_diff = False: plots the change in the variable
    :return: array - value (works with numpy array)
    '''
    print('\nShifting array {} by value {} named {}'.format(retrieve_variable_name(array), value, retrieve_variable_name(value))) if print_message == 'y' else 0
    if plot_diff:
        plt.plot(array, label = retrieve_variable_name(array))
        plt.plot(array-value, 'r', label = '{} - {:.2e}'.format(retrieve_variable_name(array), value))
        plt.legend()
        plt.title('Comparison of array before (b) and after shifting (r)')
        plt.show()
        plt.close()
    return array - value

def least_squares_between_datasets(first_array, second_array):
    '''
    Computes the square sum of differences between two datasets
    :param first_array:
    :param second_array:
    :return: L_S
    '''
    L_S = 0
    for i in range(len(first_array)):
        L_S += (first_array[i] - second_array[i])**2
    return L_S

def probability_density_of_result(value, standard_dev, mean, scale_prob_value = 'n', return_log = 'n'):
    '''
    Function that gets you the log10 of the probability of getting the observed result from a value predicted by model parameters (mean)
    and a known variance. Returns the log10 of the probability to avoid
    :param value: The observed value (so the dataset value with noise)
    :param standard_dev:
    :param mean: The predicted, ideal, theoretical value.
    :keyword scale_prob_value='n' (y|n): If y, sets the front parameter of the probability density calculation to 1, to avoid overflow/underflow errors
    :return: probability density value.
    '''

    front_value = 1.0/(np.sqrt(2*np.pi)*standard_dev) if scale_prob_value == 'n' else 1.0
    if return_log == 'y':
        return np.log10(front_value) + -(1.0/(2.0*standard_dev**2))*(value-mean)**2
    else:
        return front_value * np.exp(-(1.0/(2.0*standard_dev**2))*(value-mean)**2)

def total_combined_probability_density(value_array, standard_dev, mean_array, scale_prob_value = 'n', log_vals = 'n', exclude_outliers = 'n', exclude_outlier_percentage = 0.05):
    '''
    Function that compares a predicted signal from various inputs and calculates the likelihood of returning that input
    given a certain variance and the actual raw data. Assumes gaussian noise.
    For this, it's important that the value array is the signal with noise, and the mean array is the predicted values
    from the model with guessed parameters.
    :param value_array: the raw data
    :param standard_dev: The variance (sigma squared) in the dataset
    :param mean_array: the array of predicted values from your model, given the inputs.
    :keyword scale_prob_value='n' (y|n): If y, sets the front parameter of the probability denisty calculation to 1, to avoid overflow/underflow errors
    :return: total probability density from multiplication of results across entire dataset. Not output as 1, must be normalised.
    '''
    total_outliers = 0
    total_probability_val = 1.0 if log_vals == 'n' else 0
    for i in range(len(value_array)):
        if i == 0:
            new_val = probability_density_of_result(value_array[0], standard_dev, mean_array[0], scale_prob_value=scale_prob_value, return_log=log_vals)
        else:
            if log_vals == 'y':
                new_val = probability_density_of_result(value_array[i], standard_dev, mean_array[i], scale_prob_value=scale_prob_value, return_log=log_vals)
            else:
                new_val = probability_density_of_result(value_array[i], standard_dev, mean_array[i], scale_prob_value=scale_prob_value, return_log=log_vals)

        if exclude_outliers == 'y':
            if log_vals == 'y':
                if new_val <= -10:
                    new_val = 0
                    total_outliers+=1
            else:
                if new_val <= 1e-10:
                    new_val = 1
                    total_outliers += 1
        if log_vals == 'n':
            total_probability_val *= new_val
        else:
            total_probability_val += new_val
    if total_outliers >= exclude_outlier_percentage*len(value_array):
        total_probability_val = 0
        print('\nExceeded {} % outliers ({} outliers out of {} values), probability set to 0'.format(exclude_outlier_percentage, total_outliers, len(value_array)))
    return total_probability_val

def subscript(x):
    '''
    Makes x into subscript. Has most subscript characters
    :param x:
    :return: subscript(x)
    '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

def superscript(x):
    '''
    Makes x into superscript. Has most superscript characters.
    :param x:
    :return:
    '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def extrapolate_dataset_x_power_n(x_axis, y_axis, final_x_axis, direction = 'R', extrapolate_from = -10, power_n = 1, interpolate = False, interp_axis = None, zero_limit = False, force_end_fit = False):
    '''
    Function that extrapolates beyond or before a given dataset using a power of x function.
    :param x_axis: The x axis of the data (say energy)
    :param y_axis: The original data set
    :param final_x_axis: The final x axis of the data that you want to have after the extrapolation
    :keyword direction = 'R' ('R/L'): Whether to extrapolate to the Right or Left of the dataset. If 'L', extrapolate from must be changed
    :keyword extrapolate_from: The point from which (up to which for 'L') the data extrapolation is based on.
    :keyword power_n: The power of n in y = m*x**n + C for the fit
    :keyword interpolate = False (True/False): If True, return an interpolation for the y axis. This requires an interpolation axis to be entered,
    which will otherwise be set to the original x_axis
    :keyword interp_axis = None (array): The x axis for the interpolation to be used, if you don't want it in the original x_axis. If interpolate = True, interp_axis = None,
    then you get the final energy axis as the interpolation axis.
    :keyword zero_limit = False (True/False): If True, values are not allowed to go below 0
    :keyword force_end_fit = False: Whether to make sure that the first point of the extrapolation extends on smoothly from the last point of the data
    :return: Extrapolated_x_axis, Extrapolated_y_axis if interpolate = False or interp_axis = None, else Extrapolated_x_axis, interp_x, interpolated_y_axis
    '''
    y_data = y_axis[extrapolate_from:] if direction == 'R' else y_axis[:extrapolate_from]
    x_data = x_axis[extrapolate_from:] if direction == 'R' else x_axis[:extrapolate_from]
    function = lambda x, m, C: m*x**power_n + C

    line_params, nerds = scipy.optimize.curve_fit(function, x_data, y_data)


    y_extra = function(final_x_axis[len(x_axis):], line_params[0], line_params[1]) if direction == 'R' else function(final_x_axis[:len(final_x_axis)-len(x_axis)], line_params[0], line_params[1])

    if force_end_fit:
        if function(x_data[-1], line_params[0], line_params[1]) != y_data[-1]:
            diff = function(x_data[-1], line_params[0], line_params[1]) - y_data[-1]
            y_extra -= diff

    final_y_axis = np.concatenate((y_axis, y_extra)) if direction == 'R' else np.concatenate((y_extra, y_axis))
    if zero_limit:
        final_y_axis[final_y_axis<0] = 0
    if interpolate:
        if interp_axis is None:
            interpolate_axis = final_x_axis
        else:
            interpolate_axis = interp_axis

    final_y_axis = scipy.interpolate.interp1d(interpolate_axis, final_y_axis) if interpolate else final_y_axis

    if interpolate:
        if interp_axis is None:
            return final_x_axis, final_y_axis
        else:
            return  final_x_axis, interpolate_axis, final_y_axis
    else:
        return final_x_axis, final_y_axis

def rungekutta4(f, y_0, t, args=()):
    '''
    A RK 4 solver for initial value problems. f is the RHS of the function you are solving, which must be related to a first order
    differential in t only.     For example, if  x''(t) + x'(t) + x(t) = 0, let z = x'(t). Then, z'(t) + z(t) + x(t) = 0. So,
    x(t) = -(z'(t) + z(t)), z'(t) = -(z(t) + x(t)). Then, y = [z(t), x(t)]. Now, y'(t) = [z'(t), x'(t)] = [-z(t) - x(t), z(t)].
    y_0 would be your initial values for y, i.e y(0) = [z(0), x(0)].
    :param f: The RHS of the first order ODE in t.
    :param y_0: The initial guesses of each variable. This should always be an array, even if we only have one variable.
    :param t: the time array you are solving over. This should be t = np.linspace(start_time, end_time, num_steps). t[i+1]-t[i] = h, the step size
    :param args: any constants that you have to input to the function
    :return: y at time t
    '''
    n = len(t)
    y = np.zeros((n, len(y_0)))
    y[0] = y_0
    for i in range(n-1):
        h = t[i+1]-t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1*h/2.0, t[i] + h/2.0, *args)
        k3 = f(y[i] + k2*h/2.0, t[i] + h/2.0, *args)
        k4 = f(y[i] + k3*h, t[i] + h, *args)
        y[i+1] = y[i] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y


def rungekutta4_single_time_step(f, y_0, t_0, h, args=(), test = False):
    '''
    The RK solver but for a single time step. This is if you need to update values based on another variable at each time
    step. The example I have is that I want to solve the rate of change of n, but some of the terms in the equation have d/dx.
    However, the thing doesn't evolve with x. The d/dx is just a simple calculation based on the current form of n. So, at each
    time step, i'll calculate a new n based on the old n and its derivatives, and then use this new n to calculate new derives
    and calculate the next time step.
    :param f: The RHS of the first order ODE in t.
    :param y_0: The initial guesses of each variable. This should always be an array, even if we only have one variable.
    :param t_0: The initial time of the solver. We have this so we can record the development over time.
    :param h: The time step value.
    :param args: The arguments that the function needs to operate. if y' = 3y+2y**2, the args = (3,2)
    :return: y(t+h)
    '''
    if test == True:
        print('\nprinting y_0')
        print(y_0)
    y = np.zeros((2, len(y_0)))
    if test == True:
        print('\ny before adding initial conds')
        print(y)
    y[0] = y_0
    if test == True:
        print('\ny after adding initial conds')
        print(y)
    k1 = f(y[0], t_0, *args)
    if test == True:
        print('\nPrinting k1')
        print(k1)
        print('\nPrinting k1[0]')
        print(k1[0])
    k2 = f(y[0] + k1[0]*h/2.0, t_0+h/2.0, *args)
    k3 = f(y[0] + k2[0]*h/2.0, t_0 + h/2.0, *args)
    k4 = f(y[0] + k3[0]*h, t_0 + h, *args)
    y[1] = y[0] + (h/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    if test == True:
        print('\nPrinting final result inside RK4 function')
        print(y)
    return y

def check_if_string_condition_met(var, options = ['y', 'n']):
    '''
    Short function that checks if a string option required by a function has the correct input, and allows the user to
    correct if not. Only valid for strings (y/n or R/L)
    :param var: The string value given
    :keyword options = ['y', 'n']: The possible string values to input, as separate strings in a list.
    :return: Correct input (one of the strings in option)
    '''
    while True:
        if var in options:
            return var
        else:
            var = str(input('\nWARNGING! Incorrect input given. Please input one of the following options: {}'.format(options)))


def dist_travelled_by_charged_particle_in_E_field(E_field, q, m, initial_velocity, time_step, return_final_velocity = True, max_dist = None, max_speed = None):
    '''
    Calculates the distance travelled within a time step by a particle with charge q with an initial velocity in an electric field (assumed to be of constant value over the distance).
    It can also return the final velocity after the timestep.
    Uses x_f = x_i (0) + v_i*dt + 0.5*a*dt, where a = E*q/m
    :param E_field: The electric field value at that point
    :param q: The charge of the particle (signed)
    :param m: THe mass of the particle
    :param initial_velocity: The initial velocity of the particle
    :param time_step: The time step involved
    :keyword return_final_velocity = True: Whether to return the final velocity after the time step or not
    :keyword max_dist = None: If a value is given, this is the maximum distance the particle can travel
    :keyword max_speed = None: If a max value is put in, it caps the final maximum velocity by the maximum speed
    :return: Distance travelled, final velocity if return_final_velocity = True else Distance travelled
    '''
    a = E_field*q/m
    dist_travelled = initial_velocity*time_step + 0.5*a*time_step**2
    if max_dist is not None:
        if dist_travelled > np.abs(max_dist):
            dist_travelled = max_dist
        elif -np.abs(dist_travelled) < -np.abs(max_dist):
            dist_travelled = -max_dist
    final_vel = initial_velocity + a*time_step
    if max_speed is not None:
        if abs(final_vel) >= abs(max_speed):
            if final_vel <0:
                final_vel = -np.abs(max_speed)
            else:
                final_vel = np.abs(max_speed)
    if return_final_velocity:
        return dist_travelled, final_vel
    else:
        return dist_travelled

def density_change_from_consv_of_flux_no_collisions(initial_density, initial_velocity, final_velocity):
    '''
    Calculates n_f = n_i*v_i/v_old
    :param initial_density:
    :param initial_velocity:
    :param final_velocity:
    :return: np.abs(initial_density*initial_velocity/final_velocity)
    '''
    return np.abs(initial_density*initial_velocity/final_velocity)
