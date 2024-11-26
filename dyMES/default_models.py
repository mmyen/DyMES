#Default ecological community transition function

def eco_transition_function(n, N, params):
    return params['r0'] * n - params['d0'] * n**2 * N/(params['Nc'])

eco_params = {'r0':0.2, 'd0':0.010011, 'S':10, 'Nc':95}

#TODO: Add remaining 3 functions from paper