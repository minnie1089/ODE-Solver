import abc

class Variable(metaclass=abc.ABCMeta):
    def __init__(self, value):
        self.__value = value

    @abc.abstractmethod
    def time_derivative(self, params, consts, variables) -> int:
        pass

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, val):
        self.__value = val


def euler(params, consts, variables):
    """
    Numerically integrates a system of ODEs using Euler's method

    :param params: A dict containing step_size and domain
    :param consts: A dict containing control variables
    :param variables: A dict containing independent variables
    :return: A dict of the independent variable's values at each time step
    """

    output = {}

    deltas = {}

    for key in variables:
        output[key] = [variables[key].value]

    t = 0
    while t < params["domain"]:
        for key in variables:
            deltas[key] = variables[key].time_derivative(params, consts, variables)

        for key in variables:
            variables[key].value += deltas[key] * params["step_size"]
            output[key].append(variables[key].value)

        t += params["step_size"]


    return output


def rk4(params, consts, variables):
    """
    Numerically integrates a system of ODEs using the classic Runge-Kutta method

    :param params: A dict containing step_size and domain
    :param consts: A dict containing control variables
    :param variables: A dict containing independent variables
    :return: A dict of the independent variable's values at each time step
    """

    output = {}

    deltas = {}

    k_values = [
        {},
        {},
        {},
        {},
    ]

    k_variables = [
        {},
        {},
        {},
        {},
    ]

    for key in variables:
        output[key] = [variables[key].value]

    t = 0
    while t < params["domain"]:
        for key in variables:
            k_values[0][key] = variables[key].time_derivative(params, consts, variables)

        for key in variables:
            k_variables[1][key] = type(variables[key])(variables[key].value + k_values[0][key] * params["step_size"] / 2)
        for key in variables:
            k_values[1][key] = variables[key].time_derivative(params, consts, k_variables[1])

        for key in variables:
            k_variables[2][key] = type(variables[key])(variables[key].value + k_values[1][key] * params["step_size"] / 2)
        for key in variables:
            k_values[2][key] = variables[key].time_derivative(params, consts, k_variables[2])

        for key in variables:
            k_variables[3][key] = type(variables[key])(variables[key].value + k_values[2][key] * params["step_size"])
        for key in variables:
            k_values[3][key] = variables[key].time_derivative(params, consts, k_variables[3])

        for key in variables:
            deltas[key] = params["step_size"] / 6 * (k_values[0][key] + 2 * k_values[1][key] + 2 * k_values[2][key] + k_values[3][key])

        for key in variables:
            variables[key].value += deltas[key]

            output[key].append(variables[key].value)

        t += params["step_size"]


    return output
