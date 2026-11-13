"""
Operador generado automaticamente a partir del fenotipo:

    order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,y,move_right(order_based(fill_first_occurring,preserve_elements,x,map_list(order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,reverse(order_based(fill_first_occurring,preserve_elements,x,reverse(y)))))))),order_based(fill_first_occurring,preserve_elements,x,y)))))))))

Aridad: 2. Generado por mohh.analysis.phenotype.
"""

import numpy as np

def order_based(fill_method, preserve_method, x, y):
    """
    fill_method is how to rebuild (chosed by the grammar)
    preserve_method selects segments or elements (chosed by the grammar)
    x is a terminal
    y is a terminal
    """

    # Make sure x is one list
    x = verify_inputs(x)

    # Make sure y is one list
    y = verify_inputs(y)

    # First, select the elements to preserve
    preserved_data_x = preserve_method(x)
    preserved_data_y = preserve_method(y) 

    # Fill the empty slots with the second parent
    new_x: np.ndarray = fill_method(preserved_data_x, y)
    new_y: np.ndarray = fill_method(preserved_data_y, x)

    return new_x, new_y


def fill_first_occurring(preserved, filler):
    """
    Fill preserved with filler
    """

    # Make sure x is one list
    preserved = verify_inputs(preserved)

    # Make sure y is one list
    filler = verify_inputs(filler)
  
    # Find the elements of preserved
    existing_values = set(preserved[preserved != -1])

    j = 0
    for i in range(len(preserved)):
        # Note:
        # x/y can come from arbitrarily nested grammar expressions, so filler is not always
        # a clean permutation so, if it runs out, we need to leave the rest as -1 and let the
        # existing is_permutation/test_individual checks to reject the result.
        while preserved[i] == -1 and j < len(filler):

            if filler[j] not in existing_values:
                preserved[i] = filler[j]
            j += 1

    return preserved


def preserve_elements(x):
    """
    Select some elements.
    """

    # Make sure x is one list
    x = verify_inputs(x)

    # Select k elements
    k = np.random.randint(1, len(x))

    # Convert to numpy if not already
    x = np.array(x)

    # Select k indices without replacements
    selected = np.random.choice(len(x), size=k, replace=False)

    # Create result list
    result = -np.ones_like(x)
    result[selected] = x[selected]

    return result


def preserve_segments(x):
    """
    Select segments
    """

    # Make sure x is one list
    x = verify_inputs(x)

    # Select two random indices
    i = 0
    j = len(x) - 1
    while (i == 0 and j == len(x) - 1) or (i == j):
        selected = np.sort(np.random.choice(len(x), size=2, replace=False))
        i, j = selected
        j -= 1

    result = -np.ones_like(x)
    result[i:j] = x[i:j]

    return result


def move_right(x):
    """
    Move all elements to the right.
    """
    # Make sure x is one list
    x = verify_inputs(x)

    # Rotate the array
    return np.concatenate(([x[-1]], x[:-1]))


def map_list(x, y):

    # Make sure x is one list
    x = verify_inputs(x)
    
    # Make sure y is one list
    y = verify_inputs(y)

     # Ensure y contains valid indices
    if not np.all((y >= 0) & (y < len(x))):
        raise ValueError("All elements of y must be valid indices for x and y.")

    if not np.all((x >= 0) & (x < len(y))):
        raise ValueError("All elements of x must be valid indices for x and y.")
    
    new_x = x[y]
    new_y = y[x]

    return new_x, new_y


def reverse(x):
    """
    Reverse the elements.
    """
    # Make sure x is one list
    x = verify_inputs(x)

    return x[::-1]


def collapse(x, y=None):
    return x


def verify_inputs(z):
    if isinstance(z, tuple):
        z = collapse(z[0], z[1])
    return z


def crossover(x, y):
    """
    Aplica el operador a (x, y) y devuelve el hijo.
    """
    x = verify_inputs(x)
    y = verify_inputs(y)
    order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,y,move_right(order_based(fill_first_occurring,preserve_elements,x,map_list(order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,reverse(order_based(fill_first_occurring,preserve_elements,x,reverse(y)))))))),order_based(fill_first_occurring,preserve_elements,x,y)))))))))

    return order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,y,move_right(order_based(fill_first_occurring,preserve_elements,x,map_list(order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,reverse(order_based(fill_first_occurring,preserve_elements,x,reverse(y)))))))),order_based(fill_first_occurring,preserve_elements,x,y)))))))))


if __name__ == "__main__":
    p1 = np.arange(10)
    p2 = np.arange(10)[::-1].copy()
    print(crossover(p1, p2))
