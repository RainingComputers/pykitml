class InvalidFeatureType(Exception):
    '''
    Raised when specified feature type is invalid for the model.
    '''
    pass

class InvalidDistributionType(Exception):
    '''
    Raised when specified distribution type is invalid for the model.
    '''
    pass

def _valid_list(input_list, valid_items):
    '''
    Used to check if items in a list are valid.

    Parameters
    ----------
    input_list : list
        The list to check/validate.
    valid_items : list
        List of valid items the list can contain.
    '''
    if (all(item in valid_items for item in input_list) and len(input_list)>0):
        return True
    else:
        return False
        