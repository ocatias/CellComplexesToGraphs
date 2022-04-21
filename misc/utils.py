def list_of_dictionary_to_dictionary_of_lists(ls):
    return {k: [dic[k] for dic in ls] for k in ls[0]}

def dictionary_of_lists_to_list_of_dictionaries(dict):
    return [dict(zip(dict,t)) for t in zip(*dict.values())]
