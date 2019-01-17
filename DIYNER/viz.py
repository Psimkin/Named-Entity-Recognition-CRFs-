import pandas as pd

def radialTree(data, entity_to_cat, root='', text_col='text', value_col='value'):

    results = {}

    # string-matching lookup with minimum character threshold
    for text in data[text_col]:
        results[text] = []
        for entity in list(entity_to_cat.keys()):
            if entity > 5 and entity.lower() in text.lower():
                results[text].append(entity)
            else:
                if entity.lower() in text.lower():
                    results[text].append(entity)

    # unique matches & join back on original dataframe
    unique_entities = set(x for l in list(results.values()) for x in l)
    data['entities'] = data[text_col].map(results)

    value_dict = {}

    # find mean values associated with each entity
    for entity in unique_entities:
        mean_val = data[data['entities'].str.contains(entity, regex=False)][value_col].mean()
        value_dict[entity] = mean_val

    # d3.js formatting
    top_layer = [root + "." + x for x in set(list(entity_to_cat.values()))]
    mid_layer = [root + "." + entity_to_cat[x] + "." + x for x in value_dict.keys()]
    ids = [root] + top_layer + mid_layer

    # final csv preparation
    output = pd.DataFrame(ids, columns=['id'])
    output['index'] = output['id'].apply(lambda x: x.split('.')[2] if len(x.split('.')) > 2 else 0)
    output['value'] = output['index'].map(value_dict)
    output.fillna(0, axis=1, inplace=True)

    return output