# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# prepare_data.py
#
# Identification: /prepare_data.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//

def prepare_data(df):
    variables = []
    features = []
    easys = []

    # Process row by row in df
    for index, row in df.iterrows():
        embedding = row['embeddings']
        label = row['sentiment']
        
        # Prepare variables
        feature_set = {i: [1.0, float(val)] for i, val in enumerate(embedding)}
        variable = {
            'var_id': index,
            'is_easy': True,
            'is_evidence': True,
            'true_label': label,
            'label': label,
            'feature_set': feature_set
        }
        variables.append(variable)

        # Prepare features
        for i, val in enumerate(embedding):
            feature = {
                'feature_id': i,
                'feature_type': 'unary_feature',
                'feature_name': f'embedding_{i}',
                'weight': {index: [1.0, float(val)]}
            }
            features.append(feature)

        # Prepare the easy instance
        easy = {
            'var_id': index,
            'label': label
        }
        easys.append(easy)
    return variables, features, easys

if __name__ == "__main__":
    import read_data
    import tokenize_data

    df = read_data.read_data('../data/Sentiment_V2.xlsx')
    df = tokenize_data.tokenize_and_embed(df)
    variables, features, easys = prepare_data(df)
    print(variables[:2])
    print(features[:2])
    print(easys[:2])
    # print(df.head())
