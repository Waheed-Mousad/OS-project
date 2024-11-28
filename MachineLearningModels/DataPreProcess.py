import pandas as pd

# Function to preprocess the data by mapping catagorical columns and normalizing numeric columns
def data_preprocess(df, target_col=[], samples=0):
    if len(target_col) != 0:
        df = df[target_col]
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Remove rows where any numeric column contains 0 or negative values
    df = df[(df[numeric_cols] > 0).all(axis=1)]
    # map the catagorical column and save the mapping
    catagorical_cols = df.select_dtypes(include=['object']).columns
    catagorical_mapping = {}
    for col in catagorical_cols:
        df[col], catagorical_mapping[col] = pd.factorize(df[col])
    # Normalize the numeric columns and save min-max values
    numeric_mapping = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
        numeric_mapping[col] = (min_val, max_val)
    # Save the mapping
    mapping = {'catagorical': catagorical_mapping, 'numeric': numeric_mapping}
    # Sample the data
    if samples != 0:
        df = df.sample(n=samples, random_state=42)
    return df, mapping

# Function to visualize the data
def data_visualization(df, target_col=""):
    # Visualize the data
    import matplotlib.pyplot as plt
    df.hist(bins=50, figsize=(20, 15))
    plt.show()
    # show all columns vs target column
    if target_col != "":
        try:
            for col in df.columns:
                if col != target_col:
                    df.plot(kind='scatter', x=col, y=target_col)
                    plt.show()
        except: # if the target column is not found
            print("Target column not found")


# a function that denomrlized the data using the mapping
def denormlize_data(df, mapping):
    # Denormalize numeric columns
    for col in mapping['numeric']:
        df[col] = df[col] * (mapping['numeric'][col][1] - mapping['numeric'][col][0]) + mapping['numeric'][col][0]
    # Denormalize categorical columns
    for col in mapping['catagorical']:
        # Reverse the mapping, turn into integer before mapping
        df[col] = df[col].apply(lambda x: int(x))
        df[col] = df[col].apply(lambda x: mapping['catagorical'][col][x])
    return df




if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('processes_datasets.csv')
    # Preprocess the data
    df , mapping = data_preprocess(df, target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
], samples=100)
    # Visualize the data
    data_visualization(df, 'RunTime ')
    print(df)
    print(mapping)
    df.to_csv('Modified_Process_dataset.csv', index=False)
    # Denormalize the data
    df = denormlize_data(df, mapping)
    print(df)