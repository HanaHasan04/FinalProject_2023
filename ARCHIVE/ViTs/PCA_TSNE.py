import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
create a new Excel file with reduced dimensions.
"""

data = pd.read_excel('input_file.xlsx')

feature_columns = [col for col in data.columns if col.startswith('Feature_')]
features = data[feature_columns]

# number of components.
N_COMPONENTS = 100

# PCA.
pca = PCA(n_components=N_COMPONENTS)  
reduced_features = pca.fit_transform(features)

# # t-SNE.
# tsne = TSNE(n_components=N_COMPONENTS)  
# reduced_features = tsne.fit_transform(features)

# Excel with new fetures.
new_data = pd.DataFrame(reduced_features, columns=[f'Component {i+1}' for i in range(N_COMPONENTS)])
other_columns = ['Emotion', 'Image', 'Video', 'Horse', 'Frame']
for col in other_columns:
    new_data[col] = data[col]
new_data.to_excel('reduced.xlsx', index=False)

# PCA.
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'o-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio')
plt.show()
