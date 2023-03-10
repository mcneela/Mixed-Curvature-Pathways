import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('unique_best_spaces_3.tsv', sep='\t')
df.dropna(inplace=True)

X = df[df.columns.difference(['% Dist Diff From Euclidean'])].values
y = df['% Dist Diff From Euclidean'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))

new_X = df[df.columns.difference(['H Copies', 'H Dim', 'E Copies', 'E Dim', 'S Copies', 'S Dim', '% Dist Diff From Euclidean', 'Dist Range', 'Dist Diff from Mean', 'Dist Diff from Next Best', 'Best Euclidean Distortion', 'Best Overall Distortion'])].values
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=21)
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))

# svm = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# svm = SVR(C=1.0, epsilon=0.2)
# svm.fit(X, y)
# print(svm.score(X, y))