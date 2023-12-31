test
pipe_model = Pipeline([
        ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2))),
        ('dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ('model', HistGradientBoostingClassifier(max_depth=2, l2_regularization=0.2, verbose=2, max_iter=20))
        ])

pipe_model = Pipeline([
    ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2), max_df=0.8, norm="l2")),
    ('model',     LogisticRegression(multi_class='multinomial', random_state=1,
                                            max_iter=1000, penalty="l2",
                                            solver="newton-cg"))

    ])

 pipe_model = Pipeline([
        ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2), max_df=0.8, norm="l2")),
        ('model', VotingClassifier(estimators=models, voting='soft', weights=[2, 1, 1]))
        ])


FunctionTransformer
def to_arr(x):
    return x.toarray()

models = list()
models.append(("clf1", LogisticRegression(multi_class='multinomial', random_state=1,
                                        max_iter=1000, penalty="l2",
                                        solver="newton-cg")))
models.append(("clf2", KNeighborsClassifier(n_neighbors=10)))
models.append(("clf3", GaussianNB()))
models.append(("clf4", DecisionTreeClassifier(max_depth=10)))

pipe_model = Pipeline([
    ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2), max_df=0.8, norm="l2")),
    ('dense', FunctionTransformer(to_arr, accept_sparse=True)),
    ('model', VotingClassifier(estimators=models, voting='soft', weights=[3, 1, 2, 1]))
    ])

    models = list()
    models.append(("clf1", LogisticRegression(multi_class='multinomial', random_state=1,
                                        max_iter=1000, penalty="l2",
                                        solver="newton-cg")))
    models.append(("clf2", KNeighborsClassifier(n_neighbors=20)))
    models.append(("clf3", GaussianNB()))


 pipe_model = Pipeline([
    ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2), max_df=0.8, norm="l2")),
    ('model', LogisticRegression(C = .9,
        multi_class = 'multinomial',
        class_weight = 'balanced',
        random_state = 42,
        solver = 'newton-cg',
        max_iter = 100))
    ])


def balancing_data(df:pd.DataFrame):
    med = int(np.median(df["sdg"].value_counts().values))
    temp_ = pd.DataFrame(columns=df.columns)
    for i in range(1, 17):
        if len(df.loc[df["sdg"]==i])>med:
            temp_ = pd.concat([temp_, df.loc[df["sdg"]==i].sample(n=med, random_state=42)], axis=0)
        else:
            ratio = med/len(df.loc[df["sdg"]==i])
            temp_ = pd.concat([temp_, df.loc[df["sdg"]==i].sample(frac=ratio, replace=True, random_state=42)])
    return temp_
