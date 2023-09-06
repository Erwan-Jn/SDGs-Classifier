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


 pipe_model = Pipeline([
    ('tf_idf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 2), max_df=0.8, norm="l2")),
    ('model', LogisticRegression(C = .9,
        multi_class = 'multinomial',
        class_weight = 'balanced',
        random_state = 42,
        solver = 'newton-cg',
        max_iter = 100))
    ])
