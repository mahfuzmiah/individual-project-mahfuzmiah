def impute_polynomial_interpolation(filepath):
    df = pd.read_csv(filepath)
    # Looks at each line by line and fills the missing values, each line will need a different equation to be filled in
    # will work out which order works best for what line, as O and Q will probably be ok with 1 or 2 but F and U and will higher ones possibly
    df_filled = df.copy()
    print(df_filled.shape())
