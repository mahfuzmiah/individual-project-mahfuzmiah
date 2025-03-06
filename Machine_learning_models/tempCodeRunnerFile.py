    wmape_by_series = results.groupby('unique_id')[['y', 'SeasonalNaive']].apply(
        lambda x: wmape(x['y'], x['SeasonalNaive'])
    )