# pandas

1. link
   * [official site](https://pandas.pydata.org/)
   * [documentation](http://pandas.pydata.org/pandas-docs/stable/)
   * [getting started](http://pandas.pydata.org/pandas-docs/stable/overview.html)
   * [莫烦Python](https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/)
   * [十分钟的 pandas 入门教程](https://ericfu.me/10-minutes-to-pandas/)
   * [十分钟快速入门 Pandas](https://zhuanlan.zhihu.com/p/21933466)
2. support data format: `.csv`, `.xlsx`, `sql database`, `.hdf5`
3. pandas: panel data

## indexing

1. [indexing and selecting data](http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing)
2. three types of multi-axis indexing: `[]`, `.loc`, `.iloc`
3. `[]` basic indexing
   * `Series[label]` output scalar value
   * `DataFrame[colname]` output Series
4. `.loc` label based, raise `KeyError` when the items are not found
   * a single label: `5`, `'a'`
   * a list or array: `['a','b','c']`
   * slice object: `'a':'f'`(`slice('a','f')`), both start and stop are included
   * boolean array
   * callable function with one argument
5. `.iloc` integer position based, raise `IndexError` if a requested indexer is out-of-bounds (except slice indexers which allow out-of-bounds indexing)
   * an integer: `5`
   * a list or array: `[2,3,3]`
   * slice object: `2:3` (stop is excluded)
   * callable function with one argument

## misc00

1. create
   * `pd.Series()`
   * `pd.DataFrame()`
   * `pd.date_range()`
   * `pd.Timestamp()`
   * `pd.Categorical()`
2. property
   * `dtypes`
   * `index`, `columns`, `A`, `B`
   * `values`
   * `T`
3. method
   * `head()`
   * `tail()`
   * `describe()`
   * `sort_index()`
   * `sort_values()`
   * `copy()`
   * `df1.iloc[3]`
   * `df1.iloc[3:5,0:2]`
4. setting: `df1['F'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))`

## selection

1. recommand: `at`, `iat`, `loc`, `iloc`, `ix`
2. Python / Numpy expressions for selecting and setting: supported, but not efficient
3. property: `df1.A`
4. scalar
   * `df1.at[dates[0],'A']`
   * `df1.iat[0,1]`
5. by label
   * `df1.loc[dates[0]]`
   * `df1.loc[:,'A']`
   * `df1.loc['20130102':'20130104',['A','B']]`, include boundary
6. by position
   * `df1.iloc[3]`
   * `df1.iloc[3:5, 0:2]`
   * `df1.iloc[[1,2,4],[0,2]]`
7. boolean indexing
   * `df1[df1.A>0]`, `df1[df1>0]`
   * `df2[df2['E'].isin(['train'])]`
