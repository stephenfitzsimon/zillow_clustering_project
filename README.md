# Zillow Logerror Prediction

## Key Takeaways



## Contents

*Hyperlinks will only work on locally stored copies of this Jupyter Notebook*

1. <a href='#intro'>Introduction</a>
2. <a href='#wrangle'>Wrangle</a>

## Introduction <a name='intro'></a>

Zillow data was pulled from the database, and analyzed to determine which features can best determine the logerror of the prediction model. This project is divided into three parts: wrangle, explore, and model. Wrangle explains how the data is acquired and prepared for analysis and processing. Explore looks at the data and applies visualization and hypothesis testing to discover drivers of the logerror. Finally, model builds a model to predict logerror from the data. Each section and select subsections include a short list of key takeaways; these are followed by a discussion detailing the analysis choices made that are relevant to that (sub)section.

### Goals
- Use clustering algorithms to help determine predictors of logerror to help improve the performance of a property value model
- Using drivers of logerror to improve a model of property values
- Improve understanding of logerror to better inform the use of models for property prediction

### Project Plan
- Explore at least four variables of property prices
- Visualize variables
- Hypothesis test at least two variables
- Write this final report
- Python scripts that allow for the project to be reproducible

<a href='#contents'>Back to contents</a>

## Wrangle <a name='wrangle'></a>

### Key Wrangle Takeaways
- $0.66$ of the data is retained, the majority of the dropped data ($0.32$) is from focusing on single unit properties
- Final dataframe has 28 columns and 51606 rows
- There is no missing data in the target `logerror` column

#### Key Acqure Takeaways
- Data is acquired via `wrangle_zillow.get_zillow_data()`
- Original dataframe is 68 columns and 77414 rows
- There is significant missing data, but none in the `logerror` column

#### Key Prepare Takeaways
- 28 columns and 51606 rows are retained representing $0.66$ of original data
- The majority of the loss is represented by droping non-single unit properties

#### Discussion

Data is acquired via the `wrangle_zillow.get_zillow_data()` function. This function will query the SQL database, unless there is a saved .csv file present in the current directory. The name of the file is set by the `wrangle_zillow.FILENAME` constant. A database call can be force with the `query_db` parameter passed as `True`.

Consider the columns that have greater than $0.01$ of the rows having null values. Because there is significant data missing, all of these columns will be dropped.  Note, however, that some of these columns do have the majority of their data.  These columns could be used in a future analysis, see the <a href='#conclusion'>conclusion</a> for more discussion. 

This project is interested in single unit homes.  The column `propertylandusedesc` is filtered so that there is only the following values: `Single Family Residential`,`Mobile Hom`,`Manufactured, Modular, Prefabricated Home`,`Residential Genera`, and `Townhouse`.  Note that this is where the majority of the data is lost, as these rows represent 52495 of the original dataframe of 77414 rows (representing a loss of $0.32 = \frac{77414-52495}{77414}$.  There is a `unitcnt` column; however, it contains a significant number of nulls that are concentrated in Ventura and Orange counties.

The `fips` and `latitude`/`longitude` columns data are changed/corrected. The `fips` column is mapped to the strings `Los Angeles`, `Orange` and `Ventura`; it is then renamed `county`.  This aides in human readability. `latitude` and `longitude` are multiplied by $10^{-6}$, so that they represent the correct values for the Los Angeles metro area.

This is all done in the `wrangle_zillow.wrangle_data`, which takes a dataframe produced by `wrangle_zillow.get_zillow_data()` and calls the following functions:
- `filter_properties()` : filters the properties to only the above mentioned single unit properties.
- `handle_missing_values()` : Drops all rows and columns and columns that have greater and $0.01$ of data missing
- `clearing_fips()` : maps `fips` numbers to the correct county name and renames the column to `county`
- `latitude` and `longitude` are corrected by multiplying by $10^{-6}$ directly, not through a function

<a href='#contents'>Back to contents</a>