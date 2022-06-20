# Zillow Logerror Prediction

## Key Takeaways


## Contents

1. <a href='#intro'>Introduction</a>
2. <a href='#wrangle'>Wrangle</a>
    1. <a href='#acquire'>Acquire the Data</a>
    2. <a href='#prepare'>Prepare the Data</a>
3. <a href='#explore'>Explore</a>
    1. <a href='#split'>Splitting Data Into Subsets</a>
    1. <a href='#target'>Looking at the Target Variable</a>
    2. <a href='#geography'>Is `logerror` correlated with geography?</a>
    3. <a href='#clustering'>Using Clustering Algorithms to Explore the Data </a>
    4. <a href='#nongeo_clusters'>Exploring Non-Geographic Clusters </a>
    5. <a href='#geo_clusters'>Exploring Geographic clusters </a>
4. <a href='#model'>Modeling</a>
    1. <a href='#baseline_model'>Baseline Model </a>
    2. <a href='#model1'>Model I: Non-geo clustering `yearbuilt` stratification</a>
    3. <a href='#model2'>Model II: Geo clustering `yearbuilt` stratification</a>
    4. <a href='#model3'>Model III: Non-Geo clustering feature</a>
    5. <a href='#model4'>Model IV: Geo clustering feature</a>
    6. <a href='#model_eval'>Evaluating Models </a>
    7. <a href='#test'>Running the model on test data </a>
5. <a href='#conclusion'>Conclusions </a>
6. <a href='#appendix'>Appendix</a>

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

## Exploration <a name='explore'></a>

### Key Explore Takeaways
- Outliers for `logerror` are flagged with a boolean
- Outliers for `logerror` are more likely in Los Angeles, but Orange county has a higher mean `logerror`
- The KMeans clustering algorithm is run twice; once using geographic data, and once without it.
    - Both times there was a definite geographic split: East/West and Downtown/Suburbs
    - Both times there was a stratification based on the `yearbuilt` column

### Exploration Introduction

In the exploration stage of the analysis, data is split into subsets to prevent data leakage, the target variable is understood, and then multivariate analysis is also performed on a series of question.  For each question, there is a series of key takeaways, followed by a discussion with visualizations, and concluded by a series of hypothesis tests.  All hypothesis testing is done with a $0.95$ confidence interval ($α=0.05$).

The following `explore_zillow` custom module functions are used:
- `t_test_by_cat()` : performs a two-tailed, one sample t-test for a categorical variable over a continuous variables.  It splits the dataframe into the subcategories of the categorical variable and tests every combination with the continuous variables. It has the following parameters
    - `df` (DataFrame) : A dataframe containing the columns to be tested
    - `columns_cat` (list) : A list of categorical columns to be tested
    - `columns_cont` (list) : A list of continuous variables to be tested
- `t_test_by_cat_greater()` : same as `t_test_by_cat()` but performs a single tailed t-test where the alternative hypothesis is that the continous variable's mean is greater.  It uses the same parameters.
- `t_test_greater()` : performs a single tailed t-test on the hypothesis that the continous variable has a greater mean.  It has the following inputs:
    - `df` (Dataframe) : a dataframe containing the relevant information
    - `column_cat` (string) : name of the categorical variable
    - `subcat_val` (string) : name of the category within `column_cat`
    - `column_cont` (string) : name of the continuous variable to test

### Looking at the Target Variable: `logerror` <a name='target'></a>

#### Key Target Variable Takeaways
- `logerror` is normally distributed
- Outliers are not dropped, but they are flagged by the `is_outlier` column

### Is `logerror` correlated with geography? <a name='geography'></a>

#### Key Takeaways
- `logerror` outliers are more common in Los Angeles county
- Orange county has a higher mean `logerror` than the other two counties

<a href='#contents'>Back to content</a>

### Using Clustering Algorithms to Explore the Data <a name='clustering'></a>

#### Key Clustering Take Aways
- 6 clusters is optimal
- Clustering looks geographic, even if none of the parameters are geographic
- 2 cluster columns are made `clusters_geo` based on geographic data and `clusters` not based on geographic data

#### Discussion

Besides geography, there may be other groupings of data.  In order to expedite the process of finding useful groups, a clustering algorithm can be used to explore the data.  In order to increase a likelihood of success, the following columns are used for clustering:
- `county` : See discussion <a href='#geography'>above</a> for importance of geography to `logerror`
- `latitude` and `longitude` : Related to the question of geography, but allows for more spatial clustering
- `taxvaluedollarcnt` : Because this is the target variable of the model producing `logerror` it might be useful
- `bathroomcnt` : This was correlated with `taxvaluedollarcnt`
- `calculatedfinishedsquarefeet` : This was also correlated with `taxvaluedollarcnt`
- `yearbuilt` : It might be harder to predict whether or not an older house is valuable or not.  It could be valuable because it is historic and/or renovated (or have potential as a "fixer-upper")

<a href='#contents'>Back to contents</a>

### Exploring Non-Geographic Clusters <a name='nongeo_clusters'></a>

#### Key Takeaways
- Clusters 0 and 5 have a higher logerror
- Clusters 0 and 5 have a higher taxvaluedollarcnt
- `yearbuilt` was a good predictor of cluster, with clusters 0 and 5 being older and newer expensive houses, and the rest of the clusters representing some number of years
- It might be beneficial to use a piecewise model for clusters 0 and 5, then a model for the other clusters

<a href='#contents'>Back to contents</a>

### Exploring Geographic clusters <a name='geo_clusters'></a>

#### Key Takeaways
- Clusters 1 and 4 have a higher logerror.
- Clusters are still along `yearbuilt`, but less strongly, as geographic data might be more of a pull also
- Clusters 1 and 2 have a higher `taxvaluedollarcnt`, and this looks mainly due to significant outliers within these clusters
- The geographic clusters tend to split along an East/West divide


## Modeling <a name='model'></a>

### Key Takeaways
- Only some of the models beat the baseline prediction, and not by much

### Discussion

Four linear regression models are developed, validates, and the best model of the four is tested on unseen data.  The following metrics are used to evaluate the model:
- The Explained Variance Score : This helps determine the amount of variance the model is explaining
- RMSE : This represents the mean error, in the units of the target variable.  Although the `logerror` is unitless, this will still help evaluate the models
- Plotting residuals : Although this is largely subjective, it is still helpful in interpreting the RMSE

In considering how to model this data, two broad types of models are developed: 
- Models based on new features : These models use the data exploration stage to develop new features to pass to the linear regression model
    - Two models of this type are developed.  Each has a new feature based on the `yearbuilt` stratification found through clustering.
- Models based on clusters : Models that use the cluster number/name as a feature.
    - Two models are developed based on clusters found.  These are named.
    
All four models developed use a simple linear regression algorithm.
    
Non-engineered features to be used are the following:
- `county`
- `yearbuilt`
- `latitud`
- `longitude`
- `taxvaluedollarcnt`
- `bathroomcnt`
- `calculatedfinishedsquarefeet`

`parcelid` is also kept because it is a unique identifier for the row.

The following functions from the `wrangle_zillow` module are used:
- `make_X_and_y()` : Splits the dataframe into X and y sets where y contains only the target variable.  It also drops the `id` column.  Note that the `parcelid` will not be passed to the model.

### Evaluating Models <a name='model_eval'></a>

Models are evaluated on three criteria:
- Better RMSE than baseline model
- RMSE change between train and validate
- Explained variance score

<a href='#contents'>Back to content</a>

## Conclusions <a name='conclusion'></a>

### Key takeaways
- The model does not do better than the baseline
- `yearbuilt` or geographic data may be an indicator of logerror; however, this requires more investigation
- More information on the properties could be included from the data set, especially if some of the column values could be inferred
- Most important takeaway is that more time is needed to explore the data.
- `logerror` outliers would be beneficial to focus on; maybe develop a classification model for them

### For the future
There is a lot of data in the set that could still be explored given more time.  Some suggestions for future investigation could be the following:
- logerror might not be predicted by something intrinsic to the property; aggregating the data with wider economic trends could be beneficial
- Investigate the trends for house prices in the region.  Try modeling these to determine what factors might be unpredictable
- Research the economics of land to help determine how its value is evaluated
- Focus on logerror outlier might be a next step, as it might improve the model's RMSE more than focusing on only on lower logerror.

<a href='#contents'>Back to contents</a>

## Appendix <a name='appendix'></a>

### Reproducing this Project <a name = 'reproduce_project'></a>

1. Download `final_report.ipynb`, `explore_zillow.py` and `wrangle_zillow.py`
2. Make a `env.py` file based on `env_example.py`
3. Run `final_report.ipnyb`

<a href='#contents'>Back to contents</a>

### Data Dictionary <a name = 'data_dictionary'></a>

The following columns are retained:
- parcelid : row identifier
- bathroomcnt : number of bathrooms
- bedroomcnt : number of bedrooms
- calculatedfinishedsquarefeet : square feet in the house
- latitude
- longitude
- lotsizesquarefeet : lot size in square feet
- yearbuilt : year of construction
- taxvaluedollarcnt : target variable. Value of the house
- fips : relevant fips code mapped to county
- county : county name
- propertycountylandusecode : county law lands use code
- propertylandusetypeid : foreign key from SQL database
- rawcensustractandblock : census tract id
- regionidcounty : id for county
- regionidzip : zip code data
- landtaxvaluedollarcnt : tax value of the land plot
- censustractandblock : census track id number
- propertylandusedesc : description of the lot

The following are dropped columns:
- airconditioningtypeid
- architecturalstyletypeid
- basementsqft
- buildingclasstypeid
- buildingqualitytypeid
- decktypeid
- finishedfloor1squarefeet
- finishedsquarefeet13
- finishedsquarefeet15
- finishedsquarefeet50
- finishedsquarefeet6
- fireplacecnt
- garagecarcnt
- garagetotalsqft
- hashottuborspa
- heatingorsystemtypeid
- poolcnt
- poolsizesum
- pooltypeid10
- pooltypeid2
- pooltypeid7
- propertyzoningdesc
- regionidcity
- regionidneighborhood
- storytypeid
- threequarterbathnbr
- typeconstructiontypeid
- unitcnt
- yardbuildingsqft17
- yardbuildingsqft26
- numberofstories
- fireplaceflag
- taxdelinquencyflag
- taxdelinquencyyear
- taxamount
- structuretaxvaluedollarcnt
- calculatedbathnbr
- fullbathcnt
- finishedsquarefeet12
- propertylandusetypeid
- regionidcounty
- assessmentyear
- roomcnt
- id

<a href='#contents'>Back to contents</a>