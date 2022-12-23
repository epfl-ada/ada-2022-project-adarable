# COVID-19 related changes in people's dietary habits

## Find our data story [here](https://emma-farina.github.io)

## Abstract

We have all experienced the negative impact of the lockdown on our health. Furthermore, studies have shown that the COVID-19 pandemic led to changes in people’s dietary habits. Some people took advantage of the lockdown as an opportunity to spend more time cooking [[1](https://www.sciencedirect.com/science/article/pii/S0195666321006504)]. However, sticking to the usual recipes quickly became tedious, pushing people to get out of their comfort zone and try new dishes. While some people took the opportunity to improve their diet quality (i.e., consuming more fresh products) [[1](https://www.sciencedirect.com/science/article/pii/S0195666321006504)], others became more interested in high-calorie food, most likely out of boredom [[2](https://www.nature.com/articles/s41467-022-28498-z)-[3](https://link.springer.com/article/10.1007/s13679-021-00466-6)]. Finally, it is known that the use of food delivery services had an explosive growth during lockdown [[4](https://blog.routific.com/10-stats-that-show-how-covid-19-impacted-food-delivery-services)], by making groceries restaurant dishes accessible to people confined at home. We will therefore investigate the influence of mobility restrictions and this success.

## Research questions

With the assumption that we can use Wikipedia pageviews to infer changes in food behavior [[5](https://www.mdpi.com/2072-6643/13/11/3683/htm)], we aim at addressing the following questions: Aiming at preserving one’s health: Did healthy food become more popular, in an effort to preserve one’s health or did unhealthy food receive the most attention? In the effort of people trying out new dishes: Did individuals become more interested in their own history and traditions, or did they want to explore the cuisine of other cultures? Regarding the boom of food delivery: is there a causal link between the mobility restrictions and this success? Can the interest for food delivery be forecast using data on the mobility restrictions?

## Proposed additional datasets

- **Wikipedia pageviews**: We extended the dataset provided by the ADA course (ranging from 01/01/2018 - 31/07/2020) until 01/11/2022 in an effort to have more data during (and after) the COVID-19 pandemic. This will allow us to draw more meaningful conclusions and include the post-COVID period. To address our research questions we refined the categorization of this dataset and get the corresponding pageviews.
- **Google Trends**: Using pytrends we download reports from Google Trends. We use this data to study the interest for different companies of food deliveries in several countries. The API allows us to get the historical Google Trend data for a list of up to five keywords over time or per region. To narrow down the result, we can specify categories for the keywords, select the geographic location, the Google property we are interested in (images, news etc.) and the needed range of time (01/01/2018 - 01/11/2022). 
- **Google Mobility data**: We extended the dataset provided by the ADA course (ranging from 01/01/2018 - 31/07/2020) until 01/11/2022. For our research questions we will use the categories named: ‘residential_percent_change_from_baseline
’, ‘retail_and_recreation_percent_change_from_baseline’.


## Methods

### Dataset construction and preprocessing

**Categories selection**: for our analysis on nutritional quality, we selected a list of categories associated with healthy and unhealthy food. To study the cultural dimension of food interests, we chose a list of categories of cuisine from different countries. Finally, to investigate the use of food delivery services, we built a list of companies by considering the most popular food delivery companies in each country we were interested in. 

**Wikipedia query**: Through the wikipedia API we obtain the daily pageviews of the titles of interest, which we aggregate depending on the defined categories

**Google Trends**: With the pytrends API, we collected the sum of the Google searches related to the food delivery companies in our list.

**Weekly grouping**: To make our analysis less sensitive to single-day peaks, not necessarily correlated with the covid pandemic, we obtain the weekly average of the pageviews of the different categories and of the Google searches of the delivery companies.

**Outliers handling**: We spot outliers by setting a threshold based on the standard deviation of the time series, then replace them with the rolling mean.

**Filtering and standardization**: We will either decompose the time series into seasonal component, trend, and residual (for world cuisines, because we want to see how the Covid period disrupted the seasonality of 2020), or standardize them based on the corresponding month in 2019 (for healthy and unhealthy foods, because we want to highlight the difference compared to the pre-Covid period).

### Analysis

For the first part of our analysis, we extract and preprocess the pageviews of the healthy and unhealthy food categories in different languages. We associate the pageviews in each of the available languages to the mobility data in the countries in which the language is spoken. We perform a z-score standardization of the pageviews data using the mean and standard deviation of the pageviews from the same month in 2019. This allows us to visualize the excess and deficit of interest compared to 2019 [[1](https://www.nature.com/articles/s41467-022-28498-z)]. We then compute the ratio between the un-standardized pageviews for the healthy categories and those for the unhealthy ones. We perform a t-test to compare the distributions of this ratio in the pre-Covid period and during lockdown, using the interventions dataset to get information about lockdown dates in different countries. Then, to investigate the link between the interest shifts and the time spent at home, we compute Spearman’s correlation coefficient between the severity of mobility restrictions and the pageviews ratio of interest.

For our second research question, after having created the dataframes for each nation concerning the interest in other cuisines in the world, we evaluate the presence of a greater preference towards one's own or other cultures in the world compared to the pre-pandemic period. We therefore use a t-test, and then evaluate the presence of a Spearman correlation and a Granger causality by comparing the time series of Wikipedia pageviews with the mobility data provided by Google.

For our third research question, we investigate for each country under analysis the presence of a structural change in the levels of interest for food delivery before and after the starting of the mobility restrictions. We use a t-test to test the null hypothesis that pre- and post-pandemic interest in food delivery has the same average value. Then, to verify the hypothesis that the lockdown and all the mobility restrictions due to the pandemic have affected people's interest in food delivery services  we run a Spearman test between the mobility data related to retail and recreation, and the Google searches for food delivery services.
Further, we explore whether causality exists between mobility and the food delivery interest. To conclude, we build a forecasting model through the vector autoregression (VAR) model, an extension of the ARMA model utilized for univariate time series forecasting.


## Organization within the team

Alexia: RQ1-healthy/unhealthy food

Tobias: RQ2-world cuisines

Emma: RQ2-world cuisines, website

Desirée: RQ3-Food delivery services
