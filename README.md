# Title
subtitle

## Abstract

We have all experienced the negative impact of the lockdown on our health. Furthermore, studies have shown that the COVID-19 pandemic led to changes in people’s dietary habits. Some people took advantage of the lockdown as an opportunity to spend more time cooking [1]. However, sticking to the usual recipes quickly became tedious, pushing people to get out of their comfort zone and try new dishes. While some people took the opportunity to improve their diet quality (i.e., consuming more fresh products) [1], others became more interested in high-calorie food, most likely out of boredom [2-3]. Finally, it is known that COVID-19 affects taste perception. Therefore, changes in food habits can also be caused by a loss of taste even for successive months after having been infected [4]. Using Wikipedia pageviews from 2018 to 2022, we will investigate the evolution of food habits caused by the COVID-19 pandemic.

[ref: https://www.sciencedirect.com/science/article/pii/S0195666321006504]
[ref: https://www.nature.com/articles/s41467-022-28498-z]
[ref: https://link.springer.com/article/10.1007/s13679-021-00466-6]
[ref: https://www.swft.nhs.uk/application/files/7215/8876/5922/Loss_of_taste_COVID.pdf]

## Research questions

With the assumption that we can use Wikipedia pageviews to infer changes in food behavior [5], we aim at addressing the following questions: Aiming at preserving one’s health: Did healthy food become more popular, in an effort to preserve one’s health or did unhealthy food receive the most attention? In the effort of people trying out new dishes: Did individuals become more interested in their own history and traditions, or did they want to explore the cuisine of other cultures? Due to sensory dysfunction: Did the loss of food taste have a significant influence on food preferences? Did this shift in culinary habits persist after the pandemic?

[ref: https://www.mdpi.com/2072-6643/13/11/3683/htm]

## Proposed additional datasets

- **Wikipedia pageviews**: We extended the dataset provided by the ADA course (ranging from 01/01/2018 - 31/07/2020) until 01/11/2022 in an effort to have more data during (and after) the COVID-19 pandemic. This will allow us to draw more meaningful conclusions and include the post-COVID period. To address our research questions we need to refine the categorization of this dataset and get the pageviews for the topics we are interested in.
- **Google trends**: Using pytrends we download reports from Google Trends. We intend to use this data to confirm the validity of using Wikipedia pageviews to study dietary. The API allows us to get the historical Google Trend data for a list of up to five keywords over time or per region. To narrow down the result, we can specify categories for the keywords, select the geographic location, the Google property we are interested in (images, news etc.) and the needed form of the data (hourly, weekly, monthly).
- **Daily covid cases**: It is a collection of the COVID-19 data maintained by Our World in Data [6]. It contains daily data regarding: vaccinations, tests & positivity, hospital & ICU, confirmed cases, confirmed deaths, reproduction rate, policy responses. For our analysis we will consider the daily new cases data, and we will compute their weekly average.

[ref: https://github.com/owid/covid-19-data/tree/master/public/data]


## Methods

#### Dataset construction and preprocessing

**Categories selection**: for our analysis on nutritional quality, we selected a list of categories associated to healthy food and another for categories of unhealthy food. To study the cultural dimension of food interests, we choose a list of categories of cuisine from different countries. Finally, to investigate the influence of the loss of senses on the interest in food, we researched in the literature the foods that were more difficult to taste for covid patients, and the ones that people would consume in an effort to recover the sense of taste.

**Wikipedia query**: Through the wikipedia API we obtain the daily pageviews of the titles of interest, which we aggregate depending on the defined categories 
**Weekly grouping**: To make our analysis less sensitive to single-day peaks, not necessarily correlated with the covid pandemic, we obtain the weekly average of the pageviews of the different categories and proceed by considering this data in our analyses. 

**Outliers handling**: Even considering the weekly pageviews, sometimes there are huge spikes that reach up to 3 times the amount of pageviews ofcompared to the previous and following week. We spot these outliers by setting a threshold based on the standard deviation of the time series, then replace them with the rolling mean.

**Filtering and standardization**: We will either decompose the time series into seasonal component, trend, and residual (for world cuisines, because we want to see how the Covid period disrupted the seasonality of 2020), or standardize themthe time series based on the corresponding month in 2019 (for healthy and unhealthy foods, because we want to highlight the difference compared to the pre-Covid period).

For the first part of our analysis, we will associate Wikipedia pageviews in each of the available languages to the mobility data in the countries in which the language is spoken. We will first use a t-test to compare the distributions of the pageviews ratios (healthy/unhealthy foods and local/foreign cuisines) in the pre-Covid period and during lockdown, using the interventions dataset to get information about lockdown dates in different countries. Then, to investigate the link between the interest shifts and the time spent at home, we will define the severity of the lockdown as the increase in the percentage of time spent at home at the peak of reduced mobility [7]. Finally, we will compute Pearson’s correlation coefficient between the severity of mobility restrictions and the pageviews ratio of interest.

For the analysis regarding taste perception, we will consider the total weekly number of pageviews for several categories of foods associated with taste loss and investigate their relationship with the number of Covid cases in the countries of interest. Since we expect a possible shift in food habits to happen with a delay with respect to the Covid diagnosis, our approach consists in using cross correlation to measure the degree of similarity between the pageviews time series and the lagged version of the Covid cases time series.

[ref: https://www.nature.com/articles/s41467-022-28498-z]


## Proposed timeline

## Organization within the team

## Questions for TAs

- Could you recommend some preprocessing techniques for time series?
- How much can we copy methods from other papers like the one of the lab? (Compute average relative change in interest, Spearman correlation coefficient between pageviews and mobility reduction…) And how much can we inspire ourselves from their plots?
