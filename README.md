# Title
subtitle

## Abstract

## Research questions

We have all experienced the negative impact of the lockdown on our health, especially in terms of snacking and lack of physical activity. A recent study on Google Trends [https://www.nature.com/articles/s41467-022-28498-z] suggests that people became more interested in high-calorie food during this period, most likely out of boredom. However, the pandemic may have also led people to get more worried about their health. We thus ask ourselves: did healthy food become more popular, in an effort to preserve one’s health? Or did unhealthy food receive the most attention?

Furthermore, during COVID-19 most food was prepared and consumed at home, due to mobility restrictions. Sticking to the usual recipes quickly became boring, pushing people to get out of their comfort zone and try new dishes. Additionally, people could take more time to research more complex recipes. It would be interesting to look at this phenomenon across different countries: did individuals become more interested in their own history and traditions, or did they want to explore the cuisine of other cultures?

Finally, it is known that COVID-19 affects taste perception. Indeed, a lot of people lost their taste of food even for the successive months after they tested positive. Did sensory dysfunction affect their food preferences? Did this shift in culinary habits persist after the pandemic?

## Proposed additional datasets

- **Google trends**: we intend to use this to confirm the validity of wikipedia in the study of culinary trends.
- **Non aggregated wikipedia time series**: we need this in order to refine the categorization of our main dataset and get the pageviews for the topics we are interested in.
- **Daily covid cases**: we intend to use this to study the influence of the covid cases and the consequent loss of taste for food and the shift of interest on the type of food.

## Methods

#### Dataset construction and preprocessing
- **Categories selection**: we select the wikipedia categories we are interested in in our research: for Q1, …, for Q2, cuisines from different countries, for Q3, …. [TO BE CONTINUED]
- **Wikipedia query**: Through the wikipedia API we obtain the daily pageviews of the titles belonging to the defined categories relevant to our study questions. To have the wikipedia page titles in the twelve languages proposed in the database, we use the wiki-api library and translate the category titles starting from the English language. One drawback of using categories is that some of them may exist in some languages but not in others. Similarly, they may contain different pages depending on the language. We will therefore have to deal with some missing data. We are including in our analysis both mobile and computer access, since we want to study culinary habits regardless of which device people prefer to use to do their research. We extend the temporal duration of information acquisition of wikipedia pageviews to the present day. This may help us answer questions regarding whether certain changes due to covid are still influencing individuals' searches.
- **Weekly grouping**: To make our analysis less sensitive to single-day peaks, not necessarily correlated with the covid pandemic, we obtain the weekly average of the pageviews of the different categories and proceed by considering this data in our analyses. In this regard, considering that the API does not report daily data for all wikipedia pages, we eliminate from our analysis the search titles that do not have complete data. Besides, we know that daily covid cases are more meaningful when averaged over the week since less are counted on Sundays, and more on Mondays to catch up. As we relate the pageviews to covid cases, it makes sense to use weekly averages for both.
- **Interventions data**: We also want to use in our analysis the changes in mobility during the period of the pandemic. For this we use the provided dataset containing intervention dates for different languages.

#### Preliminary analysis
- How to deal with seasonal oscillations of the data: comparing with same month in 2019 or using the library numpy-stl
- Cross correlation

#### Q1

#### Q2

#### Q3

## Proposed timeline

## Organization within the team

## Questions for TAs

- Could you recommend some preprocessing techniques for time series?
- How much can we copy methods from other papers like the one of the lab? (Compute average relative change in interest, Spearman correlation coefficient between pageviews and mobility reduction…) And how much can we inspire ourselves from their plots?
