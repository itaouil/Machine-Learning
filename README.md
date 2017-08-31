# **Multiple Linear Regression**


## Linearity First

In the occasion where multiple predictors (independent features) are present in our dataset we need need to upgrade the simple linear regression model to a multiple one.

However, before diving into the different approaches present to create a good multiple linear regression model. I want to stress out the point that there HAS to be linearity first for the model to work. This is means that the following assumptions need to be true:

1.
2.
3.
4.
5.

## MLR models

Not all of the predictors present in the dataset are necessarily good for the model (this can fall back to the Linearity first principle) or they are just not relevant at all as a predictor. So how to we know which predictors to use and which ones to drop from our model.

These are the different methods:

- All in (throw all predictors in the model)
- Backward elimination (remove predictors from the initial pot)
- Forward elimination (insert predictors in the pot)
- Bidirectional elimination (both back and forw elimination)
- All possible methods (computationally expensive)
