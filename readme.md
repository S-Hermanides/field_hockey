## Readme

### Introduction

Welcome to the code for my first ever solo project, using linear regression to predict the win percentage for a field hockey team. The goal of this project was to try and improve upon the Pythagorean Expectation as mentioned by Chris Fry [here](https://chrisfryperformanceanalyst.wordpress.com/2020/05/25/applying-the-pythagorean-expectation-to-ncaa-d1-field-hockey/).

In the code folder for this project you'll find the following files:

* 1_scraping
  * Getting the data from the NCAA stats website
* 2_data_clean
  * Getting the data ready for modeling
* 3_eda_modeling
  * Model selection and parameter tuning leading to the final predictions
* cross_val.py
  * Helper functions for the modeling notebook, specifically for the cross-validation pipeline
* Scraping_functions.py
  * Helper functions for the scraping process
