# Market Scenario Generator Hackathon: From Stability to Storms

Most of what in here can be found in the competition, I only added works, results and other parts.
In this file, you can find:

* My submission model, parameters and data file,
* Since I do not know which stocks they are using for the competition, I use YahooFinance to take more stocks and replicate the competition for my own researches,
* The notebook indexed 0 is the one I used for most of my worsk during the competition, and the sample_bundle_submission is the file that I submitted and, fortunately, got in the top 2,
* For the other notebooks, they are me replicating the competition by taking the same amount of stocks (may be not the same ones) and having a proper in-sample and out-of-sample data for testing (of course in the competition I can only access the in-sample data). The data are in the data folder under the name: training_data.pkl, training_label.pkl, valid_data.pkl and valid_label.pkl. You can freely adjust the label (regular or crisis market) with the 1. Data notebook.

# Competition model

For the competion, my model is the Variational AutoEncoder (VAE) with some adjustments:
* Since we are generating daily log returns and daily volatility of the stocks, we cannot use the same activation functions for these 2 types. I seperated them with 'tanh' for the return and 'sigmoid' for the volatility.
* During the training process, I also changed the loss function. The total loss included: 1) An MSE between true and generated sample across whole, timesteps and features axis, 2) A KL distance between true and generated sample for each features, 3) An MSE between true and generated correlation matrix.
