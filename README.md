# Machine Learning Nanodegree Capstone

## Image Segmentation and Masking with Mask R-CNN

This project implements the Mask R-CNN model and applies it to the Kaggle 2018 Datascience Bowl (https://www.kaggle.com/c/data-science-bowl-2018). Specifically, Mask R-CNN is used to segment and mask cell nuclei presented in divergent images. The challenge is to build a model that generalizes well across varying image sizes, magnification, cell type, and image modality. The Mask R-CNN architecture is sources from https://github.com/matterport/Mask_RCNN.

A project proposal is available in the document proposal.pdf and the project report is available in report.pdf. A review of the proposal can be found [here](http://api.getblueshift.com/track?uid=355c927b-7731-47fb-b93f-a76ccb3a9790&mid=0e4ac96d-3463-42e7-89e4-3e808fc47c32&eid=6f154690-7543-4582-9be7-e397af208dbd&txnid=23d2384c-5801-44ee-a94a-8903c49b6402&a=click&redir=https%3A%2F%2Freview.udacity.com%2F%3Futm_medium%3Demail%26utm_campaign%3Dret_000_auto_ndxxx_submission-reviewed%26utm_source%3Dblueshift%26utm_content%3Dreviewsapp-submission-reviewed%23%21%2Freviews%2F1072206).

## Usage

Download or clone the repository and open the Jupyter notebook titled capstone_main.ipynb. Step through the notebook to prepare the dataset, instantiate the model, train the model, and generate mask predictions on a test set. Alternatively, explore the 2018 DSB dataset using the notebook capstone_inspect_data.ipynb. Submission files (for submission to the Kaggle competition) can be evaluated and scored using the notebook capstone_test_score.ipynb. Please note, the test set used in this project is the stage 1 test set from the 2018 DSB and can no longer be submitted to Kaggle for scoring.

*Zipped folders in /data must be unzipped with the folder name unchanged before using notebooks.

/logs contains the saved weights of the solution model of Mask R-CNN with a ResNet101 backbone.

/output contains the submission file of the predicted masks from the solution model. This file can be evaluated and scorred using capstone_test_score.ipynb.

Dependencies used in project:
- numpy			1.14.2
- pandas        	0.20.3
- matplotlib    	2.2.2
- seaborn       	0.8
- imgaug (requires cv2)	0.2.5
- tqdm          	4.19.5
- scikit-image  	0.13.0
- scikit-learn  	0.19.1
- scipy         	1.0.1
- Keras         	2.0.9
- tensorflow    	1.4.0rc0
