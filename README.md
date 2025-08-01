# ABSA_VLSP2018_RESTAURANT
Description of that problem: Given a list of Aspect#Category and for each comment of customer, you have to predict which Aspect#Category was in that comment, and what was the sentiment of that Aspect#Category (Positive, Negative, Neutral). You can see it as a Multi-Label Problem in Classification.

DATASET: Restaurant domain in Sentiment Analysis of VLSP2018

Example:

"Nhà hàng này nước uống rẻ, tuy nhiên thức ăn chưa được ngon, wifi thì tạm được, lúc nhanh lúc chậm" (Vietnamese)

=> RESTAURANT#DRINKS, POSITVE

=> RESTAURANT#FOODS, NEGATIVE

=> RESTAURANT#MISCELLANEOUS, NEUTRAL

In that problem above, I go with two approach: Fine-tune PhoBERT-base, Fine-tune viT5-base.
This is my experimental result compared to other approaches:

<img width="1243" height="606" alt="image" src="https://github.com/user-attachments/assets/ee148c85-10b6-4ad6-a2f5-a801477fd7be" />

Reference (Preprocessing, Evaluating): https://github.com/ds4v/absa-vlsp-2018
