from conceptnet5.vectors.query import VectorSpaceWrapper
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np

df = pd.read_json("train.json")

cuisines = df.cuisine.unique()
amt_cuisines = len(cuisines)

'''
    get 1% most common ingredients (across all recipes)
    and remove them, almost treating them as stopwords
'''
common_ingredients = {}


# Fetching most common
def fetch_common_from_rec(ingreds):
    for ingred in ingreds:
        if ingred not in common_ingredients.keys():
            common_ingredients[ingred] = 1
        else:
            common_ingredients[ingred] += 1


df.ingredients.apply(fetch_common_from_rec)
common_ingredients_sorted = sorted(common_ingredients.items(),
                                   key=lambda kv: kv[1],
                                   reverse=True)
one_perc = int(len(common_ingredients_sorted) / 100)
most_common = [common_ingredients_sorted[i][0] for i in range(one_perc)]


# Removing em from the df
def remove_common_from_rec(ingreds):
    for common in most_common:
        if common in ingreds:
            ingreds.remove(common)


print("Removing {} ingredients from all recipes...".format(one_perc))
df.ingredients.apply(remove_common_from_rec)

'''
    separate all ingredients into 1-word tokens
'''
tokenizer = RegexpTokenizer(r'\w+')


# separating
def separate_ingredients(ingreds):
    new_ingreds = []
    for ingred in ingreds:
        for word in tokenizer.tokenize(ingred):
            new_ingreds.append(word)
        ingreds.remove(ingred)
    return new_ingreds


separated = [separate_ingredients(recipe) for recipe in df.ingredients]
df["separated_ingredients"] = separated

'''
    Use separated ingredients to create feature vectors
'''
wrapper = VectorSpaceWrapper("../Vectors/mini.h5")


# using conceptnet to create feature vectors
def create_feature_vector(recipe):
    vecs = []
    for ingred in recipe:
        vecs.append(wrapper.text_to_vector("en", ingred))
    return sum(vecs)


inputs = [create_feature_vector(recipe) for recipe in df.separated_ingredients]
print("Input vectors are of shape {}".format(np.shape(inputs[0]))) # (300,)


'''
    Transform outputs into vectors
'''


# one-hot encoding outputs
def cuisine_to_vec(cuisine):
    out = np.array((amt_cuisines,))
    out[cuisines.index(cuisine)] = 1
    return out


outputs = [cuisine_to_vec for cuisine in df.cuisines]
print(outputs[0])