import pandas as pd

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
