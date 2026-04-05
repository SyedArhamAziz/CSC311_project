import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ml_challenge_dataset_fixed.csv")
df = df.dropna()

paintings = ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']

df['This art piece makes me feel sombre.'] = df['This art piece makes me feel sombre.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)
df['This art piece makes me feel content.'] = df['This art piece makes me feel content.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)
df['This art piece makes me feel calm.'] = df['This art piece makes me feel calm.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)
df['This art piece makes me feel uneasy.'] = df['This art piece makes me feel uneasy.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)

stats_df = pd.DataFrame()
stats_df['Paintings'] = paintings

avg_prices = []
avg_emotions = []
avg_sombrenesses = []
avg_contentnesses = []
avg_calmnesses = []
avg_uneasinesses = []
avg_objects = []
avg_colors = []

for painting in paintings:
    painting_df = df[df['Painting'] == painting]
    painting_df = painting_df[painting_df['How much (in Canadian dollars) would you be willing to pay for this painting?'] < 10**8]

    stats = []

    num = len(painting_df)
    
    total_price = sum(painting_df['How much (in Canadian dollars) would you be willing to pay for this painting?'])
    avg_price = total_price/num
    avg_prices.append(avg_price)
    #print(f'Average price for {painting}: {avg_price:.2f}')
#    plt.title(f'Box Plot showing distribution of prices for {painting}')
#    plt.boxplot(painting_df["How much (in Canadian dollars) would you be willing to pay for this painting?"])
#    plt.show()

    total_emotion = sum(painting_df['On a scale of 1–10, how intense is the emotion conveyed by the artwork?'])
    avg_emotion = total_emotion/num
    avg_emotions.append(avg_emotion)
    #print(f'Average emotional intensity for {painting}: {avg_emotion}')
#    plt.title(f'Box Plot showing distribution of emotion for {painting}')
#    plt.boxplot(painting_df["On a scale of 1–10, how intense is the emotion conveyed by the artwork?"])
#    plt.show()

    total_sombreness = sum(painting_df['This art piece makes me feel sombre.'])
    avg_sombreness = total_sombreness/num
    avg_sombrenesses.append(avg_sombreness)
    #print(f'Average sombreness for {painting}: {avg_sombreness}')

    total_contentness = sum(painting_df['This art piece makes me feel content.'])
    avg_contentness = total_contentness/num
    avg_contentnesses.append(avg_contentness)
    #print(f'Average contentness for {painting}: {avg_contentness}')

    total_calmness = sum(painting_df['This art piece makes me feel calm.'])
    avg_calmness = total_calmness/num
    avg_calmnesses.append(avg_calmness)
    #print(f'Average calmness for {painting}: {avg_calmness}')

    total_uneasiness = sum(painting_df['This art piece makes me feel uneasy.'])
    avg_uneasiness = total_uneasiness/num
    avg_uneasinesses.append(avg_uneasiness)
    #print(f'Average uneasiness for {painting}: {avg_uneasiness}')

    total_colors = sum(painting_df['How many prominent colours do you notice in this painting?'])
    avg_color = total_colors/num
    avg_colors.append(avg_color)
    #print(f'Average colors for {painting}: {avg_colors}')

    avg_object = sum(painting_df['How many objects caught your eye in the painting?'])/num
    avg_objects.append(avg_object)
    #print(f'Average objects for {painting}: {avg_objects}')

stats_df['Prices'] = avg_prices
stats_df['Emotions'] = avg_emotions
stats_df['Sombreness'] = avg_sombrenesses
stats_df['Contentness'] = avg_contentnesses
stats_df['Calmnesses'] = avg_calmnesses
stats_df['Uneasiness'] = avg_uneasinesses
stats_df['Colors'] = avg_colors
stats_df['Objects'] = avg_objects

counts = (
    df.assign(room=df['If you could purchase this painting, which room would you put that painting in?'].str.split(','))
      .explode('room')
      .groupby(['Painting', 'room'])
      .size()
      .unstack(fill_value=0)
)
counts = counts.reset_index()
counts = counts.drop(columns=['Painting'])
stats_df = pd.concat([stats_df, counts], axis=1)

counts = (
    df.assign(room=df['If you could view this art in person, who would you want to view it with?'].str.split(','))
      .explode('room')
      .groupby(['Painting', 'room'])
      .size()
      .unstack(fill_value=0)
)
counts = counts.reset_index()
counts = counts.drop(columns=['Painting'])
stats_df = pd.concat([stats_df, counts], axis=1)

counts = (
    df.assign(room=df['What season does this art piece remind you of?'].str.split(','))
      .explode('room')
      .groupby(['Painting', 'room'])
      .size()
      .unstack(fill_value=0)
)
counts = counts.reset_index()
counts = counts.drop(columns=['Painting'])
stats_df = pd.concat([stats_df, counts], axis=1)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(stats_df)
    None

stats_df.to_csv('stats.csv', index=False)

