from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv('preproc/movies.csv')
books = pd.read_csv('preproc/books.csv')
cosine_sim = np.load('preproc/cosine_sim.npy')

movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'].apply(lambda x: x.lower() if x is not np.nan else "")).drop_duplicates()

# soups = pd.concat([movies['soup'], books['soup']], ignore_index=True)

# count = CountVectorizer(stop_words="english")
# count.fit(soups)

# movies_matrix = count.transform(movies['soup'])
# books_matrix = count.transform(books['soup'])

# cosine_sim = cosine_similarity(movies_matrix, books_matrix)


def content_recommender(title, lim=10):
    idx = indices[title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:lim]
    book_indices = [i[0] for i in sim_scores]
    return list(books.iloc[book_indices]["original_title"])

# def main():
#     recommendations = content_recommender('The Matrix', lim=10)
#     for book in recommendations:
#         print(book)

# if __name__ == "__main__":
#     main()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']  # Get the movie name from the form
        recommended_movies = content_recommender(movie_name)  # Call your function to get the recommended movies
        return render_template('result.html', movies=recommended_movies)
    return render_template('index.html')


# @app.route("/")
# def hello_world():
#     return render_template('index.html')

@app.route("/m2m")
def mov2mov():
    return render_template('m2m.html')

if __name__ == "__main__" : 
    app.run(debug=True)