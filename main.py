from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn 
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = FastAPI(title='Individual Project MLOPS ',description='Joaquin Millan Lanhozo')

movies_language = pd.read_csv("movies_language.csv")
movies = pd.read_csv("Duration_of_movies.csv")
merge_franchises = pd.read_csv("franchises.csv")
countries_counts = pd.read_csv("countries_with_movies.csv")
succesful_production_companies = pd.read_csv("Succesful_companies.csv")
franquicias = pd.read_csv("franchises.csv")
directors = pd.read_csv("directors.csv")
model = pd.read_csv("model.csv")


@app.get('/movies_language/{language}')
def movies_languages(languages:str):

    '''
    Input the language, and It will return the number of movies produced in that language.
    '''
    languages = languages.lower()
    languages = str(languages)
    if languages in movies_language["country_code"].values:
        count = movies['original_language'].value_counts().get(languages, 0)
        return {'languages':languages, 'quantity':count}

    else: 
        return { "message":"Incorrect language code entered."}   

@app.get('/movies_duration/{movie}')
def movies_duration(movie:str):
    '''
    Enter the movie, and this endpoint will return the duration and the year of the movie.

    For example:"Jumanji"
    '''
    movie = movie.lower().title()
    movie = str(movie)
    if movie in movies["title"].values:
        
        movies_info = movies[movies["title"]== movie].title.values[0]
        duration = movies[movies["title"]== movie].runtime.values[0]
        year_m = movies[movies["title"]== movie].release_year.values[0]
        return {'movie':movies_info, 'duration':duration, 'year':year_m}

    else:
        return "We are sorry! The movie you are seeking doesn't have the information requested"
        
    
@app.get('/franchises/{franchise}')
def franchises_function(franchise:str):

    '''
    When you enter the franchise, the function will return the number of movies, the total earnings, and the average.

    Example: "James Bond Collection".
    '''
    
    franchise = str(franchise)
    if franchise in merge_franchises["Franchise"].values:
        
        franchise_name = merge_franchises[merge_franchises["Franchise"]== franchise].Franchise.values[0]
        amount_franchises = merge_franchises[merge_franchises["Franchise"]== franchise].Movie_Count.values[0]
        profit_total_franchise = merge_franchises[merge_franchises["Franchise"]== franchise].revenue.values[0]
        average_profit_franchise = merge_franchises[merge_franchises["Franchise"]== franchise].average_revenue_franchise.values[0]
        
        
        return {'franchise':franchise_name, 'quantity':amount_franchises, 'total_profit':profit_total_franchise, 'average_profit':average_profit_franchise}
        
    else:
        return "We are sorry! The franchise you are looking for does not have any information"


@app.get('/country_movies/{country}')
def movie_country( country: str ): 
    """
    You enter a country (as they are written in the dataset, no need to translate them!), 
    returning the number of movies produced in it.

    Example: "Argentina".
    """
    country = str(country)
    if country in countries_counts["production_countries_names"].values:
        
        country_name = countries_counts[countries_counts["production_countries_names"]== country].production_countries_names.values[0]
        count_countries = countries_counts[countries_counts["production_countries_names"]== country].Movie_Count.values[0]

        return {'country':country_name, 'Quantity':count_countries}

    else:

        return "We are sorry! The country you are looking for didn't produce movies"
        
@app.get('/succesful_companies/{Production_company}')
def succesful_companies(company:str):
    '''You enter the franchise, returning the number of movies, total revenue, and average
    
    For example: "Warner Bros.
    '''

    company = str(company)
    if company in succesful_production_companies["Production_Company"].values:
        
        company_name = succesful_production_companies[succesful_production_companies["Production_Company"]== company].Production_Company.values[0]
        amount_of_movies = succesful_production_companies[succesful_production_companies["Production_Company"]== company].Movie_Count.values[0]
        total_revenue = succesful_production_companies[succesful_production_companies["Production_Company"]== company].revenue.values[0]
        
        
        return {'Production_company':company_name, 'Quantity':amount_of_movies, 'revenue_total':total_revenue}
        
    else:
        return "We are sorry! The company searched for does not have the information requested" 

    

@app.get('/get_director/{director_name}')
def get_director(director_name):
    
    """
    You enter the name of a director that is within a dataset, and it should return their success measured through the return. 
    Additionally, it should return the name of each movie with its release date, 
    individual return, cost, and revenue in a list format."

    For example:"John Lasseter"
    """ 

    # Filter the DataFrame to obtain the movies directed by the given director.
    director_movies = directors[directors['director_name'] == director_name]
    
    # Verify if the director is in the DataFrame.
    if director_movies.empty:
        return None  # Or a message that indicates the director wasn't found
        
    # Calculate the director's success by summing the individual returns of their movies
    director_total_revenue = director_movies['return'].sum()
    
    # Create a list of dictionaries with the details for each movie

    movies = []
    for index, row in director_movies.iterrows():
        movies_info = {
            'name': row['title'],
            'year': row['release_date'],
            'movies_return': row['return'],
            'movies_budget': row['cost'],
            'movies_revenue': row['revenue']
        }
        movies.append(movies_info)
    
    # Dictionary for the answer
    answer = {
        'director': director_name,
        'directors_total_revenue': director_total_revenue,
        'movies': movies
    }
    
    return answer

@app.get('/recommendation/{title}')
def recommendation(given_movie:str):
    '''
    You enter a movie title, and it recommends similar ones in a list
    '''

    # Search for the movie title in the feature 'title'
    movie = model[model['title'] == given_movie]

    if len(movie) == 0:
        return {"message":"The movie you are looking for is not in our database."}

    # Getting the genre and popularity of the movie
    movie_genre = movie['genres_names'].values[0]
    movie_popularity = movie['popularity'].values[0]

    # Features characteristics for modelling with KNN
    features = model[['popularity']]
    genres = model['genres_names'].str.get_dummies(sep=' ')
    features = pd.concat([features, genres], axis=1)

    # Handling of missing values (NaN) replacing them by zeros
    features = features.fillna(0)

    # Nearest neighbours model
    nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn_model.fit(features)

    # Finding the more similar movies(excluding the movie which the user has given as an input)
    _, indices = nn_model.kneighbors([[movie_popularity] + [0] * len(genres.columns)], n_neighbors=6)
    similar_movies_indices = indices[0][1:]  # Exluding the first movie of the same query
    recommendations = model.iloc[similar_movies_indices]['title']

    # Si la película de consulta está en la lista de recomendaciones, la eliminamos
    if given_movie in recommendations.tolist():
        recommendations = recommendations[recommendations != given_movie]

    return {'Recommendation list': 
            recommendations}


