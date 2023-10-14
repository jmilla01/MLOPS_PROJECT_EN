<p align=center><img src=https://neurona-ba.com/wp-content/uploads/2021/07/HenryLogo.jpg><p>

# <h1 align=center> **INDIVIDUAL PROJECT: MACHINE LEARNING OPERATIONS (MLOPS)** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>







<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

# JOAQUIN MILLAN LANHOZO - AUGUST 2023 - DTPT02

## SOY HENRY - BOOTCAMP

The work presented here was carried out during my time at the Soy Henry institution. I am part of the DATAPT02 group, and this is the first integrated project focused on Machine Learning Operations.
<hr>  


## Table of contents
- [Files in repository](#files-in-repository)
- [Context](#context)
- [Project](#project)
- [Deployment](#deployment)
- [Data Soucers](#data-sources)
- [Deploy video](#deploy-video)
- [Technology Stack](#technology-stack)

<hr> 

## FILES IN REPOSITORY

<hr> 

+ Folder "env" --> It contains the virtual environment generated for this project.
+ "Duration_of_movies.csv" --> A specific file generated for the "movies_duration" function query.
+ "countries_with_movies.csv" --> A CSV file generated for the "movies_languages" function.
+ "Succesful_companies.csv" --> A sub-dataset created for the "succesful_companies" function query. 
+ "countries_with_movies.csv" --> This CSV contains information about the count of movies produced by countries. Used in the "movie_country" function. 
+ "directors.csv" --> A DataFrame exported to CSV created for the "get_director" function.
+ "franchises.csv" --> A CSV file elaborated with the necessary information for the "franchises_function" function.
+ "main.py" --> File where the functions run and where I deployed it to FastAPI, making it available through Render.
+ "merge_franquicias.csv" --> A CSV file created with the necessary information for the "franquicia" function.
+ "movies_language.csv" --> A sub-dataset created for the "movies_languages" function query. 
+ "requirements.txt" --> A file with the libraries used and their versions.
+ "model.csv" --> A file filtered used for modelling the recommendations

[Table of contents](#table-of-contents)




## Context

For this project, I simulated being a Data Scientists for a startup in the Streaming industry. The startup does not yet have a developed platform to obtain information about the movies it has in its catalog.

Throughout the project, I performed ETL (Extraction, Transformation, and Loading of data) tasks. Extract information from the provided datasets, made some transformations to certain columns to have more robust and complete dataframes for then making Exploratory Data Analysis (EDA). EDA involves analyzing the data to obtain useful insights.

Next, I build a machine learning model to provide movie recommendations. Finally, I developed an application with FASTAPI to deploy this data available to users so they can consume it.

[Table of contents](#table-of-contents)

## Project

+ As a first step, I looked at the provided datasets, which consist of two datasets: one for "movies," containing information about the movies, and another for "credits," containing information about the cast who worked on the movies. The original datasets included 45,466 movies.

+ Transformations: The data within these datasets were not perfect, which is why we carried out transformations to make this data usable. Some columns were transformed because they contained data in inappropriate formats and also nested data that needed to be unraveled to make the information in those columns accessible. Additionally, we removed irrelevant columns that will not be considered for the purpose of this project.

+ ** Handling Missing Values**: Some missing values were treated as needed to ensure data quality.


## Creation of functions
  Specific and more focused dataframes were developed, filtering the information required, to optimize the use of the following functions. These functions will be used to query information about the movies.

+ **movies_languages( *`language`: str* )**:
    This function takes an input language and returns the number of movies produced in that language. 
    
Example: "*`X` number of movies were released in {certain} `language`."
         

+ **movies_duration( *`Movie`: str* )**:
    When given a movie title as input, this function returns the duration and release year of the movie. 
    
Example: "Movie: *`X`. Duration: `x`. Year: `xx`."

+ **franchises_function( *`Franchise`: str* )**:
    This function receives the name of a franchise as input and returns the number of movies in the franchise, the total revenue of the franchise, and the average revenue. 
    
Example: "The `X` franchise has `X` movies, a total revenue of `x`, and an average revenue of `xx`."

+ **movies_language( *`Country`: str* )**:
    This function returns the number of movies produced in the country given as an input. 

Example: "`X` movies were produced in country `X`."

+ **succesful_companies( *`Production_company`: str* )**:
    This function takes the name of a production company as an input and provides the total revenue and the number of movies produced by that company. 
    
Example: *"Production company `X` has had a revenue of `x`."*

+ **get_director( *`director_name`* )**:
    When you input the name of a director found in the dataset, this function returns the director's success measured by their returns. It also provides the name of each movie directed by the director, along with its release date, individual return, cost, and revenue.

+ **Recommendation System**: 

    This function accepts the name of a movie as input and provides recommendations for 5 similar movies. While it currently employs a KNN model, please note that a more effective recommendation model using TF-IDF and Cosine similarity is detailed in the .ipynb notebook (not included here due to deployment constraints that exceed the available resources).A esta función, se ingresa el nombre de una película y recomienda 5 peliculas similares.


### Exploratory Data Analysis (EDA)

A comprehensive analysis of the data was conducted, revealing insights into various aspects of the movies dataset. This included information about the languages spoken in the films, details about countries producing those movies, the original languages of the movies, production companies, movie collections, release dates, budgets, profits, ratings, genres, actors, and directors.

The dataset provides a rich reflection of the history of cinema, capturing significant milestones in both cultural and technological evolution.

On December 28, 1895, the Lumière brothers conducted one of the earliest public screenings of moving pictures in Paris, marking the official beginning of cinema.

The introduction of Technicolor in the 1910s and 1920s significantly improved color quality in films.
In the decades leading up to the 1930s, movies were predominantly black and white. However, significant advancements like sound, the birth of musicals, the emergence of movie stars and glamour, and the expansion of genres ushered in a new era for cinema. Technical innovations in cinematography, lighting, and editing also enhanced the visual and narrative quality of films, leading to increased popularity, diversity, and sophistication.

The 1940s saw the rise of animated films, with "Snow White and the Seven Dwarfs" (1937), produced by Walt Disney, considered one of the pioneering animated feature films.

In the 1960s, another marked trend of increased film production emerged. This was driven by factors such as the "New Hollywood" movement, global influential film movements like the French New Wave and Italian Neorealism, and the growing popularity of films from various cultures worldwide.

The 1980s witnessed exponential growth in film production, driven by technological innovations like special effects and widescreen formats. High-budget, blockbuster entertainment became a significant focus, with films like "Star Wars" (1977) and "Jaws" (1975) leading the way.

Starting from the 1990s, particularly in the 2000s and beyond, digital technology began revolutionizing film production and distribution. The rise of the Internet also transformed film promotion and distribution methods, coinciding with a surge in the number of movies produced, technological advancements, and exponential growth in the film industry.

This historical overview showcases the continuous evolution of cinema, shaped by cultural, technological, and global influences, ultimately leading to the diverse and dynamic film industry we know today.


[Table of contents](#table-of-contents)


## Deployment

To conclude the project, the data and functions were made available through a web service, allowing users to access and consume the information.

+ [API](https://ejemplo-joaquinmillan-deploy.onrender.com/docs)

[Table of contents](#table-of-contents)

## Fuente de datos

+ [Dataset](https://drive.google.com/drive/folders/1mfUVyP3jS-UMdKHERknkQ4gaCRCO2e1v): The dataset consists of two files that need to be processed: "movies_dataset.csv" and "credits.csv." It's important to note that there is nested data, with some values in rows represented as dictionaries or lists.
+ [Data Dictionary](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit#gid=0): Additionally, there is a data dictionary provided, which offers descriptions of some of the columns available in the dataset to help users understand the data's structure and meaning.

[Table of contents](#table-of-contents)

## Video Deploy
+ [Video](https://drive.google.com/file/d/1uGitaE-bxqTLBstm2RpZiYsGZ_pokkSk/view?usp=drive_link)
=======
# Video Deploy
+ [Video](https://drive.google.com/file/d/1uGitaE-bxqTLBstm2RpZiYsGZ_pokkSk/view?usp=drive_link) Video prooving that the deploy works and testing of fucntions works properly


[Table of contents](#table-of-contents)

## Technology stack

+  A Python notebook developed through Visual Studio Code served as the foundation for this project. The following libraries and frameworks were utilized:

+  Numpy Library -- [Numpy](https://numpy.org/) Numpy, short for Numerical Python, proved invaluable for logical and mathematical calculations on arrays and matrices.
+  Pandas library -- [Pandas](https://pandas.pydata.org/) Pandas, a Python library specialized in data management and analysis, was employed for handling structured data effectively.
+  Matplotlib library -- [Matplotlib](https://matplotlib.org/) Matplotlib, known for its versatility, facilitated data visualization within the project.
+  Seaborn library -- [Seaborn](https://seaborn.pydata.org/) Seaborn, working seamlessly with Matplotlib, further enhanced data visualization capabilities.
+  Datetime library -- [Datetime](https://docs.python.org/es/3/library/datetime.html) The Datetime library was instrumental in transforming date formats, ensuring accurate data handling.
+  Missingno library -- [Missingno](https://pypi.org/project/missingno/) Missingno emerged as a powerful tool for visualizing and addressing missing or null data points within the dataset.
+  Ast library -- [ast](https://docs.python.org/3/library/ast.html) The Ast module assisted Python applications in processing abstract syntax tree (AST) grammar, supporting various data manipulation tasks.
+  Sklearn Library -- [sklearn](https://scikit-learn.org/stable/) Scikit-learn, a widely adopted Python library, was employed for the creation and implementation of machine learning models.
+  Plotly library [Plotly](https://plotly.com/python/) Plotly, offering interactive data visualization capabilities and high-quality graphics, played a crucial role in enhancing data presentation.
+  FastApi [FastApi](https://fastapi.tiangolo.com/) FastAPI, a Python web framework, served as the backbone for swiftly developing and deploying APIs, leveraging Python 3.7.
+  Render [Render](https://render.com/) The project's API deployment was facilitated through Render, enabling online accessibility and showcasing the project to users.

[Table of contents](#table-of-contents)

