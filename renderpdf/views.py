import json
import boto3
import pandas as pd
import numpy as np
import io

from datetime import datetime
from django.shortcuts import render
from django.views.generic import View
from django.utils import timezone
from django.contrib.auth.mixins import LoginRequiredMixin
from checkout.models import Order
from jchart import Chart

from ast import literal_eval
from sklearn.preprocessing import Imputer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .forms import FormDatePdf
from .render import Render
from .fusioncharts import FusionCharts


class Pdf(LoginRequiredMixin, View):

    def get(self, request):
        orders = Order.objects.filter(user=self.request.user)
        today = timezone.now()
        params = {
            'today': today,
            'orders': orders,
            'request': request
        }
        return Render.render('render_pdf.html', params)


class PdfDate(LoginRequiredMixin, View):

    def get(self, request):
        form = FormDatePdf()
        return render(request, 'render_pdf_date.html', {'form': form})

    def post(self, request):
        form = FormDatePdf(request.POST)
        if form.is_valid():
            date = request.POST['date']
            now = datetime.now()
            date = now.strptime(date, "%d/%m/%Y").strftime("%Y-%m-%d")
            orders = Order.objects.filter(created__date=date)
            today = timezone.now()
            params = {
                'today': today,
                'orders': orders,
                'request': request
            }
            return Render.render('render_pdf.html', params)


class RenderChart(LoginRequiredMixin, View):
    def get(self, request):
        set_dates = {}
        orders = Order.objects.filter(user=request.user)
        for order in orders: #%Y-%m-%d
            if order.created.strftime('%d/%m/%Y') not in set_dates:
                set_dates[order.created.strftime('%d/%m/%Y')] = 1
            else:
                set_dates[order.created.strftime('%d/%m/%Y')] += 1
        
        categories = list()
        categories_count = list()

        for key, value in set_dates.items():
            categories.append(key)
            categories_count.append(value)
        
        count_series = {
            'name': 'Quantidade',
            'data': categories_count,
            'color': 'blue'
        }

        chart = {
            'chart': {'type': 'column'},
            'title': {'text': 'Quantidade de Pedidos Realizados'},
            'xAxis': {'categories': categories},
            'series': [count_series]
        }

        dump = json.dumps(chart)
        MachineLearning()
        return render(request, 'render_chart.html', {'chart': dump})


def load_movies(path):
    data = pd.read_csv(path)
    print("começou movie")
    data['release_date'] = pd.to_datetime(data['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        data[column] = data[column].apply(json.loads)
    print("terminou movies")
    return data

def load_credits(path):
    data = pd.read_csv(path)
    print("começou credits")
    json_columns = ['cast', 'crew']
    for column in json_columns:
        data[column] = data[column].apply(json.loads)
    print("terminou credits")
    return data

def safe_access(container, index_values):
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan

def get_director(crew):
    directors = [x['name'] for x in crew if x['job'] == 'Director']
    return safe_access(directors, [0])

def get_words(words):
    return '|'.join([x['name'] for x in words])

def convert(dataset_movies, dataset_credits):
    COLUMNS_EQUIVALENCIES = {
        'budget': 'budget',
        'genres': 'genres',
        'revenue': 'gross',
        'title': 'title',
        'runtime': 'duration',
        'original_language': 'language',
        'keywords': 'keywords',
        'vote_count': 'num_voted_users',
    }

    movies = dataset_movies.copy()
    movies.rename(columns=COLUMNS_EQUIVALENCIES, inplace=True)
    movies['title_year'] = pd.to_datetime(movies['release_date']).apply(lambda x: x.year)
    movies['country'] = movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    movies['language'] = movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    movies['director_name'] = dataset_credits['crew'].apply(get_director)
    movies['actor_1_name'] = dataset_credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    movies['actor_2_name'] = dataset_credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    movies['actor_3_name'] = dataset_credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    movies['genres'] = movies['genres'].apply(get_words)
    movies['keywords'] = movies['keywords'].apply(get_words)
    return movies

def weighted_rating(x):
    v = x['num_voted_users']
    R = x['vote_average']
    return (v / (v + minimo) * R) + (minimo / (minimo + v) * C)

def gaussian(x, y, sigma):
    return lambda x, y, sigma: np.exp(-(x - y) ** 2 / (2 * sigma ** 2))

def capture_director_actor_keywords(df, id_):
    col_labels = []    
    if pd.notnull(df['director_name'].iloc[id_]):
        for name in df['director_name'].iloc[id_].split('|'):
            col_labels.append(name)
            
    for i in range(3):
        column = 'actor_NUM_name'.replace('NUM', str(i + 1))
        if pd.notnull(df[column].iloc[id_]):
            for name in df[column].iloc[id_].split('|'):
                col_labels.append(name)
                
    if pd.notnull(df['keywords'].iloc[id_]):
        for word in df['keywords'].iloc[id_].split('|'):
            col_labels.append(word)
    return col_labels

def add_variables(df, var):    
    for s in var: df[s] = pd.Series([0 for _ in range(len(df))])
    colonnes = ['genres', 'actor_1_name', 'actor_2_name',
                'actor_3_name', 'director_name', 'keywords']
    for categorie in colonnes:
        for index, row in df.iterrows():
            if pd.isnull(row[categorie]): continue
            for name in row[categorie].split('|'):
                if name in var: df.set_value(index, name, 1)            
    return df

def recommand(df, id_):
    df_copy = df.copy(deep = True)    
    liste_genres = set()
    for s in df['genres'].str.split('|').values:
        liste_genres = liste_genres.union(set(s))
    # Criada uma variavel adicional para chegar a similaridade
    variables = capture_director_actor_keywords(df_copy, id_)
    variables += list(liste_genres)
    df_new = add_variables(df_copy, variables)

    X = df_new.as_matrix(variables)
    model_nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='euclidean').fit(X)

    distances, indices = model_nbrs.kneighbors(X)    
    xtest = df_new.iloc[id_].as_matrix(variables)
    xtest = xtest.reshape(1, -1)

    distances, indices = model_nbrs.kneighbors(xtest)
    return indices[0][:]

def sequel(titre_1, titre_2):    
    if fuzz.ratio(titre_1, titre_2) > 50 or fuzz.token_set_ratio(titre_1, titre_2) > 50:
        return True
    else:
        return False

def critere_selection(title_main, max_users, year_ref, titre, year, imdb_score, votes):    
    if pd.notnull(year_ref):
        facteur_1 = gaussian(year_ref, year, 20)
    else:
        facteur_1 = 1        

    sigma = max_users * 1.0

    if pd.notnull(votes):
        facteur_2 = gaussian(votes, max_users, sigma)
    else:
        facteur_2 = 0
        
    if sequel(title_main, titre):
        note = 0
    else:
        note = imdb_score ** 2 * facteur_1 * facteur_2
    return note

def extract_parameters(df, list_films):     
    parametres_films = ['_' for _ in range(50)]
    i = 0
    max_users = -1
    for index in list_films:
        parametres_films[i] = list(df.iloc[index][['title', 'title_year', 'imdb_score', 'num_user_for_reviews', 'num_voted_users']])
        parametres_films[i].append(index)
        max_users = max(max_users, parametres_films[i][4] )
        i += 1
        
    title_main = parametres_films[0][0]
    year_ref  = parametres_films[0][1]
    parametres_films.sort(key = lambda x: critere_selection(title_main, max_users, year_ref, x[0], x[1], x[2], x[4]), reverse = True)
    return parametres_films

def add_to_selection(film_selection, parametres_films):    
    film_list = film_selection[:]
    count = len(film_list)    
    for i in range(50):
        already_in_list = False
        for film in film_selection:
            if film[0] == parametres_films[i][0]: already_in_list = True
            if sequel(parametres_films[i][0], film[0]): already_in_list = True            
        if already_in_list: continue
        count += 1
        if count <= 5:
            film_list.append(parametres_films[i])
    return film_list

def remove_sequels(film_selection):    
    removed_from_selection = []
    for i, film_1 in enumerate(film_selection):
        for j, film_2 in enumerate(film_selection):
            if j <= i: continue 
            if sequel(film_1[0], film_2[0]): 
                last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                removed_from_selection.append(last_film)

    film_list = [film for film in film_selection if film[0] not in removed_from_selection]
    return film_list

def find_similarities(df, id_, del_sequels=True, verbose=False):    
    if verbose: 
        print('\n' + "CONSULTA: Filmes similares ao id={} -> '{}'".format(id_, df.iloc[id_]['title']))

    liste_films = recommand(df, id_)
    parametres_films = extract_parameters(df, liste_films)
    film_selection = []
    film_selection = add_to_selection(film_selection, parametres_films)
    if del_sequels: film_selection = remove_sequels(film_selection)
    film_selection = add_to_selection(film_selection, parametres_films)

    selection_titres = []
    for i, x in enumerate(film_selection):
        selection_titres.append([x[0].replace(u'\xa0', u''), x[5]])
        if verbose: print("nº{:<2}     -> {:<50}".format(i + 1, x[0]))

    return selection_titres

def get_recommendations(title):
    data_movies = load_movies('https://movie-train.s3.amazonaws.com/tmdb_5000_movies.csv')
    data_credits = load_credits('https://movie-train.s3.amazonaws.com/tmdb_5000_credits.csv')
    print("começou predição")
    dados = convert(data_movies, data_credits)

    dados = dados.drop(['budget', 'homepage', 'gross', 'status',
                                'duration', 'production_countries',
                                'release_date', 'production_companies'], axis=1);
    dados['title_year'] = dados['title_year'].values.astype(np.int64)

#    print(dados.head(2))
    temp = dados['id']
    temp['tagline'] = dados['tagline'].fillna('')
    temp['description'] = dados['overview'] + dados['tagline']
    dados['description'] = temp['description'].fillna('')

    model_tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = model_tf.fit_transform(dados['description'])

    cosine = linear_kernel(tfidf_matrix, tfidf_matrix)

    dados = dados.reset_index()
    titles = dados['title']
    indices = pd.Series(dados.index, index=dados['title'])

    idx = indices[title]
    scores = list(enumerate(cosine[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:31]
    movie_indices = [i[0] for i in scores]
    return titles.iloc[movie_indices]
        

class Predict(LoginRequiredMixin, View):

    def get(self, request):
        set_movies = {}
        orders = Order.objects.filter(user=request.user)
        for order in orders:
            for product in order.get_all_products():
                if product.name not in set_movies:
                    set_movies[product.name] = 1
                else:
                    set_movies[product.name] += 1
        bigger = max(set_movies, key=set_movies.get)
        predictition = get_recommendations(bigger).head(5)
        x = pd.DataFrame(predictition)
        y = x.to_dict('dict')
        return render(request, 'predictition.html', {'predict': y['title'].items()}) 


pdf = Pdf.as_view()
pdf_date = PdfDate.as_view()
render_chart = RenderChart.as_view()
predict = Predict.as_view()
