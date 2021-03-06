import os
import dj_database_url
from django.contrib.messages import constants as messages_constants

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'j)ta7d^!-v1mv1dr#gre$-d^+h)!zs9_7gv)mdw^n^obkv-_in'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    #libs
    'widget_tweaks',
    'easy_thumbnails',
    'jchart',

    #apps
    'core',
    'accounts',
    'catalog',
    'checkout',
    'renderpdf',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    'checkout.middleware.cart_item_middleware',
]

ROOT_URLCONF = 'djangoecommerce.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # apps
                'catalog.context_processors.categories',
            ],
        },
    },
]

WSGI_APPLICATION = 'djangoecommerce.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
#    'default': {
#        'ENGINE': 'django.db.backends.postgresql',
#        'NAME': '',
#        'USER': '',
#        'PASSWORD': '',
#        'HOST': '',
#        'PORT': ,
#    }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'pt-br'

TIME_ZONE = 'America/Fortaleza'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

#STATIC_URL = 'static/'


#MEDIA_URL = 'media/'


# Configurações para desenvolvimento
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'

STATIC_URL = '/static/'
MEDIA_URL = '/media/'

#MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
#STATIC_ROOT = os.path.join(BASE_DIR, 'static')

# THUMBNAIL_DEFAULT_STORAGE = DEFAULT_FILE_STORAGE


db_from_env = dj_database_url.config(conn_max_age=500)
DATABASES['default'].update(db_from_env)

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

ALLOWED_HOSTS = ['*']

STATIC_ROOT = os.path.join(PROJECT_ROOT, 'staticfiles')

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

#E-mail
EMAIL_HOST = ''
EMAIL_HOST_USER = ''
EMAIL_HOST_PASSWORD = ''
DEFAULT_FROM_EMAIL = 'admin@djangoecommerce.com'
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# auth
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'index'
LOGOUT_URL = 'logout'
AUTH_USER_MODEL = 'accounts.User'
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'accounts.backends.ModelBackend',
)

# Messages
MESSAGE_TAGS = {
    messages_constants.DEBUG: 'debug',
    messages_constants.INFO: 'info',
    messages_constants.SUCCESS: 'success',
    messages_constants.WARNING: 'warning',
    messages_constants.ERROR: 'danger',
}

PAGSEGURO_TOKEN = ''
PAGSEGURO_EMAIL = 'natanaelsilva.cc16@gmail.com'
PAGSEGURO_SANDBOX = True

# AWS
#STATICFILES_LOCATION = 'static'
#MEDIAFILES_LOCATION = 'media'

#AWS_S3_SECURE_URLS = True
#AWS_QUERYSTRING_AUTH = False
#AWS_PRELOAD_METADATA = True
#AWS_ACCESS_KEY_ID = ''
#AWS_SECRET_ACCESS_KEY = ''
#AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
#AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
#AWS_STORAGE_BUCKET_NAME = 'djangoecommerce'
#AWS_S3_CUSTOM_DOMAIN = '%s.s3.amazonaws.com' % AWS_STORAGE_BUCKET_NAME

#https://djangoecommerce.s3-sa-east-1.amazonaws.com/static/assets/bootstrap.min.css

#STATICFILES_STORAGE = 'djangoecommerce.s3util.StaticStorage'
#STATIC_URL = "https://djangoecommerce.s3-sa-east-1.amazonaws.com/static/"
#STATIC_URL = "https://%s/%s/" % (AWS_S3_CUSTOM_DOMAIN, MEDIAFILES_LOCATION)

#DEFAULT_FILE_STORAGE = 'djangoecommerce.s3util.MediaStorage'
#MEDIA_URL = "https://%s/%s/" % (AWS_S3_CUSTOM_DOMAIN, MEDIAFILES_LOCATION)

#AWS_HEADERS = {
#    'x-amz-acl': 'public-read',
#    'Cache-Control': 'public, max-age=31556926'
#}



# Thumbnails
THUMBNAIL_ALIASES = {
    '': {
        'product_image': {'size': (285, 160), 'crop': True}
    },
}

#THUMBNAIL_DEFAULT_STORAGE = DEFAULT_FILE_STORAGE
