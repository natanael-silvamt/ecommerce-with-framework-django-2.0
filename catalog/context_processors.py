from .models import Category


def categories(request):
    response = {
        'categories': Category.objects.all()
    }
    return response