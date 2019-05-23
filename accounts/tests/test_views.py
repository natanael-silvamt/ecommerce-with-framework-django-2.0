from django.test import Client, TestCase
from django.urls import reverse

from accounts.models import User


class RegisterViewTestCase(TestCase):

    def setUp(self):
        self.client = Client()
        self.register_url = reverse('accounts:register')

    def test_register_error(self):
        data = {'username': 'teste', 'password1': 'teste123', 'password2': 'teste123'}
        response = self.client.post(self.register_url, data)
        self.assertFormError(response, 'form', 'email', 'Este campo é obrigatório.')
