from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from allauth.socialaccount.models import SocialApp
from django.contrib.sites.models import Site


class UserViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.home_url = reverse('home')
        self.logout_url = reverse('logout')
        self.main_url = reverse('main')

        self.user = User.objects.create_user(username='testuser', password='testpass')
        site = Site.objects.get_current()
        app = SocialApp.objects.create(
            provider='google',
            name='Google',
            client_id='dummy-id',
            secret='dummy-secret',
        )
        app.sites.add(site)

    def test_home_redirects_authenticated_user(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(self.home_url)
        self.assertRedirects(response, self.main_url)

    def test_home_renders_for_anonymous_user(self):
        response = self.client.get(self.home_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')

    def test_logout_redirects_to_home(self):
        self.client.login(username='testuser', password='testpass')
        response = self.client.get(self.logout_url)
        self.assertRedirects(response, self.home_url)
