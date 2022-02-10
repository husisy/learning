# django

1. link
   * [tutorial project](https://docs.djangoproject.com/en/2.1/intro/tutorial01/)
2. `django.get_version()`

folder structure

```bash
ws00/
    manage.py
    mysite/
        __init__.py
        settings.py
        urls.py
        wsg1.py
    polls/
        __init__.py
        admin.py
        apps.py
        migrations/
            __init__.py
        models.py
        tests.py
        views.py
```

1. `django-admin startproject mysite`
2. `manage.py`: command-line utility, [detail](https://docs.djangoproject.com/en/2.1/ref/django-admin/)
3. `urls.py`: [URL dispatcher](https://docs.djangoproject.com/en/2.1/topics/http/urls/)
4. `wsgi.py`: entry-point for WSGI-compatible web server, [detail](https://docs.djangoproject.com/en/2.1/howto/deployment/wsgi/)
5. runserver
   * **DO NOT** user this tutorial server in anything resembling a production environment
   * `python manage.py runserver`, then visit `localhost:8000`
   * `python manage.py runserver 8080`, `localhost:8080`
   * `python manage.py runserver 0:8080`, `xxx.xxx.xxx.xxx:8000`
   * `python manage.py check`

settings.py

1. `settings.py` [detail](https://docs.djangoproject.com/en/2.1/topics/settings/)
2. `TIME_ZONE = 'Asia/Shanghai'`
   * [wiki - list of tz database time zones](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
3. `INSTALLED_APPS`
   * `django.contrib.admin`
   * `django.contrib.auth`
   * `django.contrib.contenttypes`
   * `django.contrib.sessions`
   * `django.contrib.messages`
   * `django.contrib.staticfiles`
   * `polls.apps.PollsConfig`
4. `ROOT_URLCONF='mysite.urls'`
5. `TEMPLATES.APP_DIRS`

app

1. `python manage.py startapp polls`

database setup

1. [database bindings](https://docs.djangoproject.com/en/2.1/topics/install/#database-installation)
2. notice
   * `CREATE DATABASE database_name`
   * user, password, host
3. setup database for `INSTALLED_APPS`
   * `python manage.py makemigrations polls`: create migrations for the change `polls/migrations/0001_initial.py`
   * `python manage.py migrate`: apply the change to the database
   * `python manage.py sqlmigrate polls 0001`: print out migration commands
   * separate these commands to make and apply migrations is to commit migrations to version control system and ship them with apps
4. `django.db.models`
   * `models.CharField()`
   * `models.DateTimeFiedl()`
   * `models.ForeighKey()`
   * `models.IntegerField()`
   * machine-friendly format and human-readable name
5. interactive shell: `python manage.py shell`
   * [database api](https://docs.djangoproject.com/en/2.1/topics/db/queries/)
6. admin
   * `python manage.py ceratesuperuser`: `Username`, `EmailAddreass`, `Password`
   * `localhost:8000/admin/`
   * `admin.site.register(Question)`
   * `Save`, `Save and continue editing`, `Save and add another`, `Delete`

webpage design

1. blog application
   * blog homepage: the latest few entries
   * entry "detail" page: permalink page for a single entry
   * year-based archive page: all months with entries in the given year
   * month-based archive page: all days with entries in the given month
   * day-based archive page
   * comment action
2. poll application
   * question index page: the latest few questions
   * question detal page: question text, with no results but with a form to vote
   * question result page: results for a particular question
   * vote action: handle voting for a particular choice in a particular question

views

1. [templates guide](https://docs.djangoproject.com/en/2.1/topics/templates/)
2. templates: `polls/templates/polls/index.html`
   * template namespacing
3. `django.shortcuts`
   * `.render()`
   * `.get_object_or_404()`: use `.get()`, to maintain loose coupling
   * `.get_list_or_404()`: use `.filter()`
4. `{% for %}` is interpreted as the Python code method-calling
5. template tag `{% url %}`
   * `href="/polls/{{ question.id }}/"` -> `href="{% url 'detail' question.id %}"`
6. Cross-Site Request Forgery (CSRF) 跨站请求伪造
   * `{% csrf_token %}`
7. **always return an `HttpResponseRedirect` after successfully dealing with POST data to prevent data from being posted twice if a user hits the back button**
8. [request and response objects](https://docs.djangoproject.com/en/2.1/ref/request-response/)
9. template tag `{{ choice.votes|pluralize}}`
10. **race conditions**: [avoiding race condition using F()](https://docs.djangoproject.com/en/2.1/ref/models/expressions/#avoiding-race-conditions-using-f)
11. `django.views.generic`: [detail](https://docs.djangoproject.com/en/2.1/topics/class-based-views/)
    * `.ListView`: `.model`, `.template_name`
    * `.DetailView`

unittest

1. `django.test.TestCase`
   * `self.assertContains()`
   * `self.assertQuerysetEqual()`
2. `python manage.py test polls`
3. `django.test.utils`
   * `.setup_test_environment()`: install a template renderer, do NOT setup a test database
4. rules of thumb
   * a seperate `TestClass` for each `model` or `view`
   * a separate test method for each set of conditions you want to test
   * test method names that describe their function
5. `LiveServerTestCase`: integration with tools like [Selenium](https://www.seleniumhq.org/)
6. [integration with coverage](https://docs.djangoproject.com/en/2.1/topics/testing/advanced/#integration-with-coverage-py)
7. [testing in django](https://docs.djangoproject.com/en/2.1/topics/testing/)
8. stylesheet and image
   * `polls/static/polls/style.css`
   * `polls/static/polls/images/xxx.png`
   * [managing static files](https://docs.djangoproject.com/en/2.1/howto/static-files/)
   * [the staticfiles app](https://docs.djangoproject.com/en/2.1/ref/contrib/staticfiles/)
   * [deploying static files](https://docs.djangoproject.com/en/2.1/howto/static-files/deployment/)

admin

1. TODO
   * [change list pagination](https://docs.djangoproject.com/en/2.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.list_per_page)
   * [search boxes](https://docs.djangoproject.com/en/2.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.search_fields)
   * [filters](https://docs.djangoproject.com/en/2.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.list_filter)
   * [date-hierarchies](https://docs.djangoproject.com/en/2.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.date_hierarchy)
   * [column-header-ordering](https://docs.djangoproject.com/en/2.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.list_display)
