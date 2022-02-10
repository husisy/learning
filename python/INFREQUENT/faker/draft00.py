from faker import Faker
from faker.providers import BaseProvider

fake = Faker(locale='zh_CN')
# fake = Faker(locale='en_US')
print(fake.text())
print(fake.address())
print([fake.name() for _ in range(10)])


# seed
print(fake.name())
fake.seed(233)
print(fake.name())
fake.seed(233)
print(fake.name())

# custom provider
class MyProvider(BaseProvider):
    def foo(self):
        return 'bar'
fake.add_provider(MyProvider)
print(fake.foo())


# custom the Lorem Ipsum Provider
fake = Faker(locale='en_US')
print(fake.sentence())

tmp1 = ['danish','cheesecake','sugar','Lollipop','wafer','Gummies','sesame','Jelly','beans','pie','bar','Ice','oat']
print(fake.sentence(ext_word_list=tmp1))


# date_time
fake = Faker(locale='en_US')

fake.date_this_year(before_today=True, after_today=False)
# datetime.date(2018, 1, 19)

fake.date(pattern="%Y-%m-%d", end_datetime=None)
# '1978-05-06'

fake.date_time_this_century(before_now=True, after_now=False, tzinfo=None)
# datetime.datetime(2001, 12, 24, 4, 49, 41)

fake.iso8601(tzinfo=None, end_datetime=None)
# '1998-06-09T03:27:27'

fake.future_datetime(end_date="+30d", tzinfo=None)
# datetime.datetime(2018, 9, 26, 19, 29, 50)

fake.date_this_decade(before_today=True, after_today=False)
# datetime.date(2014, 6, 27)

fake.year()
# '1981'

fake.day_of_week()
# 'Saturday'

fake.date_time_between_dates(datetime_start=None, datetime_end=None, tzinfo=None)
# datetime.datetime(2018, 9, 6, 14, 16, 50)

fake.century()
# 'XIII'

fake.am_pm()
# 'AM'

fake.time(pattern="%H:%M:%S", end_datetime=None)
# '16:46:35'

fake.date_this_month(before_today=True, after_today=False)
# datetime.date(2018, 9, 4)

fake.past_date(start_date="-30d", tzinfo=None)
# datetime.date(2018, 8, 18)

fake.date_time_between(start_date="-30y", end_date="now", tzinfo=None)
# datetime.datetime(2015, 4, 3, 23, 22, 19)

fake.future_date(end_date="+30d", tzinfo=None)
# datetime.date(2018, 9, 26)

fake.time_object(end_datetime=None)
# datetime.time(19, 55, 4)

fake.month_name()
# 'May'

fake.unix_time(end_datetime=None, start_datetime=None)
# 1512464832

fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None)
# datetime.datetime(2018, 8, 21, 4, 40, 9)

fake.day_of_month()
# '09'

fake.date_time_this_decade(before_now=True, after_now=False, tzinfo=None)
# datetime.datetime(2012, 5, 17, 11, 27, 45)

fake.timezone()
# 'Africa/Porto-Novo'

fake.date_this_century(before_today=True, after_today=False)
# datetime.date(2013, 5, 31)

fake.month()
# '11'

fake.date_time_this_month(before_now=True, after_now=False, tzinfo=None)
# datetime.datetime(2018, 9, 5, 17, 8, 10)

fake.date_between_dates(date_start=None, date_end=None)
# datetime.date(2018, 9, 6)

fake.date_of_birth(tzinfo=None, minimum_age=0, maximum_age=115)
# datetime.date(1931, 3, 16)

fake.date_object(end_datetime=None)
# datetime.date(2012, 10, 13)

fake.date_time(tzinfo=None, end_datetime=None)
# datetime.datetime(2016, 1, 21, 15, 53, 52)

fake.date_time_ad(tzinfo=None, end_datetime=None, start_datetime=None)
# datetime.datetime(1546, 7, 30, 13, 54, 59)

fake.date_between(start_date="-30y", end_date="today")
# datetime.date(2001, 9, 1)

fake.time_delta(end_datetime=None)
# datetime.timedelta(8659, 15795)

fake.past_datetime(start_date="-30d", tzinfo=None)
# datetime.datetime(2018, 8, 17, 2, 29, 35)

fake.time_series(start_date="-30d", end_date="now", precision=None, distrib=None, tzinfo=None)
# <generator object time_series at 0x7f19bb6fd990>