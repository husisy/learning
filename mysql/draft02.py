import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class User(Base):
    __tablename__ = 'users00'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String(50))
    fullname = sqlalchemy.Column(sqlalchemy.String(50))
    nickname = sqlalchemy.Column(sqlalchemy.String(50))

    def __repr__(self):
        tmp0 = "<User(name='{}', fullname='{}', nickname='{}')>"
        return tmp0.format(self.name, self.fullname, self.nickname)

# see https://docs.sqlalchemy.org/en/13/dialects/mysql.html#module-sqlalchemy.dialects.mysql.pymysql
engine = sqlalchemy.create_engine('mysql+pymysql://xxx-username:xxx-password@localhost:23333/test', echo=True)

Base.metadata.create_all(engine)

user00 = User(name='ed', fullname='Ed jones', nickname='edsnickname')
user00.name
user00.nickname

# see https://docs.sqlalchemy.org/en/13/orm/tutorial.html#creating-a-session
session = sessionmaker(bind=engine)() #shenmegui...

session.add(user00)
session.add_all([
    User(name='wendy', fullname='Wendy Williams', nickname='windy'),
    User(name='mary', fullname='Mary Contrary', nickname='mary'),
    User(name='fred', fullname='Fred Flintstone', nickname='freddy'),
])
tmp0 = session.query(User).filter_by(name='ed').first()
tmp0.nickname = 'eddie'
session.dirty
session.new
session.commit()
print(session.query(User).first())
print(session.query(User).all())
