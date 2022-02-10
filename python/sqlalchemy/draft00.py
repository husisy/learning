import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

TRASH_DIR = Path('tbd00').resolve()
if not TRASH_DIR.exists():
    TRASH_DIR.mkdir()
database_path = TRASH_DIR / 'example.sqlite'
if database_path.exists():
    database_path.unlink()

engine = create_engine('sqlite:///{}'.format(database_path), echo=True)
# engine = create_engine('sqlite:///:memory:', echo=True)
# engine = create_engine('sqlite:///./tbd00/example.sqlite', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)
# Session.configure(bind=engine)

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    fullname = Column(String)
    nickname = Column(String)

    def __repr__(self):
        return f"<User(name={self.name}, fullname={self.fullname}, nickname={self.nickname})>"

# User.__table__
User.metadata.create_all(engine)
session = Session()

ed_user = User(name='ed', fullname='Ed Jones', nickname='edsnickname')
ed_user.name
ed_user.nickname
ed_user.id #None

session.add(ed_user)
tmp0 = session.query(User).filter_by(name='ed').first() #query will flush first
tmp0 is ed_user #the same object, id(tmp0)==id(ed_user)
ed_user.id #1

ed_user.nickname = 'eddie'
session.dirty

session.add_all([
    User(name='wendy', fullname='Wendy Williams', nickname='windy'),
    User(name='mary', fullname='Mary Contrary', nickname='mary'),
    User(name='fred', fullname='Fred Flintstone', nickname='freddy')
])
session.new
session.commit()
session.close()


# read database using sqlite3
conn = sqlite3.connect(database_path)
cursor = conn.cursor()
print(cursor.execute('SELECT * FROM users').fetchall())
cursor.close()
conn.close()
