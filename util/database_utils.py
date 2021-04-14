import sqlalchemy
from config import Config


class DatabaseUtils(object):
    def __init__(self):
        self.engine = sqlalchemy.create_engine(Config.DATABASE_URI)

    def get_db_engine(self):
        return self.engine
