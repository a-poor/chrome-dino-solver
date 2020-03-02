
import sqlite3


class DbConnect:
    def __init__(self, db_path='dino.db'):
        self.db = sqlite3.connect()
        return

    def close(self):
        self.db.close()
        return



