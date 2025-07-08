import psycopg2

DSN = dict(
    dbname='orion',
    user='prefect',
    password='pZn!cT21x^dR',
    host='192.168.191.194',
    port=9876,
)

CREATE_QUERY = '''CREATE TABLE test (
    id serial PRIMARY KEY, 
    num integer,
    data varchar);'''

DROP_QUERY = '''DROP TABLE test;'''


def execute(query: str, cur):
    print('Running query %s' % (query))
    cur.execute(query)


conn = psycopg2.connect(**DSN)
with conn.cursor() as cur:
    execute(CREATE_QUERY, cur)
    execute(DROP_QUERY, cur)
conn.commit()
