import pymysql
passwords = ['', 'root', '1234', '12345', 'admin', 'mysql', 'password', '123456', 'toor', 'test']
found = False
for p in passwords:
    try:
        conn = pymysql.connect(host='localhost', user='root', password=p, connect_timeout=3)
        print(f'SUCCESS! MySQL password is: [{p}]')
        conn.close()
        found = True
        break
    except Exception as e:
        print(f'Failed [{p}]')
if not found:
    print("\nNone worked. Please set MYSQL_PASSWORD in config.py manually.")
