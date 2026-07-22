import pymysql

conn = pymysql.connect(host='localhost', user='root', password='@@@jeeva18ani###')
cur  = conn.cursor()

cur.execute("CREATE DATABASE IF NOT EXISTS ai_doctor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
cur.execute("USE ai_doctor")

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    full_name  VARCHAR(100) NOT NULL,
    email      VARCHAR(150) NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)""")

cur.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT NOT NULL,
    role       ENUM('user','assistant') NOT NULL,
    message    TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)""")

cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT NOT NULL,
    symptoms    TEXT NOT NULL,
    predicted   VARCHAR(200) NOT NULL,
    confidence  FLOAT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)""")

conn.commit()
cur.close()
conn.close()
print("Database ai_doctor setup complete!")
print("Tables created: users, chat_history, predictions")
