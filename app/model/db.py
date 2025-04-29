import pymysql
import os
from dotenv import load_dotenv

# .env 파일에서 DB 연결 정보 로드
load_dotenv()

# DB 연결 설정
def get_db_connection():
    try:
        connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            charset='utf8mb4',  # 한글 처리 및 Unicode 지원
            cursorclass=pymysql.cursors.DictCursor  # 쿼리 결과를 딕셔너리 형태로 반환
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# UID에 해당하는 웹로그를 불러오는 함수
def fetch_user_logs(uid: int):
    connection = get_db_connection()
    if not connection:
        return None

    try:
        with connection.cursor() as cursor:
            query = """
                SELECT UID, ClickPath, TAG, CurrentPath, Event, Content, Timestamp
                FROM weblogs
                WHERE UID = %s
                ORDER BY Timestamp DESC;  -- 최신 순으로 정렬
            """
            cursor.execute(query, (uid,))
            result = cursor.fetchall()
            return result
    except Exception as e:
        print(f"Error fetching user logs: {e}")
        return None
    finally:
        connection.close()
