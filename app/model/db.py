import mysql.connector
from mysql.connector import Error
import logging
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# .env 파일에서 DB 연결 정보 로드
load_dotenv()

# DB 연결 설정
def get_db_connection():
    """데이터베이스 연결 생성"""
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'eum_classifier')
    )

# UID에 해당하는 웹로그를 불러오는 함수
def fetch_user_logs(uid: int, limit: int = None) -> List[Dict[str, Any]]:
    """특정 사용자의 웹로그 조회"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
            SELECT 
                id,
                uid,
                click_path,
                tag,
                current_path,
                event,
                content,
                timestamp
            FROM weblog
            WHERE uid = %s
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query, (uid,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def fetch_user_profile(uid: int) -> Optional[Dict[str, Any]]:
    """사용자 프로필 데이터 조회"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
            SELECT 
                uid,
                community_preference,
                info_preference,
                discussion_preference
            FROM user_profile
            WHERE uid = %s
        """
        cursor.execute(query, (uid,))
        result = cursor.fetchone()
        return result
    finally:
        cursor.close()
        conn.close()

# 모든 웹로그를 불러오는 함수
def fetch_all_logs(limit: int = None) -> List[Dict[str, Any]]:
    """전체 웹로그 조회"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
            SELECT 
                w.id,
                w.uid,
                w.click_path,
                w.tag,
                w.current_path,
                w.event,
                w.content,
                w.timestamp,
                p.community_preference,
                p.info_preference,
                p.discussion_preference
            FROM weblog w
            LEFT JOIN user_profile p ON w.uid = p.uid
            ORDER BY w.timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def check_distinct_values():
    """웹로그 테이블의 고유값 확인"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # current_path 확인
        cursor.execute("SELECT DISTINCT current_path FROM weblog")
        paths = cursor.fetchall()
        logging.info("All current_paths:")
        for path in paths:
            logging.info(path['current_path'])
            
        # click_path 확인
        cursor.execute("SELECT DISTINCT click_path FROM weblog")
        click_paths = cursor.fetchall()
        logging.info("\nAll click_paths:")
        for path in click_paths:
            logging.info(path['click_path'])
            
        # content 샘플 확인
        cursor.execute("SELECT content FROM weblog WHERE content IS NOT NULL LIMIT 5")
        contents = cursor.fetchall()
        logging.info("\nSample contents:")
        for content in contents:
            logging.info(content['content'][:100] + "..." if content['content'] else "None")
            
        # tag 확인
        cursor.execute("SELECT DISTINCT tag FROM weblog")
        tags = cursor.fetchall()
        logging.info(f"\nUnique tags ({len(tags)}):")
        for tag in tags:
            logging.info(tag['tag'])
            
        # uid 개수 확인
        cursor.execute("SELECT COUNT(DISTINCT uid) as user_count FROM weblog")
        user_count = cursor.fetchone()['user_count']
        logging.info(f"\nTotal unique users: {user_count}")
        
        return paths, click_paths, contents, tags, user_count
    except Error as e:
        logging.error(f"Error checking distinct values: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def check_data_structure():
    """데이터 구조 확인"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 샘플 데이터 확인
        cursor.execute("SELECT * FROM weblog LIMIT 5")
        sample_logs = cursor.fetchall()
        logging.info("\nSample logs:")
        for log in sample_logs:
            logging.info(f"Log: {log}")
            
        # click_path와 tag 조합 확인
        cursor.execute("""
            SELECT click_path, tag, COUNT(*) as count 
            FROM weblog 
            WHERE click_path IS NOT NULL AND tag IS NOT NULL 
            GROUP BY click_path, tag 
            LIMIT 10
        """)
        combinations = cursor.fetchall()
        logging.info("\nClick path and tag combinations:")
        for combo in combinations:
            logging.info(f"Path: {combo['click_path']}, Tag: {combo['tag']}, Count: {combo['count']}")
        
        return sample_logs, combinations
    except Error as e:
        logging.error(f"Error checking data structure: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_distinct_values()
    check_data_structure()
