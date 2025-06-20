# backend/database.py
import mysql.connector

def conectar_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",        # cambia si tu usuario es distinto
        password="AlexisUVG2303.",        # coloca tu contraseña si tienes
        database="vehiculos"
    )

def crear_tabla_robados():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS placas_robadas (
            id INT AUTO_INCREMENT PRIMARY KEY,
            placa VARCHAR(20) UNIQUE NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def placa_robada(placa):
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM placas_robadas WHERE placa = %s", (placa,))
    resultado = cursor.fetchone()
    conn.close()
    return resultado is not None
