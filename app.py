from flask import Flask, render_template, request, flash, redirect, url_for, session, send_file, flash
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf.csrf import CSRFProtect
from contextlib import contextmanager
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from whitenoise import WhiteNoise
from sklearn.preprocessing import LabelEncoder
import io
import pickle
import json
import os
import math
import logging
import sys
import unicodedata
import pandas as pd
import numpy as np
from functools import wraps

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:
    mysql = None
    MySQLError = Exception

try:
    import psycopg2
except ImportError:
    psycopg2 = None

# ==================== CONFIGURACIÓN ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_super_segura_123456'
app.config['WTF_CSRF_ENABLED'] = False
csrf = CSRFProtect(app)

# Agregar funciones útiles a Jinja2
app.jinja_env.globals.update(
    min=min,
    max=max,
    len=len,
    range=range,
    round=round,
    str=str,
    int=int,
    float=float
)

# Configuración de archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    flash('El archivo supera el límite de 16 MB. Guarda una copia más pequeña e inténtalo de nuevo.', 'error')
    return redirect(url_for('upload'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/profile_pics', exist_ok=True)

# ==================== CONFIGURACIÓN DE BASE DE DATOS ====================
def _safe_env_value(name, default=''):
    """Normaliza valores de entorno para evitar UnicodeDecodeError por caracteres inválidos."""
    value = os.environ.get(name, default)
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value.decode('utf-8', errors='replace')
    return str(value)


DB_CONFIG = {
    'host': _safe_env_value('DB_HOST', 'localhost'),
    'user': _safe_env_value('DB_USER', 'postgres'),
    'password': _safe_env_value('DB_PASSWORD', 'postgres'),
    'database': _safe_env_value('DB_NAME', 'cybershield'),
    'port': int(_safe_env_value('DB_PORT', '5432'))
}


def _read_dataframe_from_file(filepath):
    """Lee CSV/Excel con soporte a codificaciones comunes y fallback seguro."""
    file_path = str(filepath)
    lower_path = file_path.lower()

    if lower_path.endswith('.csv'):
        with open(file_path, 'rb') as f:
            raw = f.read()

        encodings = ['utf-8-sig', 'utf-16', 'cp1252', 'latin-1']
        last_error = None

        for encoding in encodings:
            try:
                text = raw.decode(encoding)
                return pd.read_csv(io.StringIO(text), sep=None, engine='python')
            except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
                last_error = exc
                continue

        logger.warning('No se pudo decodificar el CSV %s con codificaciones comunes; se usará fallback. Error: %s', file_path, last_error)
        return pd.read_csv(file_path, sep=None, engine='python', encoding_errors='replace')

    return pd.read_excel(file_path)


def get_database_connection():
    """Conexión compatible con PostgreSQL/Render y fallback a MySQL."""
    database_url = _safe_env_value('DATABASE_URL', '')
    db_type = _safe_env_value('DB_TYPE', 'postgresql').lower()

    if database_url:
        if psycopg2 is None:
            raise RuntimeError('psycopg2 no está instalado')
        try:
            return psycopg2.connect(database_url, sslmode='require')
        except Exception as exc:
            logger.warning('DATABASE_URL inválida, intentando fallback a variables PostgreSQL: %s', exc)

    if db_type == 'mysql' and mysql is not None:
        return mysql.connector.connect(**DB_CONFIG)

    if psycopg2 is not None:
        pg_host = _safe_env_value('PGHOST', DB_CONFIG['host'])
        pg_port = int(_safe_env_value('PGPORT', str(DB_CONFIG['port'])))
        pg_user = _safe_env_value('PGUSER', DB_CONFIG['user'])
        pg_password = _safe_env_value('PGPASSWORD', DB_CONFIG['password'])
        pg_dbname = _safe_env_value('PGDATABASE', DB_CONFIG['database'])

        try:
            return psycopg2.connect(
                host=pg_host,
                port=pg_port,
                user=pg_user,
                password=pg_password,
                dbname=pg_dbname
            )
        except Exception as exc:
            logger.warning('No se pudo conectar con PostgreSQL usando PG*; intentando MySQL si está disponible: %s', exc)

    if mysql is not None:
        return mysql.connector.connect(**DB_CONFIG)

    raise RuntimeError('No hay driver de base de datos disponible. Instala psycopg2-binary o mysql-connector-python.')


class DatabaseConnectionProxy:
    """Adaptador que hace compatible PostgreSQL con el código existente."""
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, item):
        return getattr(self._connection, item)

    def cursor(self, *args, **kwargs):
        dictionary = kwargs.pop('dictionary', False)
        if dictionary and psycopg2 is not None:
            from psycopg2.extras import RealDictCursor
            return self._connection.cursor(*args, cursor_factory=RealDictCursor, **kwargs)
        return self._connection.cursor(*args, **kwargs)


@contextmanager
def get_db():
    """Conexión a la base de datos desde PostgreSQL/Render o MySQL."""
    conn = None
    try:
        conn = DatabaseConnectionProxy(get_database_connection())
        yield conn
    except Exception as e:
        logger.error(f'Error BD: {e}')
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn is not None:
            conn.close()

# ==================== MODELO DE MACHINE LEARNING ====================
class RiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced')
        self.file_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
        self.is_trained = False
        self.file_is_trained = False
        self.le_tipo = LabelEncoder()
        self.le_impacto = LabelEncoder()
        self.le_probabilidad = LabelEncoder()

    def train_model(self):
        """Entrena el modelo con datos históricos más completos, incluyendo contexto operativo y de seguridad."""
        X_train = [
            [0, 2, 2, 1, 2, 2, 2, 2, 1, 1],
            [1, 2, 2, 2, 2, 3, 2, 2, 1, 1],
            [2, 2, 2, 2, 3, 3, 2, 2, 1, 2],
            [3, 2, 2, 3, 3, 3, 3, 2, 2, 2],
            [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 2, 1, 2, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 1, 2, 2, 2, 1, 1, 1],
            [2, 1, 2, 1, 2, 2, 2, 2, 1, 1],
            [3, 2, 2, 2, 3, 3, 3, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
            [2, 1, 1, 2, 1, 2, 2, 1, 1, 1],
            [3, 1, 2, 2, 2, 2, 2, 2, 1, 1],
            [4, 2, 2, 3, 3, 3, 3, 2, 2, 2],
            [0, 2, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 2, 1, 2, 2, 2, 2, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
            [3, 2, 2, 3, 3, 3, 3, 2, 2, 2],
            [4, 2, 2, 3, 3, 3, 3, 2, 2, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ]
        y_train = [3, 3, 3, 3, 1, 2, 2, 2, 3, 0, 1, 2, 2, 3, 1, 2, 3, 3, 3, 2]
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def train_file_model(self):
        """Entrena un modelo más completo para evaluar documentos y archivos."""
        X_train = [
            [1, 0, 0, 0, 85, 15, 1],
            [4, 1, 0, 1, 70, 28, 4],
            [7, 3, 1, 2, 58, 40, 7],
            [12, 5, 2, 3, 42, 58, 12],
            [18, 8, 4, 5, 30, 70, 18],
            [2, 0, 0, 0, 90, 10, 2],
            [6, 2, 1, 1, 76, 22, 6],
            [10, 4, 2, 2, 63, 35, 10],
            [14, 7, 3, 4, 48, 50, 15],
            [20, 10, 5, 6, 34, 66, 18],
            [28, 16, 9, 8, 12, 88, 30],
            [35, 22, 12, 10, 6, 94, 38],
        ]
        y_train = [0, 1, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3]
        self.file_model.fit(X_train, y_train)
        self.file_is_trained = True

    def _extract_file_features(self, df):
        total_rows = max(1, len(df))
        total_cols = max(1, len(df.columns))
        total_cells = total_rows * total_cols

        null_ratio = (df.isna().sum().sum() / total_cells) * 100
        empty_ratio = (df.astype(str).apply(lambda s: s.str.strip().eq('')).sum().sum() / total_cells) * 100
        duplicate_ratio = df.duplicated().mean() * 100

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_score = 0.0
        numeric_count = 0
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            numeric_count += 1
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((series < lower) | (series > upper)).mean() * 100
            outlier_score += outliers

        outlier_ratio = outlier_score / max(1, numeric_count)
        numeric_ratio = (len(numeric_cols) / total_cols) * 100
        string_ratio = 100 - numeric_ratio
        inconsistency_ratio = float(df.astype(str).apply(lambda s: s.str.contains(r'[^0-9.,-]', case=False, na=False)).sum().sum()) / max(1, total_cells) * 100

        return [
            null_ratio,
            empty_ratio,
            duplicate_ratio,
            outlier_ratio,
            numeric_ratio,
            string_ratio,
            inconsistency_ratio,
        ]

    def predict_file_risk(self, df):
        """Calcula riesgo de un archivo basado en calidad y anomalías."""
        if not self.file_is_trained:
            self.train_file_model()

        features = self._extract_file_features(df)
        prediction = self.file_model.predict([features])[0]
        proba = self.file_model.predict_proba([features])[0]

        nivel_map = {0: 'Bajo', 1: 'Medio', 2: 'Alto', 3: 'Crítico'}
        return nivel_map[prediction], proba

    def predict_risk(self, tipo_riesgo, impacto, probabilidad, sector=None, tamano=None,
                    historial_incidentes=None, sensibilidad_datos=None,
                    controles_seguridad=None, copias_respaldo=None, mfa=None):
        """Predice el nivel de riesgo usando ML con contexto operacional y de seguridad."""
        if not self.is_trained:
            self.train_model()

        tipo_map = {'Tecnológico': 0, 'Financiero': 1, 'Operativo': 2, 'Reputacional': 3, 'Legal': 4, 'Comercial': 5}
        impacto_map = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
        prob_map = {'Baja': 0, 'Media': 1, 'Alta': 2}
        sector_map = {'Tecnología': 0, 'Financiero': 1, 'Salud': 2, 'Retail': 3, 'Manufactura': 4, 'Servicios': 5, 'Educación': 6, 'Gobierno': 7, 'Otro': 8}
        tamano_map = {'Micro': 0, 'Pequeña': 1, 'Mediana': 2, 'Grande': 3}
        historial_map = {'Nunca': 0, 'Única vez': 1, 'Varios': 2, 'Frecuentes': 3}
        sensibilidad_map = {'Baja': 0, 'Media': 1, 'Alta': 2}
        controles_map = {'Fuertes': 0, 'Medios': 1, 'Bajos': 2}
        respaldo_map = {'Diarias': 0, 'Regulares': 1, 'Inconsistentes': 2}
        mfa_map = {'Total': 0, 'Parcial': 1, 'Ninguna': 2}

        tipo_encoded = tipo_map.get(tipo_riesgo, 0)
        impacto_encoded = impacto_map.get(impacto, 0)
        prob_encoded = prob_map.get(probabilidad, 0)
        sector_encoded = sector_map.get(sector, sector_map.get('Otro', 8))
        tamano_encoded = tamano_map.get(tamano, tamano_map.get('Micro', 0))
        historial_encoded = historial_map.get(historial_incidentes, historial_map.get('Nunca', 0))
        sensibilidad_encoded = sensibilidad_map.get(sensibilidad_datos, sensibilidad_map.get('Media', 1))
        controles_encoded = controles_map.get(controles_seguridad, controles_map.get('Medios', 1))
        respaldo_encoded = respaldo_map.get(copias_respaldo, respaldo_map.get('Regulares', 1))
        mfa_encoded = mfa_map.get(mfa, mfa_map.get('Parcial', 1))

        features = [
            tipo_encoded, impacto_encoded, prob_encoded,
            sector_encoded, tamano_encoded, historial_encoded,
            sensibilidad_encoded, controles_encoded, respaldo_encoded, mfa_encoded
        ]

        prediction = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]

        nivel_map = {0: 'Bajo', 1: 'Medio', 2: 'Alto', 3: 'Crítico'}
        return nivel_map[prediction], proba

# Instancia global del predictor
risk_predictor = RiskPredictor()

# ==================== CONFIGURACIÓN DE PLANES ====================
PLANES = {
    'lite': {
        'nombre': 'Plan Lite',
        'diagnosticos_mes': 10,
        'archivos_mes': 5,
        'reportes_pdf': True,
        'precio': 29900
    },
    'premium': {
        'nombre': 'Plan Premium',
        'diagnosticos_mes': 999999,
        'archivos_mes': 50,
        'reportes_pdf': True,
        'precio': 49900
    }
}

# ==================== FUNCIONES AUXILIARES ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_by_id(user_id):
    """Obtiene usuario por ID"""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE id = %s", (user_id,))
        return cursor.fetchone()

def get_user_stats(user_id):
    """Obtiene estadísticas del usuario."""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) as total FROM diagnosticos WHERE usuario_id = %s", (user_id,))
        diagnosticos = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) as total FROM archivos_analizados WHERE usuario_id = %s", (user_id,))
        archivos = cursor.fetchone()['total']

        cursor.execute("""
            SELECT COUNT(*) as total FROM diagnosticos
            WHERE usuario_id = %s AND EXTRACT(MONTH FROM fecha_analisis) = EXTRACT(MONTH FROM CURRENT_DATE)
        """, (user_id,))
        diagnosticos_mes = cursor.fetchone()['total']

        return {
            'diagnosticos_realizados': diagnosticos,
            'archivos_analizados': archivos,
            'diagnosticos_este_mes': diagnosticos_mes
        }

def calcular_diagnosticos_restantes(user_id):
    """Calcula diagnósticos restantes este mes."""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            return 0

        if user['rol'] == 'admin':
            return 'Ilimitados'

        plan = PLANES.get(user['plan'], PLANES['lite'])

        if plan['diagnosticos_mes'] == 999999:
            return 'Ilimitados'

        cursor.execute("""
            SELECT COUNT(*) as total FROM diagnosticos
            WHERE usuario_id = %s AND EXTRACT(MONTH FROM fecha_analisis) = EXTRACT(MONTH FROM CURRENT_DATE)
        """, (user_id,))
        diagnosticos_mes = cursor.fetchone()['total']

        restantes = plan['diagnosticos_mes'] - diagnosticos_mes
        return max(0, restantes)

# ==================== DECORADORES ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Debes iniciar sesión', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('rol') != 'admin':
            flash('Acceso denegado: Solo administradores', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== FUNCIONES DE USUARIO ====================
def get_user_by_username(username):
    """Obtiene usuario por username"""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE username = %s AND activo = TRUE", (username,))
        return cursor.fetchone()

def verificar_limites(user_id, tipo):
    """Verifica si el usuario puede realizar una acción."""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            return False, "Usuario no encontrado"

        if user['rol'] == 'admin':
            return True, "Acceso completo"

        plan = PLANES.get(user['plan'], PLANES['lite'])

        if tipo == 'diagnostico':
            limite = plan['diagnosticos_mes']
            usado = user.get('diagnosticos_este_mes', 0)
        elif tipo == 'archivo':
            limite = plan['archivos_mes']
            usado = user.get('archivos_este_mes', 0)
        else:
            return False, "Tipo no válido"

        if limite in (None, float('inf'), 999999):
            return True, "Ilimitado"

        if usado >= limite:
            return False, f"Límite alcanzado ({usado}/{limite})"

        return True, f"Disponible ({usado}/{limite})"

def incrementar_contador(user_id, tipo):
    """Incrementa el contador de uso"""
    with get_db() as conn:
        cursor = conn.cursor()
        if tipo == 'diagnostico':
            cursor.execute("""
                UPDATE usuarios 
                SET diagnosticos_realizados = diagnosticos_realizados + 1,
                    diagnosticos_este_mes = diagnosticos_este_mes + 1
                WHERE id = %s
            """, (user_id,))
        elif tipo == 'archivo':
            cursor.execute("""
                UPDATE usuarios 
                SET archivos_analizados = archivos_analizados + 1,
                    archivos_este_mes = archivos_este_mes + 1
                WHERE id = %s
            """, (user_id,))
        conn.commit()

# ==================== FUNCIONES DE ANÁLISIS ====================
def analizar_riesgo(form_data):
    """Calcula una evaluación explicable de riesgo inherente y residual (no es un modelo predictivo)."""
    required_values = {
        'sector': {'Tecnología', 'Financiero', 'Salud', 'Retail', 'Manufactura', 'Servicios', 'Educación', 'Gobierno', 'Otro'},
        'tamano': {'Micro', 'Pequeña', 'Mediana', 'Grande'},
        'dependencia_sistemas': {'baja', 'parcial', 'alta', 'no_se'},
        'tiempo_recuperacion': {'mas_3_dias', '1_3_dias', 'menos_1_dia', 'no_puede_operar', 'no_se'},
        'proveedores': {'ninguno', 'algunos', 'varios', 'no_se'},
        'tipo_datos': {'ninguno', 'contacto', 'financiera', 'sensible', 'confidencial', 'no_se'},
        'acceso_remoto': {'no', 'controlado', 'debil', 'no_se'},
        'servicios_nube': {'no', 'controlados', 'desconocidos', 'no_se'},
        'historial_incidentes': {'ninguno', 'uno', 'multiples', 'no_se'},
        'mfa': {'todos', 'algunos', 'ninguno', 'no_se'},
        'gestion_cuentas': {'ambas', 'parcial', 'ninguna', 'no_se'},
        'actualizaciones': {'definidas', 'ocasionales', 'raras', 'no_se'},
        'proteccion_equipos': {'todos', 'algunos', 'ninguno', 'no_se'},
        'copias_respaldo': {'definidas', 'ocasionales', 'ninguna', 'no_se'},
        'prueba_respaldo': {'exitosa', 'problemas', 'no', 'no_se'},
        'aislamiento_respaldo': {'si', 'parcial', 'no', 'no_se'},
        'respuesta_incidentes': {'documentado', 'informal', 'ninguno', 'no_se'},
        'capacitacion': {'periodica', 'ocasional', 'nunca', 'no_se'},
    }
    for field, allowed_values in required_values.items():
        if form_data.get(field) not in allowed_values:
            raise ValueError('Completa todas las preguntas del diagnóstico con una opción válida.')
    if not form_data.get('empresa', '').strip():
        raise ValueError('El nombre de la empresa es obligatorio.')

    probability_factors = {
        'historial_incidentes': {'ninguno': 1, 'uno': 3, 'multiples': 5, 'no_se': 3},
        'acceso_remoto': {'no': 1, 'controlado': 2, 'debil': 5, 'no_se': 3},
        'servicios_nube': {'no': 1, 'controlados': 2, 'desconocidos': 4, 'no_se': 3},
        'proveedores': {'ninguno': 1, 'algunos': 3, 'varios': 4},
    }
    likelihood = int(sum(mapping[form_data[key]] for key, mapping in probability_factors.items()) / len(probability_factors) + 0.5)

    impact_factors = {
        'dependencia_sistemas': {'baja': 1, 'parcial': 3, 'alta': 5, 'no_se': 3},
        'tiempo_recuperacion': {'mas_3_dias': 1, '1_3_dias': 2, 'menos_1_dia': 4, 'no_puede_operar': 5, 'no_se': 3},
        'tipo_datos': {'ninguno': 1, 'contacto': 2, 'financiera': 4, 'sensible': 5, 'confidencial': 4, 'no_se': 3},
    }
    impact = int(sum(mapping[form_data[key]] for key, mapping in impact_factors.items()) / len(impact_factors) + 0.5)

    control_factors = {
        'mfa': {'todos': 1, 'algunos': 0.5, 'ninguno': 0, 'no_se': 0},
        'gestion_cuentas': {'ambas': 1, 'parcial': 0.5, 'ninguna': 0, 'no_se': 0},
        'actualizaciones': {'definidas': 1, 'ocasionales': 0.5, 'raras': 0, 'no_se': 0},
        'proteccion_equipos': {'todos': 1, 'algunos': 0.5, 'ninguno': 0, 'no_se': 0},
        'copias_respaldo': {'definidas': 1, 'ocasionales': 0.5, 'ninguna': 0, 'no_se': 0},
        'prueba_respaldo': {'exitosa': 1, 'problemas': 0.5, 'no': 0, 'no_se': 0},
        'aislamiento_respaldo': {'si': 1, 'parcial': 0.5, 'no': 0, 'no_se': 0},
        'respuesta_incidentes': {'documentado': 1, 'informal': 0.5, 'ninguno': 0, 'no_se': 0},
        'capacitacion': {'periodica': 1, 'ocasional': 0.5, 'nunca': 0, 'no_se': 0},
    }
    control_effectiveness = sum(mapping[form_data[key]] for key, mapping in control_factors.items()) / len(control_factors)
    inherent_score = likelihood * impact * 4
    residual_score = round(inherent_score * (1 - 0.6 * control_effectiveness))
    puntuacion_final = min(100, residual_score)

    if puntuacion_final >= 75:
        nivel, color = 'Crítico', 'danger'
    elif puntuacion_final >= 50:
        nivel, color = 'Alto', 'warning'
    elif puntuacion_final >= 25:
        nivel, color = 'Medio', 'info'
    else:
        nivel, color = 'Bajo', 'success'

    level_map = {1: 'Baja', 2: 'Baja', 3: 'Media', 4: 'Alta', 5: 'Alta'}
    impact_level_map = {1: 'Bajo', 2: 'Bajo', 3: 'Medio', 4: 'Alto', 5: 'Alto'}
    if form_data['tipo_datos'] == 'financiera':
        tipo_riesgo = 'Financiero'
    elif form_data['tipo_datos'] in {'sensible', 'contacto'}:
        tipo_riesgo = 'Legal'
    elif form_data['dependencia_sistemas'] == 'alta':
        tipo_riesgo = 'Operativo'
    else:
        tipo_riesgo = 'Tecnológico'

    findings = []
    recommendations = []
    recommendation_map = [
        ('mfa', {'algunos', 'ninguno', 'no_se'}, 'Habilitar MFA en todas las cuentas, empezando por correo y cuentas administrativas.'),
        ('gestion_cuentas', {'parcial', 'ninguna', 'no_se'}, 'Asignar cuentas individuales y retirar accesos al cambiar funciones o salir de la empresa.'),
        ('actualizaciones', {'ocasionales', 'raras', 'no_se'}, 'Establecer un proceso periódico para actualizar sistemas y aplicaciones.'),
        ('proteccion_equipos', {'algunos', 'ninguno', 'no_se'}, 'Verificar que todos los equipos de trabajo tengan protección contra malware.'),
        ('copias_respaldo', {'ocasionales', 'ninguna', 'no_se'}, 'Definir y automatizar copias de seguridad de la información crítica.'),
        ('prueba_respaldo', {'problemas', 'no', 'no_se'}, 'Probar periódicamente la restauración de copias y corregir los fallos encontrados.'),
        ('aislamiento_respaldo', {'parcial', 'no', 'no_se'}, 'Mantener copias protegidas y separadas de los sistemas principales.'),
        ('respuesta_incidentes', {'informal', 'ninguno', 'no_se'}, 'Documentar un procedimiento para reportar, contener y atender incidentes.'),
        ('capacitacion', {'ocasional', 'nunca', 'no_se'}, 'Capacitar periódicamente al personal para reconocer y reportar amenazas.'),
    ]
    labels = {
        'mfa': 'MFA', 'gestion_cuentas': 'Gestión de cuentas',
        'actualizaciones': 'Actualizaciones', 'proteccion_equipos': 'Protección de equipos',
        'copias_respaldo': 'Copias de respaldo', 'prueba_respaldo': 'Pruebas de recuperación',
        'aislamiento_respaldo': 'Aislamiento de respaldos',
        'respuesta_incidentes': 'Respuesta a incidentes', 'capacitacion': 'Capacitación',
        'acceso_remoto': 'Acceso remoto', 'servicios_nube': 'Servicios en la nube',
        'proveedores': 'Dependencia de proveedores',
        'dependencia_sistemas': 'Dependencia de sistemas',
        'tiempo_recuperacion': 'Continuidad operativa',
        'tipo_datos': 'Protección de información',
    }
    priority = 'URGENTE' if nivel == 'Crítico' else 'Alta' if nivel == 'Alto' else 'Media'

    exposure_recommendations = [
        ('acceso_remoto', {'debil', 'no_se'}, 'Revisar cómo se habilita el acceso remoto y protegerlo con MFA y mecanismos seguros.'),
        ('servicios_nube', {'desconocidos', 'no_se'}, 'Inventariar los servicios en la nube y revisar sus permisos, autenticación y opciones de recuperación.'),
        ('proveedores', {'varios'}, 'Identificar proveedores críticos y confirmar sus medidas de seguridad y continuidad.'),
        ('dependencia_sistemas', {'alta'}, 'Definir una alternativa operativa para mantener procesos críticos durante interrupciones.'),
        ('tiempo_recuperacion', {'no_puede_operar'}, 'Preparar un plan de continuidad para reducir la interrupción de la operación.'),
        ('tipo_datos', {'financiera', 'sensible'}, 'Limitar el acceso a información financiera o sensible y revisar su protección y conservación.'),
    ]
    for field, relevant_values, recommendation in exposure_recommendations:
        if form_data[field] in relevant_values:
            findings.append(labels[field])
            recommendations.append({'prioridad': priority, 'titulo': recommendation})

    for field, weak_values, recommendation in recommendation_map:
        value = form_data[field]
        if value in weak_values:
            findings.append(labels[field])
            recommendations.append({
                'prioridad': priority,
                'titulo': recommendation,
            })

    if form_data['historial_incidentes'] in {'uno', 'multiples'}:
        recommendations.insert(0, {
            'prioridad': 'URGENTE' if form_data['historial_incidentes'] == 'multiples' else 'Alta',
            'titulo': 'Revisar incidentes anteriores, documentar sus causas y verificar que las medidas correctivas sigan activas.',
        })
    if not recommendations:
        recommendations.append({
            'prioridad': 'Media',
            'titulo': 'Mantener los controles actuales y revisar periódicamente que continúen funcionando.',
        })
    unknown_count = sum(form_data[field] == 'no_se' for field in required_values)
    answered_fields = len(required_values) - unknown_count
    completeness = round(answered_fields / len(required_values) * 100)

    return {
        'empresa': form_data['empresa'].strip(),
        'tipo_riesgo': tipo_riesgo,
        'impacto': impact_level_map[impact],
        'probabilidad': level_map[likelihood],
        'puntuacion_final': puntuacion_final,
        'nivel_riesgo': nivel,
        'color_riesgo': color,
        'recomendaciones': recommendations,
        'hallazgos': findings,
        'puntuacion_inherente': inherent_score,
        'efectividad_controles': round(control_effectiveness * 100),
        'completitud': completeness,
        'likelihood': likelihood,
        'impact': impact,
        'observaciones': form_data.get('observaciones', ''),
    }

def obtener_datos_formulario(form):
    """Extrae las respuestas del nuevo formulario estructurado."""
    fields = (
        'sector', 'tamano', 'dependencia_sistemas', 'tiempo_recuperacion',
        'proveedores', 'tipo_datos', 'acceso_remoto', 'servicios_nube',
        'historial_incidentes', 'mfa', 'gestion_cuentas', 'actualizaciones',
        'proteccion_equipos', 'copias_respaldo', 'prueba_respaldo',
        'aislamiento_respaldo', 'respuesta_incidentes', 'capacitacion',
    )
    data = {field: form.get(field, '') for field in fields}
    data.update({
        'empresa': form.get('empresa', '').strip(),
        'observaciones': form.get('observaciones', '').strip()[:1000],
    })
    return data

def check_password_hash_custom(pwhash, password):
    """Verificar contraseña manejando hashes vacíos o inválidos"""
    if not pwhash or pwhash.strip() == '':
        return False
    
    try:
        return check_password_hash(pwhash, password)
    except ValueError as e:
        print(f"Error verificando hash: {e}")
        return False

# ==================== RUTAS PRINCIPALES ====================

@app.route('/')
def index():
    user_info = None
    if 'user_id' in session:
        user_info = {
            'username': session['username'],
            'empresa': session['empresa'],
            'rol': session['rol'],
            'plan': session['plan']
        }
    return render_template('index.html', user=user_info)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = get_user_by_username(username)
        
        if user and check_password_hash_custom(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['empresa'] = user['empresa']
            session['rol'] = user['rol']
            session['plan'] = user['plan']
            
            flash(f'Bienvenido {user["username"]}!', 'success')
            
            # REDIRECCIÓN AUTOMÁTICA PARA ADMINISTRADORES
            if user['rol'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos', 'error')
    
    return render_template('login.html')

@app.route('/fix-passwords')
def fix_passwords():
    """Corregir contraseñas problemáticas"""
    usuarios_problematicos = [
        {'username': 'admin', 'password': 'admin123'},
        {'username': 'demo', 'password': 'demo123'},
        {'username': 'premium', 'password': 'premium123'}
    ]
    
    resultados = []
    
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        
        for usuario in usuarios_problematicos:
            cursor.execute("SELECT id, password FROM usuarios WHERE username = %s", (usuario['username'],))
            user_data = cursor.fetchone()
            
            if user_data:
                # Verificar si el hash es válido
                hash_valido = True
                try:
                    if user_data['password'] and user_data['password'].strip():
                        check_password_hash(user_data['password'], 'test')
                    else:
                        hash_valido = False
                except:
                    hash_valido = False
                
                if not hash_valido:
                    # Corregir el hash
                    nuevo_hash = generate_password_hash(usuario['password'], method='pbkdf2:sha256')
                    cursor.execute("UPDATE usuarios SET password = %s WHERE username = %s", 
                                (nuevo_hash, usuario['username']))
                    resultados.append(f"✅ {usuario['username']} - Contraseña corregida")
                else:
                    resultados.append(f"✅ {usuario['username']} - Hash válido")
            else:
                resultados.append(f"❌ {usuario['username']} - No encontrado")
        
        conn.commit()
    
    return "<br>".join(resultados) + "<br><br><a href='/login'>Ir a Login</a>"

@app.route('/crear-admin')
def crear_admin():
    """Ruta temporal para crear usuario admin"""
    username = 'admin'
    password = 'admin123'
    email = 'admin@cybershield.com'
    empresa = 'CyberShield Admin'
    
    # Verificar si ya existe
    user_existente = get_user_by_username(username)
    
    if user_existente:
        # Si existe pero tiene hash problemático, corregirlo
        try:
            check_password_hash(user_existente['password'], 'test')
            return """
            <div class='container mt-5'>
                <div class='alert alert-warning'>
                    <h4>⚠️ El usuario admin ya existe</h4>
                    <p>Usa estas credenciales:</p>
                    <p><strong>Usuario:</strong> admin</p>
                    <p><strong>Contraseña:</strong> admin123</p>
                </div>
                <a href='/login' class='btn btn-primary'>Ir a Login</a>
            </div>
            """
        except:
            # Hash inválido, corregirlo
            hashed = generate_password_hash(password, method='pbkdf2:sha256')
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE usuarios SET password = %s WHERE username = %s", 
                             (hashed, username))
                conn.commit()
            return "✅ Admin corregido. <a href='/login'>Ir a Login</a>"
    
    # Crear nuevo admin
    hashed = generate_password_hash(password, method='pbkdf2:sha256')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO usuarios (username, password, email, empresa, rol, plan, activo)
            VALUES (%s, %s, %s, %s, 'admin', 'premium', TRUE)
        """, (username, hashed, email, empresa))
        conn.commit()
    
    return """
    <div class='container mt-5'>
        <div class='alert alert-success'>
            <h4>✅ Usuario administrador creado</h4>
            <p><strong>Usuario:</strong> admin</p>
            <p><strong>Contraseña:</strong> admin123</p>
        </div>
        <a href='/login' class='btn btn-primary'>Ir a Login</a>
    </div>
    """

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        email = request.form.get('email')
        empresa = request.form.get('empresa')
        plan = request.form.get('plan', 'lite')
        
        if password != confirm:
            flash('Las contraseñas no coinciden', 'error')
            return redirect(url_for('register'))
        
        if get_user_by_username(username):
            flash('El usuario ya existe', 'error')
            return redirect(url_for('register'))
        
        # VERSIÓN CORREGIDA - Especificar el método explícitamente
        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO usuarios (username, password, email, empresa, rol, plan)
                VALUES (%s, %s, %s, %s, 'pyme', %s)
            """, (username, hashed, email, empresa, plan))
            conn.commit()
        
        flash('Cuenta creada exitosamente!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', planes=PLANES)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('Has cerrado sesión', 'success')
    return redirect(url_for('index'))

@app.route('/diagnostico', methods=['GET', 'POST'])
def diagnostico():
    # Modo invitado
    if 'user_id' not in session:
        if 'guest_diagnosticos' not in session:
            session['guest_diagnosticos'] = 0
        
        if request.method == 'POST':
            if session['guest_diagnosticos'] >= 2:
                flash('Límite de invitado alcanzado (2/2). Regístrate para continuar.', 'warning')
                return redirect(url_for('register'))
            form_data = obtener_datos_formulario(request.form)
            try:
                resultado = analizar_riesgo(form_data)
            except ValueError as exc:
                flash(str(exc), 'error')
                return render_template(
                    'diagnostico.html',
                    modo='invitado',
                    restantes=2-session.get('guest_diagnosticos', 0)
                ), 400

            session['guest_diagnosticos'] += 1
            resultado['modo'] = 'invitado'
            resultado['restantes'] = 2 - session['guest_diagnosticos']
            
            return render_template('resultado.html', resultado=resultado, now=datetime.now())
        
        return render_template('diagnostico.html', modo='invitado', restantes=2-session.get('guest_diagnosticos', 0))
    
    # Modo registrado
    puede, mensaje = verificar_limites(session['user_id'], 'diagnostico')
    
    if request.method == 'POST':
        if not puede:
            flash(mensaje, 'error')
            return redirect(url_for('diagnostico'))
        
        form_data = obtener_datos_formulario(request.form)
        try:
            resultado = analizar_riesgo(form_data)
        except ValueError as exc:
            flash(str(exc), 'error')
            return render_template(
                'diagnostico.html',
                modo='registrado',
                mensaje=mensaje
            ), 400

        form_data['tipo_riesgo'] = resultado['tipo_riesgo']
        form_data['impacto'] = resultado['impacto']
        form_data['probabilidad'] = resultado['probabilidad']
        
        # Guardar en BD
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO diagnosticos 
                (usuario_id, empresa, tipo_riesgo, impacto, probabilidad, observaciones, puntuacion_final, nivel_riesgo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (session['user_id'], form_data['empresa'], form_data['tipo_riesgo'],
                form_data['impacto'], form_data['probabilidad'], form_data['observaciones'],
                resultado['puntuacion_final'], resultado['nivel_riesgo']))
            diagnostico_id = cursor.fetchone()[0]

            for rec in resultado['recomendaciones']:
                cursor.execute("""
                    INSERT INTO recomendaciones (diagnostico_id, titulo, prioridad)
                    VALUES (%s, %s, %s)
                """, (diagnostico_id, rec['titulo'], rec['prioridad']))

            conn.commit()
        
        incrementar_contador(session['user_id'], 'diagnostico')
        resultado['modo'] = 'registrado'
        
        return render_template('resultado.html', resultado=resultado, now=datetime.now())
    
    return render_template('diagnostico.html', modo='registrado', mensaje=mensaje)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    puede, mensaje = verificar_limites(session['user_id'], 'archivo')
    
    if request.method == 'POST':
        if not puede:
            flash(mensaje, 'error')
            return redirect(url_for('upload'))
        
        if 'file' not in request.files:
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('upload'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No se seleccionó archivo', 'error')
            return redirect(url_for('upload'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = _read_dataframe_from_file(filepath)
                
                analisis = analizar_archivo(df)
                analisis['archivo_nombre'] = filename
                
                # Enriquecer con datos de visualización
                analisis = enriquecer_analisis_con_vista_tabla(analisis, df)
                
                # Guardar en BD - CORREGIDO: Asegurar que nivel_riesgo esté definido
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO archivos_analizados 
                        (usuario_id, archivo_nombre, total_registros, puntuacion_riesgo, nivel_riesgo)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        session['user_id'], 
                        filename, 
                        analisis['total_registros'],
                        analisis['puntuacion_riesgo'], 
                        analisis.get('nivel_riesgo', 'Bajo')  # Valor por defecto
                    ))
                    conn.commit()
                
                incrementar_contador(session['user_id'], 'archivo')
                
                return render_template('upload.html', 
                                    mensaje=mensaje, 
                                    analisis=analisis)
                
            except Exception as e:
                logger.error(f"Error al analizar archivo: {str(e)}")
                flash(f'Error al analizar archivo: {str(e)}', 'error')
                return redirect(url_for('upload'))
        else:
            flash('Tipo de archivo no permitido. Use CSV o Excel.', 'error')
            return redirect(url_for('upload'))
    
    return render_template('upload.html', mensaje=mensaje)

###########**NUEVA FUNCIÓN: Enriquecer el análisis con datos para la tabla**
def enriquecer_analisis_con_vista_tabla(analisis, df):
    """Agrega datos necesarios para mostrar la tabla con MAPA DE CALOR"""
    analisis['columnas'] = df.columns.tolist()
    analisis['total_registros'] = len(df)
    analisis['total_columnas'] = len(df.columns)
    
    # Vista previa de los datos (primeras 50 filas)
    preview_data = df.head(50).fillna('N/A')
    analisis['datos_preview'] = preview_data.to_dict('records')
    
    # Estadísticas por columna
    analisis['estadisticas_columnas'] = calcular_estadisticas_columnas(df)
    
    # ========== NUEVO: MAPA DE CALOR DE ANOMALÍAS ==========
    analisis['mapa_calor'] = generar_mapa_calor(df, analisis['anomalias'])
    analisis['celdas_marcadas_preview'] = sum(
        level > 0
        for row in analisis['mapa_calor']
        for level in row.values()
    )
    
    # ========== NUEVO: DATOS PARA GRÁFICAS CORREGIDAS ==========
    analisis['graficas'] = {
        'anomalias_por_tipo': generar_datos_grafica_anomalias(analisis['anomalias']),
        'nulos_por_columna': generar_datos_grafica_nulos(df),
        'distribucion_valores': generar_datos_distribucion(df)
    }
    
    return analisis

# ==================== NUEVAS FUNCIONES: MAPA DE CALOR ====================
def generar_mapa_calor(df, anomalias):
    """Marca en la vista previa las celdas que corresponden a alertas detectadas."""
    preview = df.head(50)
    levels = pd.DataFrame(0, index=preview.index, columns=preview.columns, dtype=int)
    families = {anomaly.get('familia') for anomaly in anomalias}

    def mark(mask, columns, level):
        for column in columns:
            if column in levels.columns:
                selected = mask.reindex(levels.index, fill_value=False)
                levels.loc[selected, column] = levels.loc[selected, column].clip(lower=level)

    if 'celdas_vacias' in families:
        for column in df.columns:
            mark(df[column].isna() | df[column].astype(str).str.strip().eq(''), [column], 3)

    if 'filas_repetidas' in families:
        duplicate_rows = df.duplicated(keep=False)
        mark(duplicate_rows, df.columns, 2)

    if 'valores_atipicos' in families:
        columns = {
            item['columna']
            for anomaly in anomalias if anomaly.get('familia') == 'valores_atipicos'
            for item in anomaly.get('columnas_afectadas', [])
        }
        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                values = df[column]
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                mark((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr), [column], 2)

    if 'importes_redondos' in families:
        columns = {
            item['columna']
            for anomaly in anomalias if anomaly.get('familia') == 'importes_redondos'
            for item in anomaly.get('columnas_afectadas', [])
        }
        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                values = df[column]
                rounded = values.notna() & (
                    values.mod(100).eq(0) | values.mod(500).eq(0) | values.mod(1000).eq(0)
                )
                mark(rounded, [column], 1)

    if 'cambios_importe' in families:
        columns = {
            item['columna']
            for anomaly in anomalias if anomaly.get('familia') == 'cambios_importe'
            for item in anomaly.get('columnas_afectadas', [])
        }
        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                differences = df[column].dropna().diff().abs()
                mean, std = differences.mean(), differences.std()
                mark(differences > mean + 3 * std, [column], 1)

    if 'numeracion_comprobantes' in families:
        columns = {
            item['columna']
            for anomaly in anomalias if anomaly.get('familia') == 'numeracion_comprobantes'
            for item in anomaly.get('columnas_afectadas', [])
        }
        for column in columns:
            sequence = pd.to_numeric(df[column], errors='coerce')
            ordered = sequence.dropna().sort_values()
            gap_values = ordered[ordered.diff() > 1]
            gap_rows = sequence.isin(gap_values)
            mark(gap_rows, [column], 1)

    if 'balance_contable' in families:
        debit_columns = [
            column for column in df.columns
            if any(term in normalizar_nombre_columna(column) for term in ('debito', 'debit', 'debe'))
        ]
        credit_columns = [
            column for column in df.columns
            if any(term in normalizar_nombre_columna(column) for term in ('credito', 'credit', 'haber'))
        ]
        mark(pd.Series(True, index=df.index), debit_columns + credit_columns, 2)

    return levels.to_dict(orient='records')

def generar_datos_grafica_anomalias(anomalias):
    """Prepara datos para gráfica de anomalías"""
    categorias = {}
    for anomalia in anomalias:
        categoria = anomalia.get('categoria', 'Otros')
        categorias[categoria] = categorias.get(categoria, 0) + 1
    
    return {
        'labels': list(categorias.keys()),
        'data': list(categorias.values())
    }

def generar_datos_grafica_nulos(df):
    """Prepara datos para gráfica de valores nulos"""
    nulos = df.isnull().sum()
    columnas_con_nulos = [(col, count) for col, count in nulos.items() if count > 0]
    columnas_con_nulos.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'labels': [col for col, _ in columnas_con_nulos[:10]],  # Top 10
        'data': [count for _, count in columnas_con_nulos[:10]]
    }

def generar_datos_distribucion(df):
    """Genera datos de distribución para columnas numéricas"""
    distribucion = {}
    
    for col in df.select_dtypes(include=[np.number]).columns[:5]:  # Top 5
        if len(df[col].dropna()) > 0:
            # Crear histograma
            counts, bins = np.histogram(df[col].dropna(), bins=10)
            distribucion[col] = {
                'bins': [f"{bins[i]:.1f}" for i in range(len(bins)-1)],
                'counts': counts.tolist()
            }
    
    return distribucion

def calcular_estadisticas_columnas(df):
    """
    Calcula estadísticas descriptivas para cada columna
    """
    stats = {}
    for col in df.columns:
        col_data = df[col]
        nulos = col_data.isnull().sum()
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats[col] = {
                'tipo': 'numérico',
                'promedio': round(col_data.mean(), 2) if nulos < len(col_data) else 0,
                'minimo': round(col_data.min(), 2) if nulos < len(col_data) else 0,
                'maximo': round(col_data.max(), 2) if nulos < len(col_data) else 0,
                'nulos': nulos,
                'porcentaje_nulos': round((nulos / len(col_data)) * 100, 2)
            }
        else:
            stats[col] = {
                'tipo': 'texto',
                'valores_unicos': col_data.nunique(),
                'nulos': nulos,
                'porcentaje_nulos': round((nulos / len(col_data)) * 100, 2),
                'ejemplo': str(col_data.dropna().iloc[0]) if len(col_data.dropna()) > 0 else 'N/A'
            }
    return stats

# **FUNCIÓN: Análisis de archivos**
def analizar_archivo(df):
    """
    Función para análisis de archivos que detecta anomalías financieras y contables.
    Se aplica un factor adaptativo según tamaño del archivo para evitar falsos positivos
    en datasets pequeños.
    """
    puntuaciones_por_regla = {}
    anomalias = []

    def agregar_puntuacion(regla, valor):
        puntuaciones_por_regla[regla] = max(puntuaciones_por_regla.get(regla, 0), valor)

    # Metadatos
    total_registros = len(df)
    total_columnas = len(df.columns)

    # Factor adaptativo según tamaño del dataset
    if total_registros >= 100:
        scale = 1.0
    elif total_registros >= 30:
        scale = 0.6
    elif total_registros >= 10:
        scale = 0.4
    else:
        scale = 0.2

    # Parámetros mínimos para penalizaciones "fuertes"
    min_strong = 10

    # ========== DETECCIÓN BÁSICA ==========

    # Valores nulos
    nulos_por_columna = df.isnull().sum()
    for columna, nulos in nulos_por_columna.items():
        if nulos > 0:
            porcentaje = (nulos / total_registros) * 100 if total_registros > 0 else 0
            if total_registros >= min_strong:
                if porcentaje > 20:
                    agregar_puntuacion('valores_nulos', 30 * scale)
                    anomalias.append({
                        'tipo': 'VALORES NULOS CRÍTICOS',
                        'columna': columna,
                        'cantidad': int(nulos),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
                elif porcentaje > 5:
                    agregar_puntuacion('valores_nulos', 15 * scale)
                    anomalias.append({
                        'tipo': 'Valores nulos',
                        'columna': columna,
                        'cantidad': int(nulos),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
            else:
                # Archivo pequeño: sólo alertas muy altas
                if porcentaje >= 50:
                    agregar_puntuacion('valores_nulos', 8 * scale)
                    anomalias.append({
                        'tipo': 'Valores nulos (archivo pequeño)',
                        'columna': columna,
                        'cantidad': int(nulos),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })

    # Duplicados (amortiguar impacto en archivos pequeños)
    duplicados = int(df.duplicated().sum())
    if duplicados > 0 and total_registros > 1:
        porcentaje_dup = (duplicados / total_registros) * 100
        if total_registros >= min_strong:
            if porcentaje_dup > 5:
                agregar_puntuacion('duplicados', 25 * scale)
                anomalias.append({
                    'tipo': 'DUPLICADOS EXACTOS - POSIBLE ERROR',
                    'cantidad': duplicados,
                    'porcentaje': f'{porcentaje_dup:.1f}%',
                    'categoria': 'Integridad'
                })
            elif porcentaje_dup > 0:
                agregar_puntuacion('duplicados', 10 * scale)
                anomalias.append({
                    'tipo': 'Registros duplicados',
                    'cantidad': duplicados,
                    'porcentaje': f'{porcentaje_dup:.1f}%',
                    'categoria': 'Integridad'
                })
        else:
            if porcentaje_dup > 20:
                agregar_puntuacion('duplicados', 6 * scale)
                anomalias.append({
                    'tipo': 'Duplicados (archivo pequeño)',
                    'cantidad': duplicados,
                    'porcentaje': f'{porcentaje_dup:.1f}%',
                    'categoria': 'Integridad'
                })

    # ========== DETECCIÓN FINANCIERA AVANZADA ==========

    columnas_monetarias = identificar_columnas_monetarias(df)
    columnas_fecha = identificar_columnas_fecha(df)

    # Transacciones redondas
    for col in columnas_monetarias:
        nonnull_count = int(df[col].dropna().shape[0])
        transacciones_redondas = int(detectar_transacciones_redondas(df[col]))
        if transacciones_redondas > 0 and nonnull_count > 0:
            porcentaje = (transacciones_redondas / nonnull_count) * 100
            if nonnull_count >= min_strong:
                if porcentaje > 20:
                    agregar_puntuacion('importes_redondos', 35 * scale)
                    anomalias.append({
                        'tipo': 'TRANSACCIONES REDONDAS SOSPECHOSAS',
                        'columna': col,
                        'cantidad': int(transacciones_redondas),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Fraude'
                    })
                elif porcentaje > 5:
                    agregar_puntuacion('importes_redondos', 12 * scale)
                    anomalias.append({
                        'tipo': 'Transacciones redondas',
                        'columna': col,
                        'cantidad': int(transacciones_redondas),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Patrón Inusual'
                    })
            else:
                if porcentaje > 50:
                    agregar_puntuacion('importes_redondos', 6 * scale)
                    anomalias.append({
                        'tipo': 'Transacciones redondas (muestra pequeña)',
                        'columna': col,
                        'cantidad': int(transacciones_redondas),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Patrón Inusual'
                    })

    # Outliers financieros
    for col in columnas_monetarias:
        nonnull_count = int(df[col].dropna().shape[0])
        outliers_financieros = int(detectar_outliers_financieros(df[col]))
        if outliers_financieros > 0 and nonnull_count > 0:
            porcentaje = (outliers_financieros / nonnull_count) * 100
            if nonnull_count >= min_strong and porcentaje > 5:
                agregar_puntuacion('valores_extremos', 30 * scale)
                anomalias.append({
                    'tipo': 'VALORES EXTREMOS EN MONTOS',
                    'columna': col,
                    'cantidad': int(outliers_financieros),
                    'porcentaje': f'{porcentaje:.1f}%',
                    'categoria': 'Riesgo Financiero'
                })
            elif porcentaje > 25:
                agregar_puntuacion('valores_extremos', 8 * scale)
                anomalias.append({
                    'tipo': 'Valores extremos (muestra pequeña)',
                    'columna': col,
                    'cantidad': int(outliers_financieros),
                    'porcentaje': f'{porcentaje:.1f}%',
                    'categoria': 'Riesgo Financiero'
                })

    # Transacciones fuera de horario
    for col in columnas_fecha:
        nonnull_count = int(df[col].dropna().shape[0])
        transacciones_nocturnas = int(detectar_transacciones_nocturnas(df[col]))
        if transacciones_nocturnas > 0 and nonnull_count >= min_strong:
            porcentaje = (transacciones_nocturnas / nonnull_count) * 100
            if porcentaje > 15:
                agregar_puntuacion('horario_nocturno', 20 * scale)
                anomalias.append({
                    'tipo': 'TRANSACCIONES FUERA DE HORARIO',
                    'columna': col,
                    'cantidad': int(transacciones_nocturnas),
                    'porcentaje': f'{porcentaje:.1f}%',
                    'categoria': 'Control Interno'
                })

    # Ley de Benford (solo con suficientes datos)
    for col in columnas_monetarias:
        if len(df[col].dropna()) > 100:
            desviacion_benford = analizar_ley_benford(df[col])
            if desviacion_benford > 0.15:
                agregar_puntuacion('benford', 25 * scale)
                anomalias.append({
                    'tipo': 'DESVIACIÓN LEY DE BENFORD',
                    'columna': col,
                    'valor': f'{desviacion_benford:.3f}',
                    'categoria': 'Análisis Estadístico'
                })

    # Saltos bruscos
    for col in columnas_monetarias:
        nonnull_count = int(df[col].dropna().shape[0])
        saltos_bruscos = int(detectar_saltos_bruscos(df[col]))
        if saltos_bruscos > 0 and nonnull_count >= min_strong:
            agregar_puntuacion('saltos_montos', 12 * scale)
            anomalias.append({
                'tipo': 'Saltos bruscos en valores',
                'columna': col,
                'cantidad': int(saltos_bruscos),
                'categoria': 'Consistencia'
            })

    # Transacciones en fin de semana
    for col in columnas_fecha:
        nonnull_count = int(df[col].dropna().shape[0])
        transacciones_finde = int(detectar_transacciones_finde(df[col]))
        if transacciones_finde > 0 and nonnull_count >= min_strong:
            porcentaje = (transacciones_finde / nonnull_count) * 100
            if porcentaje > 10:
                agregar_puntuacion('operaciones_fin_semana', 12 * scale)
                anomalias.append({
                    'tipo': 'Transacciones en fin de semana',
                    'columna': col,
                    'cantidad': int(transacciones_finde),
                    'porcentaje': f'{porcentaje:.1f}%',
                    'categoria': 'Patrón Inusual'
                })

    # Asientos desbalanceados (requerir algo de datos)
    desbalance = int(detectar_asientos_desbalanceados(df))
    if desbalance > 0:
        if total_registros >= 5:
            agregar_puntuacion('balance_contable', 30 * scale)
        else:
            agregar_puntuacion('balance_contable', 10 * scale)
        anomalias.append({
            'tipo': 'ASIENTOS CONTABLES DESBALANCEADOS',
            'cantidad': int(desbalance),
            'categoria': 'Error Contable'
        })

    # Huecos en secuencia (muestras pequeñas penalizadas menos)
    sequence_terms = ('consecutivo', 'comprobante', 'asiento', 'factura', 'recibo', 'voucher', 'invoice', 'folio')
    for col in df.select_dtypes(include=[np.number]).columns:
        if not any(term in normalizar_nombre_columna(col) for term in sequence_terms):
            continue
        huecos_secuencia = int(detectar_huecos_secuencia(df[col]))
        if huecos_secuencia > 0:
            agregar_puntuacion('huecos_secuencia', 6 * scale)
            anomalias.append({
                'tipo': 'Huecos en secuencia numérica',
                'columna': col,
                'cantidad': int(huecos_secuencia),
                'categoria': 'Integridad'
            })

    # Outliers por columnas numéricas (IQR)
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        nonnull_count = int(df[col].dropna().shape[0])
        if outliers > 0 and nonnull_count > 0:
            porcentaje_out = (outliers / nonnull_count) * 100
            if nonnull_count >= min_strong:
                if porcentaje_out > 10:
                    agregar_puntuacion('outliers_numericos', 25 * scale)
                    anomalias.append({
                        'tipo': 'OUTLIERS CRÍTICOS',
                        'columna': col,
                        'cantidad': int(outliers),
                        'porcentaje': f'{porcentaje_out:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
                else:
                    agregar_puntuacion('outliers_numericos', 8 * scale)
                    anomalias.append({
                        'tipo': 'Valores atípicos',
                        'columna': col,
                        'cantidad': int(outliers),
                        'porcentaje': f'{porcentaje_out:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
            else:
                if porcentaje_out > 30:
                    agregar_puntuacion('outliers_numericos', 6 * scale)
                    anomalias.append({
                        'tipo': 'Valores atípicos (muestra pequeña)',
                        'columna': col,
                        'cantidad': int(outliers),
                        'porcentaje': f'{porcentaje_out:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })

    puntuacion = sum(puntuaciones_por_regla.values())

    # ========== CLASIFICACIÓN DE RIESGO ==========
    nivel_ml, probabilidades = risk_predictor.predict_file_risk(df)
    nivel_score = {'Bajo': 20, 'Medio': 48, 'Alto': 75, 'Crítico': 95}[nivel_ml]
    puntuacion_final = min(100, round(max(puntuacion, nivel_score * 0.82), 2))

    if puntuacion_final >= 85:
        nivel_riesgo = 'Crítico'
    elif puntuacion_final >= 60:
        nivel_riesgo = 'Alto'
    elif puntuacion_final >= 35:
        nivel_riesgo = 'Medio'
    else:
        nivel_riesgo = 'Bajo'

    anomalias = agrupar_anomalias(anomalias)
    for anomalia in anomalias:
        explicacion = explicar_anomalia(anomalia)
        anomalia.update(explicacion)

    return {
        'total_registros': total_registros,
        'total_columnas': total_columnas,
        'puntuacion_riesgo': puntuacion_final,
        'nivel_riesgo': nivel_riesgo,
        'anomalias': anomalias,
        'ml_prediction': nivel_ml,
        'ml_confidence': {
            'Bajo': round(probabilidades[0] * 100, 2),
            'Medio': round(probabilidades[1] * 100, 2),
            'Alto': round(probabilidades[2] * 100, 2),
            'Crítico': round(probabilidades[3] * 100, 2)
        },
        'metricas_avanzadas': {
            'columnas_monetarias': columnas_monetarias,
            'columnas_fecha': columnas_fecha,
            'total_anomalias_financieras': len([a for a in anomalias if a.get('categoria') in ['Fraude', 'Riesgo Financiero']])
        }
    }


def explicar_anomalia(anomalia):
    """Traduce alertas técnicas del análisis a explicaciones y pasos comprensibles."""
    family_explanations = {
        'celdas_vacias': (
            'Hay celdas vacías en una o varias columnas.',
            'Revise si esos espacios son esperados. Si el dato es obligatorio para el negocio, complete o corrija el registro.'
        ),
        'filas_repetidas': (
            'Hay filas iguales o repetidas en el archivo.',
            'Compare fecha, comprobante y cuenta para confirmar si son movimientos duplicados o registros legítimos. No borre filas sin comprobarlo.'
        ),
        'importes_redondos': (
            'Algunos importes terminan en ceros o son múltiplos exactos.',
            'Compare esos movimientos con facturas o recibos. Los importes redondos son comunes y por sí solos no indican un problema.'
        ),
        'valores_atipicos': (
            'Algunos valores se alejan bastante del resto.',
            'Revise los registros de esas columnas y confirme que la moneda, la unidad y el periodo sean correctos. Un valor diferente puede ser legítimo.'
        ),
        'cambios_importe': (
            'Hay cambios grandes entre importes registrados uno después de otro.',
            'Compruebe si corresponden a pagos extraordinarios, cambios de precio o ajustes normales del negocio.'
        ),
        'numeracion_comprobantes': (
            'La numeración de comprobantes salta algunos números.',
            'Por ejemplo, podría pasar de 12 a 14 y no aparecer el 13. Esto puede ser normal si se anularon comprobantes; revise si su negocio exige una numeración continua.'
        ),
    }
    if anomalia.get('familia') in family_explanations:
        explanation, suggested_action = family_explanations[anomalia['familia']]
        return {'explicacion': explanation, 'accion_sugerida': suggested_action}

    explanations = {
        'VALORES NULOS CRÍTICOS': (
            'Faltan muchos datos en esta columna.',
            'Compruebe si los datos faltantes son esperados y complete o corrija los registros cuando sea posible.'
        ),
        'Valores nulos': (
            'Hay algunas celdas vacías en esta columna.',
            'Revise si esos campos deberían tener información antes de usar el archivo para reportes.'
        ),
        'Valores nulos (archivo pequeño)': (
            'En este archivo pequeño hay varias celdas vacías.',
            'Revise las filas indicadas; en archivos pequeños, unas pocas celdas pueden influir bastante en el análisis.'
        ),
        'DUPLICADOS EXACTOS - POSIBLE ERROR': (
            'Se encontraron filas completamente repetidas.',
            'Confirme si son movimientos distintos o si algún registro se cargó más de una vez; no elimine filas sin verificar.'
        ),
        'Registros duplicados': (
            'Hay filas repetidas en el archivo.',
            'Revise los identificadores y fechas para confirmar si son duplicados reales o movimientos legítimos.'
        ),
        'Duplicados (archivo pequeño)': (
            'Se detectaron repeticiones en una muestra pequeña.',
            'Verifique manualmente las filas repetidas; el tamaño de la muestra puede hacer que esta alerta sea menos representativa.'
        ),
        'TRANSACCIONES REDONDAS SOSPECHOSAS': (
            'Aparecen muchos importes redondos o con cambios poco habituales.',
            'Compare esos movimientos con facturas, recibos o soportes. Un importe redondo por sí solo no demuestra fraude.'
        ),
        'Transacciones redondas': (
            'Se observaron algunos importes redondos.',
            'Compruebe si los valores corresponden a precios o pagos habituales del negocio.'
        ),
        'Transacciones redondas (muestra pequeña)': (
            'En esta muestra pequeña hay importes redondos.',
            'Tómelo como una señal para revisar, no como evidencia de un problema.'
        ),
        'VALORES EXTREMOS EN MONTOS': (
            'Algunos importes son mucho mayores o menores que la mayoría.',
            'Verifique los movimientos extremos con sus soportes y confirme que la moneda y el formato sean correctos.'
        ),
        'Valores extremos (muestra pequeña)': (
            'Hay importes que se apartan de los demás en una muestra pequeña.',
            'Revise esos valores manualmente; con pocos registros es más difícil distinguir lo normal de lo inusual.'
        ),
        'TRANSACCIONES FUERA DE HORARIO': (
            'Hay movimientos registrados en horarios poco habituales.',
            'Confirme el horario de operación y quién autorizó esos movimientos; algunos negocios operan fuera del horario convencional.'
        ),
        'DESVIACIÓN LEY DE BENFORD': (
            'La distribución de los primeros dígitos difiere de un patrón estadístico esperado.',
            'Considérelo solo como una señal estadística. No prueba fraude y puede no aplicar a todos los tipos de datos.'
        ),
        'Saltos bruscos en valores': (
            'La columna presenta cambios grandes entre algunos importes consecutivos.',
            'Compruebe si hubo cambios de precio, pagos extraordinarios, ajustes o errores de digitación.'
        ),
        'Transacciones en fin de semana': (
            'Se registraron movimientos durante fines de semana.',
            'Verifique si la empresa realiza operaciones esos días; puede ser completamente normal según la actividad.'
        ),
        'ASIENTOS CONTABLES DESBALANCEADOS': (
            'La suma de las columnas de débito y crédito no coincide según esta revisión.',
            'Confirme que las columnas estén bien identificadas, usen la misma moneda y correspondan al mismo conjunto de asientos; consulte a una persona contadora.'
        ),
        'Huecos en secuencia numérica': (
            'La numeración de comprobantes salta algunos números.',
            'Por ejemplo, podría pasar de 12 a 14 y no aparecer el 13. Esto puede ser normal si se anularon comprobantes; revise si su negocio exige una numeración continua.'
        ),
        'OUTLIERS CRÍTICOS': (
            'Muchos valores se alejan del rango habitual de esta columna.',
            'Revise los valores señalados y valide la unidad, la moneda y el periodo comparado.'
        ),
        'Valores atípicos': (
            'Algunos valores se alejan del rango habitual de la columna.',
            'Verifique los registros con sus soportes. Un valor atípico puede ser válido.'
        ),
        'Valores atípicos (muestra pequeña)': (
            'Algunos valores se apartan del resto de esta muestra pequeña.',
            'Revise los registros manualmente; la alerta puede cambiar mucho con más datos.'
        ),
    }
    explanation, suggested_action = explanations.get(
        anomalia.get('tipo'),
        ('Se encontró un patrón que merece revisión.', 'Verifique los registros originales antes de tomar una decisión.')
    )
    return {'explicacion': explanation, 'accion_sugerida': suggested_action}


def agrupar_anomalias(anomalias):
    """Agrupa alertas repetidas de la misma regla y conserva el detalle por columna."""
    families = {
        'VALORES NULOS CRÍTICOS': 'celdas_vacias',
        'Valores nulos': 'celdas_vacias',
        'Valores nulos (archivo pequeño)': 'celdas_vacias',
        'DUPLICADOS EXACTOS - POSIBLE ERROR': 'filas_repetidas',
        'Registros duplicados': 'filas_repetidas',
        'Duplicados (archivo pequeño)': 'filas_repetidas',
        'TRANSACCIONES REDONDAS SOSPECHOSAS': 'importes_redondos',
        'Transacciones redondas': 'importes_redondos',
        'Transacciones redondas (muestra pequeña)': 'importes_redondos',
        'VALORES EXTREMOS EN MONTOS': 'valores_atipicos',
        'Valores extremos (muestra pequeña)': 'valores_atipicos',
        'OUTLIERS CRÍTICOS': 'valores_atipicos',
        'Valores atípicos': 'valores_atipicos',
        'Valores atípicos (muestra pequeña)': 'valores_atipicos',
        'Saltos bruscos en valores': 'cambios_importe',
        'Huecos en secuencia numérica': 'numeracion_comprobantes',
        'ASIENTOS CONTABLES DESBALANCEADOS': 'balance_contable',
    }
    grouped = {}
    for anomaly in anomalias:
        family = families.get(anomaly['tipo'], anomaly['tipo'])
        entry = grouped.get(family)
        if entry is None:
            entry = dict(anomaly)
            entry['familia'] = family
            entry['variantes'] = []
            entry['columnas_afectadas'] = []
            grouped[family] = entry

        if anomaly['tipo'] not in entry['variantes']:
            entry['variantes'].append(anomaly['tipo'])
        if anomaly.get('columna'):
            entry['columnas_afectadas'].append({
                'columna': anomaly['columna'],
                'cantidad': anomaly.get('cantidad'),
                'porcentaje': anomaly.get('porcentaje'),
            })
        elif not entry.get('cantidad') and anomaly.get('cantidad'):
            entry['cantidad'] = anomaly['cantidad']
        if anomaly.get('porcentaje'):
            entry['porcentaje'] = anomaly['porcentaje']
        if anomaly.get('categoria'):
            entry['categoria'] = anomaly['categoria']

    category_labels = {
        'Calidad de Datos': 'Datos incompletos o inusuales',
        'Integridad': 'Revisión de registros',
        'Fraude': 'Importes para verificar',
        'Patrón Inusual': 'Patrón para verificar',
        'Riesgo Financiero': 'Importes para verificar',
        'Control Interno': 'Horario de operación',
        'Análisis Estadístico': 'Patrón estadístico',
        'Consistencia': 'Cambios entre importes',
        'Error Contable': 'Balance de débitos y créditos',
    }
    for entry in grouped.values():
        entry['categoria_amigable'] = category_labels.get(entry.get('categoria'), 'Revisión sugerida')
        if not entry['columnas_afectadas']:
            entry.pop('columnas_afectadas')
        entry.pop('variantes')
    return list(grouped.values())

#----- FUNCIONES DE DETECCIÓN FINANCIERA AVANZADA ----- 
# ========== FUNCIONES DE DETECCIÓN FINANCIERA ==========

def identificar_columnas_monetarias(df):
    """Identifica columnas que probablemente contengan valores monetarios"""
    columnas_monetarias = []
    palabras_clave = ['monto', 'valor', 'importe', 'amount', 'total', 'saldo', 'precio',
                      'costo', 'debito', 'credito', 'balance', 'payment', 'pago', 'debe', 'haber']
    
    for col in df.columns:
        col_lower = normalizar_nombre_columna(col)
        # Verificar por nombre de columna
        if any(palabra in col_lower for palabra in palabras_clave):
            columnas_monetarias.append(col)
        # Verificar por tipo de datos (numéricos)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Si la columna tiene valores que parecen montos monetarios
            if not df[col].dropna().empty and df[col].dropna().between(0, 10000000).all():
                columnas_monetarias.append(col)
    
    return list(set(columnas_monetarias))

def identificar_columnas_fecha(df):
    """Identifica columnas que probablemente contengan fechas"""
    columnas_fecha = []
    palabras_clave = ['fecha', 'date', 'time', 'hora', 'dia', 'mes', 'ano', 'timestamp']
    
    for col in df.columns:
        col_lower = normalizar_nombre_columna(col)
        if any(palabra in col_lower for palabra in palabras_clave):
            columnas_fecha.append(col)
        # Verificar si la columna es de tipo datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            columnas_fecha.append(col)
    
    return list(set(columnas_fecha))


def normalizar_nombre_columna(columna):
    """Normaliza nombres para reconocer encabezados con tildes y mayúsculas."""
    normalized = unicodedata.normalize('NFKD', str(columna).lower())
    return ''.join(character for character in normalized if not unicodedata.combining(character))


def detectar_transacciones_redondas(serie):
    """Detecta transacciones con montos redondos (ej: 1000, 5000)"""
    if not pd.api.types.is_numeric_dtype(serie) or serie.empty:
        return 0
    
    serie_limpia = serie.dropna()
    # Montos redondos comunes (múltiplos de 100, 1000, etc.)
    transacciones_redondas = serie_limpia[
        (serie_limpia % 100 == 0) | 
        (serie_limpia % 1000 == 0) |
        (serie_limpia % 500 == 0)
    ]
    
    return len(transacciones_redondas)

def detectar_outliers_financieros(serie):
    """Detecta outliers usando método IQR mejorado para datos financieros"""
    if not pd.api.types.is_numeric_dtype(serie) or serie.empty:
        return 0
    
    serie_limpia = serie.dropna()
    if len(serie_limpia) < 4:
        return 0
    
    Q1 = serie_limpia.quantile(0.25)
    Q3 = serie_limpia.quantile(0.75)
    IQR = Q3 - Q1
    
    # Límites más estrictos para datos financieros
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    
    outliers = serie_limpia[(serie_limpia < lower_bound) | (serie_limpia > upper_bound)]
    return len(outliers)

def detectar_transacciones_nocturnas(serie_fecha):
    """Detecta transacciones fuera del horario comercial (6 PM - 8 AM)"""
    try:
        if pd.api.types.is_datetime64_any_dtype(serie_fecha):
            horas = serie_fecha.dt.hour
            # Transacciones entre 6 PM y 8 AM
            nocturnas = horas[(horas >= 18) | (horas < 8)]
            return len(nocturnas)
    except:
        pass
    return 0

def detectar_transacciones_finde(serie_fecha):
    """Detecta transacciones en sábado o domingo"""
    try:
        if pd.api.types.is_datetime64_any_dtype(serie_fecha):
            dias_semana = serie_fecha.dt.dayofweek
            # 5 = sábado, 6 = domingo
            finde = dias_semana[dias_semana >= 5]
            return len(finde)
    except:
        pass
    return 0

def analizar_ley_benford(serie):
    """Analiza la distribución del primer dígito según la Ley de Benford"""
    if not pd.api.types.is_numeric_dtype(serie) or serie.empty:
        return 0
    
    serie_limpia = serie.dropna()
    if len(serie_limpia) < 100:
        return 0
    
    # Obtener primer dígito
    primeros_digitos = serie_limpia.abs().apply(
        lambda x: int(str(x).replace('.', '').lstrip('0')[0]) if x != 0 else 0
    )
    primeros_digitos = primeros_digitos[primeros_digitos > 0]
    
    # Distribución esperada según Benford
    benford_expected = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    
    # Distribución observada
    counts = primeros_digitos.value_counts().sort_index()
    total = len(primeros_digitos)
    
    if total == 0:
        return 0
    
    # Calcular desviación
    desviacion = 0
    for i in range(1, 10):
        observed = counts.get(i, 0) / total
        expected = benford_expected[i-1]
        desviacion += abs(observed - expected)
    
    return desviacion

def detectar_saltos_bruscos(serie):
    """Detecta saltos bruscos en secuencias numéricas"""
    if not pd.api.types.is_numeric_dtype(serie) or len(serie) < 2:
        return 0
    
    serie_limpia = serie.dropna()
    if len(serie_limpia) < 2:
        return 0
    
    diferencias = serie_limpia.diff().abs()
    media = diferencias.mean()
    std = diferencias.std()
    
    if std == 0:
        return 0
    
    # Saltos mayores a 3 desviaciones estándar
    saltos = diferencias[diferencias > media + 3 * std]
    return len(saltos)

def detectar_asientos_desbalanceados(df):
    """Detecta desbalance entre columnas de débito/crédito o debe/haber."""
    columnas_debito = [
        col for col in df.columns
        if any(term in normalizar_nombre_columna(col) for term in ('debito', 'debit', 'debe'))
    ]
    columnas_credito = [
        col for col in df.columns
        if any(term in normalizar_nombre_columna(col) for term in ('credito', 'credit', 'haber'))
    ]
    
    if not columnas_debito or not columnas_credito:
        return 0
    
    try:
        total_debito = df[columnas_debito].sum().sum()
        total_credito = df[columnas_credito].sum().sum()
        
        # Tolerancia del 1% para diferencias de redondeo
        tolerancia = max(total_debito, total_credito) * 0.01
        
        if abs(total_debito - total_credito) > tolerancia:
            return 1
    except:
        pass
    
    return 0

def detectar_huecos_secuencia(serie):
    """Detecta huecos en secuencias numéricas (ej: números de comprobante faltantes)"""
    if not pd.api.types.is_numeric_dtype(serie) or serie.empty:
        return 0
    
    serie_limpia = serie.dropna().sort_values()
    if len(serie_limpia) < 2:
        return 0
    
    diferencias = serie_limpia.diff()
    # Huecos mayores a 1 en una secuencia que debería ser consecutiva
    huecos = diferencias[diferencias > 1]
    return len(huecos)

@app.route('/perfil')
@login_required
def perfil():
    try:
        with get_db() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # Obtener datos del usuario
            cursor.execute("SELECT * FROM usuarios WHERE id = %s", (session['user_id'],))
            user = cursor.fetchone()
            
            if not user:
                flash('Usuario no encontrado', 'error')
                return redirect('/')
            
            # Obtener estadísticas
            stats = get_user_stats(session['user_id'])
            plan_info = PLANES.get(user['plan'], PLANES['lite'])
            restantes = calcular_diagnosticos_restantes(session['user_id'])
            
            # OBTENER ACTIVIDAD RECIENTE (solo 5 para el preview)
            actividad_reciente = []
            
            # 1. Obtener últimos diagnósticos
            cursor.execute("""
                SELECT id, empresa, tipo_riesgo, nivel_riesgo, fecha_analisis, 'diagnostico' as tipo
                FROM diagnosticos 
                WHERE usuario_id = %s 
                ORDER BY fecha_analisis DESC 
                LIMIT 5
            """, (session['user_id'],))
            diagnosticos = cursor.fetchall()
            
            for diag in diagnosticos:
                actividad_reciente.append({
                    'id': diag['id'],
                    'titulo': f"Diagnóstico: {diag['empresa']}",
                    'descripcion': f"Riesgo {diag['tipo_riesgo']} - Nivel {diag['nivel_riesgo']}",
                    'fecha': diag['fecha_analisis'],
                    'tipo': 'diagnostico',
                    'color': 'primary',
                    'icono': 'fas fa-clipboard-list'
                })
            
            # 2. Obtener últimos archivos analizados
            cursor.execute("""
                SELECT id, archivo_nombre, nivel_riesgo, fecha_analisis, 'archivo' as tipo
                FROM archivos_analizados 
                WHERE usuario_id = %s 
                ORDER BY fecha_analisis DESC 
                LIMIT 5
            """, (session['user_id'],))
            archivos = cursor.fetchall()
            
            for archivo in archivos:
                actividad_reciente.append({
                    'id': archivo['id'],
                    'titulo': f"Archivo: {archivo['archivo_nombre'][:30]}{'...' if len(archivo['archivo_nombre']) > 30 else ''}",
                    'descripcion': f"Riesgo {archivo['nivel_riesgo']}",
                    'fecha': archivo['fecha_analisis'],
                    'tipo': 'archivo',
                    'color': 'success',
                    'icono': 'fas fa-file-excel'
                })
            
            # Ordenar por fecha (más reciente primero) y limitar a 3
            actividad_reciente.sort(key=lambda x: x['fecha'], reverse=True)
            actividad_reciente = actividad_reciente[:3]
            
            # Calcular total de actividades para mostrar botón "Ver más"
            cursor.execute("""
                SELECT COUNT(*) as total FROM (
                    SELECT id FROM diagnosticos WHERE usuario_id = %s
                    UNION ALL
                    SELECT id FROM archivos_analizados WHERE usuario_id = %s
                ) as combined
            """, (session['user_id'], session['user_id']))
            total_actividades = cursor.fetchone()['total']
            
    except Exception as e:
        print(f"Error en perfil: {e}")
        flash('Error al cargar el perfil', 'error')
        return redirect('/')
    
    return render_template('perfil.html', 
                            user=user, 
                            plan_info=plan_info, 
                            stats=stats, 
                            restantes=restantes,
                            actividad_reciente=actividad_reciente,
                            total_actividades=total_actividades,
                            now=datetime.now())

@app.route('/perfil/editar', methods=['GET', 'POST'])
@login_required
def editar_perfil():
    try:
        with get_db() as conn:
            cursor = conn.cursor(dictionary=True)

            # Obtener datos actuales del usuario
            cursor.execute("SELECT * FROM usuarios WHERE id = %s", (session['user_id'],))
            user = cursor.fetchone()
            
            if not user:
                flash('Usuario no encontrado', 'error')
                return redirect('/')

            if request.method == 'POST':
                # Tomar datos del formulario
                email = request.form.get('email')
                telefono = request.form.get('telefono')
                ciudad = request.form.get('ciudad')
                departamento = request.form.get('departamento')
                pais = request.form.get('pais')
                sector = request.form.get('sector')

                # Actualizar en la BD
                cursor.execute("""
                    UPDATE usuarios
                    SET email=%s, telefono=%s, ciudad=%s, departamento=%s, pais=%s, sector=%s
                    WHERE id=%s
                """, (email, telefono, ciudad, departamento, pais, sector, session['user_id']))
                conn.commit()

                flash('Perfil actualizado correctamente', 'success')
                return redirect(url_for('perfil'))

    except Exception as e:
        print(f"Error en editar perfil: {e}")
        flash('Error al actualizar el perfil', 'error')
        return redirect(url_for('perfil'))

    # Mostrar el formulario con los datos actuales
    return render_template('perfil_editar.html', user=user)


@app.route('/planes')
def planes():
    return render_template('planes.html', planes=PLANES)

#======QUIENES SOMOS ======
@app.route('/quienes-somos')
def quienes_somos():
    return render_template('quienes_somos.html')

@app.route('/upgrade_plan', methods=['POST'])
@login_required
def upgrade_plan():
    nuevo_plan = request.form.get('plan')
    
    if nuevo_plan not in PLANES:
        flash('Plan no válido', 'error')
        return redirect(url_for('planes'))
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET plan = %s WHERE id = %s", (nuevo_plan, session['user_id']))
        conn.commit()
    
    session['plan'] = nuevo_plan
    flash(f'Plan actualizado a {PLANES[nuevo_plan]["nombre"]}', 'success')
    return redirect(url_for('perfil'))

# ==================== RUTAS DE ADMINISTRADOR ====================

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        
        # Estadísticas generales
        cursor.execute("SELECT COUNT(*) as total FROM usuarios")
        total_usuarios = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM diagnosticos")
        total_diagnosticos = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM archivos_analizados")
        total_archivos = cursor.fetchone()['total']
        
        # Usuarios por plan
        cursor.execute("SELECT plan, COUNT(*) as cantidad FROM usuarios GROUP BY plan")
        usuarios_plan = cursor.fetchall()
        
        # Diagnosticos por mes (últimos 6 meses)
        cursor.execute("""
            SELECT TO_CHAR(fecha_analisis, 'YYYY-MM') as mes,
                   COUNT(*) as total
            FROM diagnosticos
            WHERE fecha_analisis >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY mes
            ORDER BY mes DESC
        """)
        diagnosticos_mes = cursor.fetchall()
        
        # Últimos diagnósticos
        cursor.execute("""
            SELECT d.*, u.username, u.empresa 
            FROM diagnosticos d 
            JOIN usuarios u ON d.usuario_id = u.id 
            ORDER BY d.fecha_analisis DESC 
            LIMIT 10
        """)
        ultimos_diagnosticos = cursor.fetchall()
        
        # ========== CORREGIDO: Archivos recientes ==========
        cursor.execute("""
            SELECT a.*, u.username, u.empresa 
            FROM archivos_analizados a
            JOIN usuarios u ON a.usuario_id = u.id 
            ORDER BY a.fecha_analisis DESC 
            LIMIT 10
        """)
        ultimos_archivos = cursor.fetchall()
        
        # ========== NUEVO: Estadísticas de riesgo ==========
        cursor.execute("""
            SELECT nivel_riesgo, COUNT(*) as cantidad
            FROM diagnosticos
            GROUP BY nivel_riesgo
        """)
        riesgos_diagnosticos = cursor.fetchall()
        
        cursor.execute("""
            SELECT nivel_riesgo, COUNT(*) as cantidad
            FROM archivos_analizados
            GROUP BY nivel_riesgo
        """)
        riesgos_archivos = cursor.fetchall()
    
    return render_template('admin/dashboard.html', 
                        total_usuarios=total_usuarios,
                        total_diagnosticos=total_diagnosticos,
                        total_archivos=total_archivos,
                        usuarios_plan=usuarios_plan,
                        diagnosticos_mes=diagnosticos_mes,
                        ultimos_diagnosticos=ultimos_diagnosticos,
                        ultimos_archivos=ultimos_archivos,
                        riesgos_diagnosticos=riesgos_diagnosticos,
                        riesgos_archivos=riesgos_archivos)

@app.route('/admin/usuarios')
@admin_required
def admin_usuarios():
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, username, email, empresa, rol, plan, 
                   diagnosticos_realizados, archivos_analizados,
                   fecha_registro, activo
            FROM usuarios 
            ORDER BY id DESC
        """)
        usuarios = cursor.fetchall()
    return render_template('admin/usuarios.html', usuarios=usuarios)

@app.route('/admin/diagnosticos')
@admin_required
def admin_diagnosticos():
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT d.*, u.username, u.empresa, u.plan as usuario_plan
            FROM diagnosticos d 
            JOIN usuarios u ON d.usuario_id = u.id 
            ORDER BY d.fecha_analisis DESC
        """)
        diagnosticos = cursor.fetchall()
    return render_template('admin/diagnosticos.html', diagnosticos=diagnosticos)

@app.route('/admin/archivos')
@admin_required
def admin_archivos():
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT a.*, u.username, u.empresa, u.plan as usuario_plan
            FROM archivos_analizados a 
            JOIN usuarios u ON a.usuario_id = u.id 
            ORDER BY a.fecha_analisis DESC
        """)
        archivos = cursor.fetchall()
    return render_template('admin/archivos.html', archivos=archivos)

@app.route('/mi_actividad')
@login_required
def mi_actividad():
    if session.get('rol') == 'admin':
        return redirect('/admin/dashboard')
    
    # Parámetros de paginación
    pagina = request.args.get('pagina', 1, type=int)
    por_pagina = 10  # 10 actividades por página
    
    try:
        with get_db() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # Obtener todas las actividades combinadas
            cursor.execute("""
                (SELECT id, empresa as nombre, tipo_riesgo, nivel_riesgo, fecha_analisis, 'diagnostico' as tipo
                 FROM diagnosticos 
                 WHERE usuario_id = %s)
                UNION ALL
                (SELECT id, archivo_nombre as nombre, '' as tipo_riesgo, nivel_riesgo, fecha_analisis, 'archivo' as tipo
                 FROM archivos_analizados 
                 WHERE usuario_id = %s)
                ORDER BY fecha_analisis DESC
                LIMIT %s OFFSET %s
            """, (session['user_id'], session['user_id'], por_pagina, (pagina - 1) * por_pagina))
            
            actividades = cursor.fetchall()
            
            # Contar total de actividades para paginación
            cursor.execute("""
                SELECT COUNT(*) as total FROM (
                    SELECT id FROM diagnosticos WHERE usuario_id = %s
                    UNION ALL
                    SELECT id FROM archivos_analizados WHERE usuario_id = %s
                ) as combined
            """, (session['user_id'], session['user_id']))
            
            total_actividades = cursor.fetchone()['total']
            total_paginas = (total_actividades + por_pagina - 1) // por_pagina
            
    except Exception as e:
        print(f"Error en mi_actividad: {e}")
        flash('Error al cargar la actividad', 'error')
        return redirect('/perfil')
    
    return render_template('mi_actividad.html', 
                         actividades=actividades,
                         pagina_actual=pagina,
                         total_paginas=total_paginas,
                         total_actividades=total_actividades)


@app.route('/admin/toggle_usuario/<int:user_id>')
@admin_required
def toggle_usuario(user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET activo = NOT activo WHERE id = %s", (user_id,))
        conn.commit()
    
    flash('Estado de usuario actualizado', 'success')
    return redirect(url_for('admin_usuarios'))

@app.route('/admin/cambiar_plan/<int:user_id>', methods=['POST'])
@admin_required
def cambiar_plan(user_id):
    nuevo_plan = request.form.get('plan')
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET plan = %s WHERE id = %s", (nuevo_plan, user_id))
        conn.commit()
    
    flash(f'Plan actualizado a {nuevo_plan}', 'success')
    return redirect(url_for('admin_usuarios'))

@app.route('/admin/cambiar_rol/<int:user_id>', methods=['POST'])
@admin_required
def cambiar_rol(user_id):
    nuevo_rol = request.form.get('rol')
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET rol = %s WHERE id = %s", (nuevo_rol, user_id))
        conn.commit()
    
    flash(f'Rol actualizado a {nuevo_rol}', 'success')
    return redirect(url_for('admin_usuarios'))

@app.route('/admin/crear_usuario', methods=['POST'])
@admin_required
def crear_usuario():
    """Crear usuario desde el panel de administración"""
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    email = request.form.get('email')
    empresa = request.form.get('empresa')
    rol = request.form.get('rol', 'pyme')
    plan = request.form.get('plan', 'lite')
    
    # Validaciones
    if password != confirm_password:
        flash('Las contraseñas no coinciden', 'error')
        return redirect(url_for('admin_usuarios'))
    
    if get_user_by_username(username):
        flash('El usuario ya existe', 'error')
        return redirect(url_for('admin_usuarios'))
    
    # Crear usuario
    hashed = generate_password_hash(password, method='pbkdf2:sha256')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO usuarios (username, password, email, empresa, rol, plan, activo)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
        """, (username, hashed, email, empresa, rol, plan))
        conn.commit()
    
    flash(f'Usuario {username} creado exitosamente', 'success')
    return redirect(url_for('admin_usuarios'))

@app.route('/admin/eliminar_diagnostico/<int:diagnostico_id>')
@admin_required
def eliminar_diagnostico(diagnostico_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM diagnosticos WHERE id = %s", (diagnostico_id,))
        conn.commit()
    
    flash('Diagnóstico eliminado correctamente', 'success')
    return redirect(url_for('admin_diagnosticos'))

@app.route('/admin/eliminar_archivo/<int:archivo_id>')
@admin_required
def eliminar_archivo(archivo_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM archivos_analizados WHERE id = %s", (archivo_id,))
        conn.commit()
    
    flash('Archivo analizado eliminado correctamente', 'success')
    return redirect(url_for('admin_archivos'))
# ==================== SISTEMA DE NOTIFICACIONES ====================
def crear_notificacion(usuario_id, titulo, mensaje, tipo='info'):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notificaciones (usuario_id, titulo, mensaje, tipo, leida)
            VALUES (%s, %s, %s, %s, FALSE)
        """, (usuario_id, titulo, mensaje, tipo))
        conn.commit()

@app.route('/notificaciones')
@login_required
def notificaciones():
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM notificaciones 
            WHERE usuario_id = %s 
            ORDER BY fecha_creacion DESC
        """, (session['user_id'],))
        notificaciones = cursor.fetchall()
    return render_template('notificaciones.html', notificaciones=notificaciones)

# ==================== MANEJO DE ERRORES ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error_code=404, error_message='Página no encontrada'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500, error_message='Error del servidor'), 500

# ==================== INICIO ====================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("CYBERSHIELD PYME INICIANDO...")
    print("="*50)
    print("Base de datos: cybershield_db")
    print("🌐 URL: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
