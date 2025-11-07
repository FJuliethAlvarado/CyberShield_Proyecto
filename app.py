from flask import Flask, render_template, request, flash, redirect, url_for, session, send_file
from flask_wtf.csrf import CSRFProtect
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os
import math
import logging
import pandas as pd
import numpy as np
from functools import wraps

# ==================== CONFIGURACIÓN ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_super_segura_123456'
app.config['WTF_CSRF_ENABLED'] = False  # ← AGREGAR ESTA LÍNEA
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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/profile_pics', exist_ok=True)

# ==================== CONFIGURACIÓN DE MYSQL ====================
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST'),
    'user': os.environ.get('POSTGRES_USER'),
    'password': os.environ.get('POSTGRES_PASSWORD'),
    'database': os.environ.get('POSTGRES_DATABASE'),
    'port': os.environ.get('POSTGRES_PORT', 5432)
}

@contextmanager
def get_db():
    """Conexión a la base de datos"""
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        yield conn
    except Error as e:
        logger.error(f"Error BD: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

# ==================== MODELO DE MACHINE LEARNING ====================
class RiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.le_tipo = LabelEncoder()
        self.le_impacto = LabelEncoder()
        self.le_probabilidad = LabelEncoder()
        self.is_trained = False
        
    def train_model(self):
        """Entrena el modelo con datos históricos"""
        # Datos de entrenamiento simulados
        X_train = [
            [0, 2, 2],  # Tecnológico, Alto, Alta -> Crítico
            [0, 2, 1],  # Tecnológico, Alto, Media -> Alto
            [1, 1, 2],  # Financiero, Medio, Alta -> Alto
            [2, 0, 0],  # Operativo, Bajo, Baja -> Bajo
            [3, 2, 2],  # Reputacional, Alto, Alta -> Crítico
            [0, 1, 1],  # Tecnológico, Medio, Media -> Medio
            [1, 2, 2],  # Financiero, Alto, Alta -> Crítico
            [2, 1, 0],  # Operativo, Medio, Baja -> Medio
            [3, 0, 1],  # Reputacional, Bajo, Media -> Bajo
            [0, 2, 0],  # Tecnológico, Alto, Baja -> Medio
        ]
        y_train = [3, 2, 2, 0, 3, 1, 3, 1, 0, 1]  # 0=Bajo, 1=Medio, 2=Alto, 3=Crítico
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict_risk(self, tipo_riesgo, impacto, probabilidad):
        """Predice el nivel de riesgo usando ML"""
        if not self.is_trained:
            self.train_model()
            
        # Mapeo manual de valores
        tipo_map = {'Tecnológico': 0, 'Financiero': 1, 'Operativo': 2, 'Reputacional': 3, 'Legal': 4}
        impacto_map = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
        prob_map = {'Baja': 0, 'Media': 1, 'Alta': 2}
        
        tipo_encoded = tipo_map.get(tipo_riesgo, 0)
        impacto_encoded = impacto_map.get(impacto, 0)
        prob_encoded = prob_map.get(probabilidad, 0)
        
        prediction = self.model.predict([[tipo_encoded, impacto_encoded, prob_encoded]])[0]
        proba = self.model.predict_proba([[tipo_encoded, impacto_encoded, prob_encoded]])[0]
        
        nivel_map = {0: 'Bajo', 1: 'Medio', 2: 'Alto', 3: 'Crítico'}
        return nivel_map[prediction], proba

# Instancia global del predictor
risk_predictor = RiskPredictor()

# ==================== CONFIGURACIÓN DE PLANES ====================
PLANES = {
    'gratuito': {
        'nombre': 'Plan Gratuito',
        'diagnosticos_mes': 3,  # Número entero, no float('inf')
        'archivos_mes': 1,
        'reportes_pdf': False,
        'precio': 0
    },
    'premium': {
        'nombre': 'Plan Premium',
        'diagnosticos_mes': 999999,  # Número grande para "ilimitado"
        'archivos_mes': 20,
        'reportes_pdf': True,
        'precio': 49900
    },
    'empresarial': {
        'nombre': 'Plan Empresarial',
        'diagnosticos_mes': 999999,  # Número grande para "ilimitado"
        'archivos_mes': 999999,
        'reportes_pdf': True,
        'precio': 99900
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
    """Obtiene estadísticas del usuario - VERSIÓN CORREGIDA"""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        
        # Diagnosticos realizados (total)
        cursor.execute("SELECT COUNT(*) as total FROM diagnosticos WHERE usuario_id = %s", (user_id,))
        diagnosticos = cursor.fetchone()['total']
        
        # Archivos analizados
        cursor.execute("SELECT COUNT(*) as total FROM archivos_analizados WHERE usuario_id = %s", (user_id,))
        archivos = cursor.fetchone()['total']
        
        # Diagnosticos este mes - USAR fecha_analisis
        cursor.execute("""
            SELECT COUNT(*) as total FROM diagnosticos 
            WHERE usuario_id = %s AND MONTH(fecha_analisis) = MONTH(CURRENT_DATE())
        """, (user_id,))
        diagnosticos_mes = cursor.fetchone()['total']
        
        return {
            'diagnosticos_realizados': diagnosticos,
            'archivos_analizados': archivos,
            'diagnosticos_este_mes': diagnosticos_mes
        }

def calcular_diagnosticos_restantes(user_id):
    """Calcula diagnósticos restantes este mes - VERSIÓN CORREGIDA"""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return 0
        
        # Admin tiene acceso ilimitado
        if user['rol'] == 'admin':
            return 'Ilimitados'
        
        plan = PLANES.get(user['plan'], PLANES['gratuito'])
        
        # Si el plan es ilimitado
        if plan['diagnosticos_mes'] == 999999:
            return 'Ilimitados'
        
        # Obtener diagnósticos este mes
        cursor.execute("""
            SELECT COUNT(*) as total FROM diagnosticos 
            WHERE usuario_id = %s AND MONTH(fecha_analisis) = MONTH(CURRENT_DATE())
        """, (user_id,))
        diagnosticos_mes = cursor.fetchone()['total']
        
        # Calcular restantes
        restantes = plan['diagnosticos_mes'] - diagnosticos_mes
        
        # Asegurarse de que no sea negativo
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
    """Verifica si el usuario puede realizar una acción"""
    with get_db() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return False, "Usuario no encontrado"
        
        # Admin sin límites
        if user['rol'] == 'admin':
            return True, "Acceso completo"
        
        plan = PLANES.get(user['plan'], PLANES['gratuito'])
        
        if tipo == 'diagnostico':
            limite = plan['diagnosticos_mes']
            usado = user['diagnosticos_este_mes']
        elif tipo == 'archivo':
            limite = plan['archivos_mes']
            usado = user['archivos_este_mes']
        else:
            return False, "Tipo no válido"
        
        if limite == float('inf'):
            return True, f"Ilimitado"
        
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
    """Analiza el riesgo basado en el formulario usando ML"""
    PUNTUACION = {
        'tipo_riesgo': {'Tecnológico': 30, 'Financiero': 25, 'Operativo': 20, 'Reputacional': 25, 'Legal': 22},
        'impacto': {'Alto': 40, 'Medio': 25, 'Bajo': 10},
        'probabilidad': {'Alta': 30, 'Media': 20, 'Baja': 10}
    }
    
    # Puntuación tradicional
    puntuacion = (
        PUNTUACION['tipo_riesgo'].get(form_data['tipo_riesgo'], 0) +
        PUNTUACION['impacto'].get(form_data['impacto'], 0) +
        PUNTUACION['probabilidad'].get(form_data['probabilidad'], 0)
    )
    
    # Predicción ML
    nivel_ml, probabilidades = risk_predictor.predict_risk(
        form_data['tipo_riesgo'],
        form_data['impacto'],
        form_data['probabilidad']
    )
    
    # Ajustar puntuación con ML (50% tradicional + 50% ML)
    nivel_map = {'Bajo': 25, 'Medio': 50, 'Alto': 75, 'Crítico': 95}
    puntuacion_ml = nivel_map[nivel_ml]
    puntuacion_final = (puntuacion * 0.5) + (puntuacion_ml * 0.5)
    
    # Determinar nivel final
    if puntuacion_final >= 75:
        nivel = 'Crítico'
        color = 'danger'
    elif puntuacion_final >= 55:
        nivel = 'Alto'
        color = 'warning'
    elif puntuacion_final >= 35:
        nivel = 'Medio'
        color = 'info'
    else:
        nivel = 'Bajo'
        color = 'success'
    
    recomendaciones = generar_recomendaciones(nivel, form_data['tipo_riesgo'])
    
    return {
        'empresa': form_data['empresa'],
        'tipo_riesgo': form_data['tipo_riesgo'],
        'impacto': form_data['impacto'],
        'probabilidad': form_data['probabilidad'],
        'puntuacion_final': round(puntuacion_final, 2),
        'nivel_riesgo': nivel,
        'color_riesgo': color,
        'recomendaciones': recomendaciones,
        'observaciones': form_data.get('observaciones', ''),
        'ml_prediction': nivel_ml,
        'ml_confidence': {
            'Bajo': round(probabilidades[0] * 100, 2),
            'Medio': round(probabilidades[1] * 100, 2),
            'Alto': round(probabilidades[2] * 100, 2),
            'Crítico': round(probabilidades[3] * 100, 2)
        }
    }

def generar_recomendaciones(nivel, tipo_riesgo):
    """Genera recomendaciones según el nivel de riesgo"""
    recomendaciones = []
    
    if nivel in ['Crítico', 'Alto']:
        recomendaciones.extend([
            {'prioridad': 'URGENTE', 'titulo': 'Implementar plan de respuesta a incidentes'},
            {'prioridad': 'Alta', 'titulo': 'Auditoría de seguridad completa'}
        ])
    
    if tipo_riesgo == 'Tecnológico':
        recomendaciones.extend([
            {'prioridad': 'Alta', 'titulo': 'Actualizar sistemas antivirus'},
            {'prioridad': 'Media', 'titulo': 'Configurar firewall empresarial'}
        ])
    elif tipo_riesgo == 'Financiero':
        recomendaciones.extend([
            {'prioridad': 'Alta', 'titulo': 'Implementar autenticación de dos factores'},
            {'prioridad': 'Alta', 'titulo': 'Monitoreo de transacciones'}
        ])
    
    recomendaciones.append({'prioridad': 'Media', 'titulo': 'Capacitación del personal'})
    
    return recomendaciones

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
        plan = request.form.get('plan', 'gratuito')
        
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
            
            session['guest_diagnosticos'] += 1
            
            form_data = {
                'empresa': request.form.get('empresa'),
                'tipo_riesgo': request.form.get('tipo_riesgo'),
                'impacto': request.form.get('impacto'),
                'probabilidad': request.form.get('probabilidad'),
                'observaciones': request.form.get('observaciones', '')
            }
            
            resultado = analizar_riesgo(form_data)
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
        
        form_data = {
            'empresa': request.form.get('empresa'),
            'tipo_riesgo': request.form.get('tipo_riesgo'),
            'impacto': request.form.get('impacto'),
            'probabilidad': request.form.get('probabilidad'),
            'observaciones': request.form.get('observaciones', '')
        }
        
        resultado = analizar_riesgo(form_data)
        
        # Guardar en BD
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
    INSERT INTO diagnosticos 
    (usuario_id, empresa, tipo_riesgo, impacto, probabilidad, observaciones, puntuacion_final, nivel_riesgo)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
""", (session['user_id'], form_data['empresa'], form_data['tipo_riesgo'], 
        form_data['impacto'], form_data['probabilidad'], form_data['observaciones'],
        resultado['puntuacion_final'], resultado['nivel_riesgo']))
            diagnostico_id = cursor.lastrowid
            
            # Guardar recomendaciones
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
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
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
    
    # ========== NUEVO: DATOS PARA GRÁFICAS CORREGIDAS ==========
    analisis['graficas'] = {
        'anomalias_por_tipo': generar_datos_grafica_anomalias(analisis['anomalias']),
        'nulos_por_columna': generar_datos_grafica_nulos(df),
        'distribucion_valores': generar_datos_distribucion(df)
    }
    
    return analisis

# ==================== NUEVAS FUNCIONES: MAPA DE CALOR ====================
def generar_mapa_calor(df, anomalias):
    """Genera un mapa de calor para visualizar anomalías por celda"""
    mapa = []
    
    for idx, row in df.head(50).iterrows():
        fila_calor = {}
        for col in df.columns:
            nivel_riesgo = 0  # 0=Sin riesgo, 1=Bajo, 2=Medio, 3=Alto
            
            # Verificar si hay valor nulo
            if pd.isna(row[col]) or row[col] == '':
                nivel_riesgo = 3
            
            # Verificar si es un outlier numérico
            elif pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                if row[col] < lower or row[col] > upper:
                    nivel_riesgo = 2
            
            fila_calor[col] = nivel_riesgo
        
        mapa.append(fila_calor)
    
    return mapa

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
    puntuacion = 0.0
    anomalias = []

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
                    puntuacion += 30 * scale
                    anomalias.append({
                        'tipo': 'VALORES NULOS CRÍTICOS',
                        'columna': columna,
                        'cantidad': int(nulos),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
                elif porcentaje > 5:
                    puntuacion += 15 * scale
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
                    puntuacion += 8 * scale
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
                puntuacion += 25 * scale
                anomalias.append({
                    'tipo': 'DUPLICADOS EXACTOS - POSIBLE ERROR',
                    'cantidad': duplicados,
                    'porcentaje': f'{porcentaje_dup:.1f}%',
                    'categoria': 'Integridad'
                })
            elif porcentaje_dup > 0:
                puntuacion += 10 * scale
                anomalias.append({
                    'tipo': 'Registros duplicados',
                    'cantidad': duplicados,
                    'porcentaje': f'{porcentaje_dup:.1f}%',
                    'categoria': 'Integridad'
                })
        else:
            if porcentaje_dup > 20:
                puntuacion += 6 * scale
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
                    puntuacion += 35 * scale
                    anomalias.append({
                        'tipo': 'TRANSACCIONES REDONDAS SOSPECHOSAS',
                        'columna': col,
                        'cantidad': int(transacciones_redondas),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Fraude'
                    })
                elif porcentaje > 5:
                    puntuacion += 12 * scale
                    anomalias.append({
                        'tipo': 'Transacciones redondas',
                        'columna': col,
                        'cantidad': int(transacciones_redondas),
                        'porcentaje': f'{porcentaje:.1f}%',
                        'categoria': 'Patrón Inusual'
                    })
            else:
                if porcentaje > 50:
                    puntuacion += 6 * scale
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
                puntuacion += 30 * scale
                anomalias.append({
                    'tipo': 'VALORES EXTREMOS EN MONTOS',
                    'columna': col,
                    'cantidad': int(outliers_financieros),
                    'porcentaje': f'{porcentaje:.1f}%',
                    'categoria': 'Riesgo Financiero'
                })
            elif porcentaje > 25:
                puntuacion += 8 * scale
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
                puntuacion += 20 * scale
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
                puntuacion += 25 * scale
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
            puntuacion += 12 * scale
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
                puntuacion += 12 * scale
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
            puntuacion += 30 * scale
        else:
            puntuacion += 10 * scale
        anomalias.append({
            'tipo': 'ASIENTOS CONTABLES DESBALANCEADOS',
            'cantidad': int(desbalance),
            'categoria': 'Error Contable'
        })

    # Huecos en secuencia (muestras pequeñas penalizadas menos)
    for col in df.select_dtypes(include=[np.number]).columns:
        huecos_secuencia = int(detectar_huecos_secuencia(df[col]))
        if huecos_secuencia > 0:
            puntuacion += 6 * scale
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
                    puntuacion += 25 * scale
                    anomalias.append({
                        'tipo': 'OUTLIERS CRÍTICOS',
                        'columna': col,
                        'cantidad': int(outliers),
                        'porcentaje': f'{porcentaje_out:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
                else:
                    puntuacion += 8 * scale
                    anomalias.append({
                        'tipo': 'Valores atípicos',
                        'columna': col,
                        'cantidad': int(outliers),
                        'porcentaje': f'{porcentaje_out:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })
            else:
                if porcentaje_out > 30:
                    puntuacion += 6 * scale
                    anomalias.append({
                        'tipo': 'Valores atípicos (muestra pequeña)',
                        'columna': col,
                        'cantidad': int(outliers),
                        'porcentaje': f'{porcentaje_out:.1f}%',
                        'categoria': 'Calidad de Datos'
                    })

    # ========== CLASIFICACIÓN DE RIESGO ==========

    puntuacion_final = min(100, round(puntuacion, 2))

    if puntuacion_final >= 85:
        nivel_riesgo = 'Muy Alto'
    elif puntuacion_final >= 60:
        nivel_riesgo = 'Alto'
    elif puntuacion_final >= 35:
        nivel_riesgo = 'Medio'
    else:
        nivel_riesgo = 'Bajo'

    return {
        'total_registros': total_registros,
        'total_columnas': total_columnas,
        'puntuacion_riesgo': puntuacion_final,
        'nivel_riesgo': nivel_riesgo,
        'anomalias': anomalias,
        'metricas_avanzadas': {
            'columnas_monetarias': columnas_monetarias,
            'columnas_fecha': columnas_fecha,
            'total_anomalias_financieras': len([a for a in anomalias if a.get('categoria') in ['Fraude', 'Riesgo Financiero']])
        }
    }

#----- FUNCIONES DE DETECCIÓN FINANCIERA AVANZADA ----- 
# ========== FUNCIONES DE DETECCIÓN FINANCIERA ==========

def identificar_columnas_monetarias(df):
    """Identifica columnas que probablemente contengan valores monetarios"""
    columnas_monetarias = []
    palabras_clave = ['monto', 'valor', 'importe', 'amount', 'total', 'saldo', 'precio', 
                     'costo', 'debito', 'credito', 'balance', 'payment', 'pago']
    
    for col in df.columns:
        col_lower = col.lower()
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
    palabras_clave = ['fecha', 'date', 'time', 'hora', 'dia', 'mes', 'año', 'timestamp']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(palabra in col_lower for palabra in palabras_clave):
            columnas_fecha.append(col)
        # Verificar si la columna es de tipo datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            columnas_fecha.append(col)
    
    return list(set(columnas_fecha))

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
    """Detecta si hay desbalance entre débitos y créditos"""
    columnas_debito = [col for col in df.columns if 'debito' in col.lower() or 'debit' in col.lower()]
    columnas_credito = [col for col in df.columns if 'credito' in col.lower() or 'credit' in col.lower()]
    
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
            plan_info = PLANES.get(user['plan'], PLANES['gratuito'])
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
            SELECT DATE_FORMAT(fecha_analisis, '%Y-%m') as mes, 
                   COUNT(*) as total 
            FROM diagnosticos 
            WHERE fecha_analisis >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
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
    plan = request.form.get('plan', 'gratuito')
    
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
