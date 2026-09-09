-- ============================================
-- CYBERSHIELD PYME - BASE DE DATOS POSTGRES (Render)
-- Sistema con 2 planes de pago:
-- 1. LITE (limitado)
-- 2. PREMIUM (completo)
-- 3. ADMIN (acceso total)
-- ============================================

CREATE TABLE IF NOT EXISTS usuarios (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    empresa VARCHAR(200) NOT NULL,

    rol VARCHAR(20) DEFAULT 'pyme',
    plan VARCHAR(20) DEFAULT 'lite',
    profile_pic VARCHAR(255) DEFAULT 'default-avatar.png',

    diagnosticos_realizados INT DEFAULT 0,
    archivos_analizados INT DEFAULT 0,
    diagnosticos_este_mes INT DEFAULT 0,
    archivos_este_mes INT DEFAULT 0,

    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_reset_contador DATE DEFAULT CURRENT_DATE,
    activo BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_usuarios_username ON usuarios(username);
CREATE INDEX IF NOT EXISTS idx_usuarios_rol ON usuarios(rol);

CREATE TABLE IF NOT EXISTS diagnosticos (
    id SERIAL PRIMARY KEY,
    usuario_id INT NOT NULL,
    empresa VARCHAR(200) NOT NULL,
    tipo_riesgo VARCHAR(50) NOT NULL,
    impacto VARCHAR(20) NOT NULL,
    probabilidad VARCHAR(20) NOT NULL,
    observaciones TEXT,
    puntuacion_final DECIMAL(5,2),
    nivel_riesgo VARCHAR(20),
    fecha_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_diagnosticos_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recomendaciones (
    id SERIAL PRIMARY KEY,
    diagnostico_id INT NOT NULL,
    titulo VARCHAR(255),
    descripcion TEXT,
    prioridad VARCHAR(20),
    CONSTRAINT fk_recomendaciones_diagnostico FOREIGN KEY (diagnostico_id) REFERENCES diagnosticos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archivos_analizados (
    id SERIAL PRIMARY KEY,
    usuario_id INT NOT NULL,
    archivo_nombre VARCHAR(255) NOT NULL,
    total_registros INT,
    puntuacion_riesgo DECIMAL(5,2),
    nivel_riesgo VARCHAR(20),
    fecha_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_archivos_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

INSERT INTO usuarios (username, password, email, empresa, rol, plan)
VALUES
    ('admin', 'pbkdf2:sha256:600000$XYZ$hash_temporal', 'admin@cybershield.com', 'CyberShield PyME', 'admin', 'premium')
ON CONFLICT (username) DO NOTHING;

INSERT INTO usuarios (username, password, email, empresa, rol, plan)
VALUES
    ('demo', 'pbkdf2:sha256:600000$XYZ$hash_temporal', 'demo@pyme.com', 'PyME Demo', 'pyme', 'lite'),
    ('premium', 'pbkdf2:sha256:600000$XYZ$hash_temporal', 'premium@pyme.com', 'PyME Premium', 'pyme', 'premium')
ON CONFLICT (username) DO NOTHING;

SELECT 'Base de datos PostgreSQL creada' AS mensaje;
SELECT username, rol, plan FROM usuarios;