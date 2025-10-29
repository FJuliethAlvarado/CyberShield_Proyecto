-- ============================================
-- CYBERSHIELD PYME - BASE DE DATOS
-- Sistema con 3 tipos de usuarios:
-- 1. INVITADO (sin cuenta, acceso limitado)
-- 2. PYME (con cuenta, según plan)
-- 3. ADMIN (acceso total)
-- ============================================

DROP DATABASE IF EXISTS cybershield_db;
CREATE DATABASE cybershield_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cybershield_db;

-- ============================================
-- TABLA: usuarios
-- ============================================
CREATE TABLE usuarios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    empresa VARCHAR(200) NOT NULL,
    
    -- ROL: 'pyme' o 'admin'
    rol VARCHAR(20) DEFAULT 'pyme',
    
    -- PLAN: 'gratuito', 'premium', 'empresarial'
    plan VARCHAR(20) DEFAULT 'gratuito',
    
    profile_pic VARCHAR(255) DEFAULT 'default-avatar.png',
    
    -- Contadores
    diagnosticos_realizados INT DEFAULT 0,
    archivos_analizados INT DEFAULT 0,
    diagnosticos_este_mes INT DEFAULT 0,
    archivos_este_mes INT DEFAULT 0,
    
    fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP,
    fecha_reset_contador DATE DEFAULT (CURRENT_DATE),
    activo BOOLEAN DEFAULT TRUE,
    
    INDEX idx_username (username),
    INDEX idx_rol (rol)
) ENGINE=InnoDB;

-- ============================================
-- TABLA: diagnosticos (diagnósticos manuales)
-- ============================================
CREATE TABLE diagnosticos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    usuario_id INT NOT NULL,
    
    -- Datos del formulario
    empresa VARCHAR(200) NOT NULL,
    tipo_riesgo VARCHAR(50) NOT NULL,
    impacto VARCHAR(20) NOT NULL,
    probabilidad VARCHAR(20) NOT NULL,
    observaciones TEXT,
    
    -- Resultados
    puntuacion_final DECIMAL(5,2),
    nivel_riesgo VARCHAR(20),
    
    fecha_analisis DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- TABLA: recomendaciones
-- ============================================
CREATE TABLE recomendaciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    diagnostico_id INT NOT NULL,
    
    titulo VARCHAR(255),
    descripcion TEXT,
    prioridad VARCHAR(20),
    
    FOREIGN KEY (diagnostico_id) REFERENCES diagnosticos(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- TABLA: archivos_analizados
-- ============================================
CREATE TABLE archivos_analizados (
    id INT AUTO_INCREMENT PRIMARY KEY,
    usuario_id INT NOT NULL,
    
    archivo_nombre VARCHAR(255) NOT NULL,
    total_registros INT,
    puntuacion_riesgo DECIMAL(5,2),
    nivel_riesgo VARCHAR(20),
    
    fecha_analisis DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- INSERTAR USUARIOS DE PRUEBA
-- ============================================

-- ADMIN (acceso total)
INSERT INTO usuarios (username, password, email, empresa, rol, plan) VALUES 
('admin', 'pbkdf2:sha256:600000$XYZ$hash_temporal', 'admin@cybershield.com', 'CyberShield PyME', 'admin', 'empresarial');

-- PYME GRATUITA
INSERT INTO usuarios (username, password, email, empresa, rol, plan) VALUES 
('demo', 'pbkdf2:sha256:600000$XYZ$hash_temporal', 'demo@pyme.com', 'PyME Demo', 'pyme', 'gratuito');

-- PYME PREMIUM
INSERT INTO usuarios (username, password, email, empresa, rol, plan) VALUES 
('premium', 'pbkdf2:sha256:600000$XYZ$hash_temporal', 'premium@pyme.com', 'PyME Premium', 'pyme', 'premium');

-- ============================================
-- VERIFICACIÓN
-- ============================================
SELECT 'Base de datos creada' AS mensaje;
SHOW TABLES;
SELECT username, rol, plan FROM usuarios;