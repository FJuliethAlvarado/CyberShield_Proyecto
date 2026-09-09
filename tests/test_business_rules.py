import tempfile
from pathlib import Path

import pandas as pd

import app


def test_plan_configuration_has_no_free_tier_and_has_two_paid_plans():
    assert 'gratuito' not in app.PLANES
    assert 'lite' in app.PLANES
    assert 'premium' in app.PLANES
    assert app.PLANES['lite']['precio'] == 29900
    assert app.PLANES['premium']['precio'] == 49900


def test_file_risk_model_scores_unhealthy_dataset_as_high_risk():
    df = pd.DataFrame({
        'monto': [100, 120, 150, 200, 50000, 110, 130, 140, 150, 160],
        'saldo': [10, 12, 11, 13, 15, 16, None, 14, 13, 12],
        'estado': ['ok', 'ok', 'ok', 'ok', 'fraud', 'ok', 'ok', 'ok', 'error', 'ok'],
        'fecha': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10']
    })

    result = app.analizar_archivo(df)

    assert result['puntuacion_riesgo'] >= 60
    assert result['nivel_riesgo'] in {'Alto', 'Crítico'}
    assert len(result['anomalias']) >= 1


def test_csv_with_latin1_encoding_is_loaded_successfully():
    path = Path(tempfile.gettempdir()) / 'cybershield_latin1_test.csv'
    path.write_text('monto;saldo;estado\n100;10;ok\n50000;15;fraud\n', encoding='latin-1')

    try:
        df = app._read_dataframe_from_file(path)
        assert not df.empty
        assert 'monto' in df.columns
        assert df.shape[0] == 2
    finally:
        if path.exists():
            path.unlink()


def test_manual_risk_analysis_uses_extended_ml_inputs():
    form_data = {
        'empresa': 'Empresa prueba',
        'tipo_riesgo': 'Financiero',
        'impacto': 'Alto',
        'probabilidad': 'Alta',
        'sector': 'Financiero',
        'tamano': 'Mediana',
        'historial_incidentes': 'Varios',
        'sensibilidad_datos': 'Alta',
        'controles_seguridad': 'Bajos',
        'copias_respaldo': 'Regulares',
        'mfa': 'Parcial',
        'observaciones': 'Se usan equipos compartidos y no hay MFA generalizado.'
    }

    result = app.analizar_riesgo(form_data)

    assert 'ml_inputs' in result
    assert result['ml_inputs']['sector'] == 'Financiero'
    assert result['ml_inputs']['tamano'] == 'Mediana'
    assert result['puntuacion_final'] >= 60
