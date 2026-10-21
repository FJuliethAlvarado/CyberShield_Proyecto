import tempfile
from pathlib import Path

import pandas as pd
import pytest

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


def test_accounting_balance_detection_recognizes_accented_spanish_headers():
    balanced = pd.DataFrame({
        'Débito': [100, 25],
        'Crédito': [50, 75],
    })
    unbalanced = pd.DataFrame({
        'Debe': [100, 25],
        'Haber': [50, 60],
    })

    assert app.detectar_asientos_desbalanceados(balanced) == 0
    assert app.detectar_asientos_desbalanceados(unbalanced) == 1


def test_anomaly_explanation_warns_that_round_amounts_do_not_prove_fraud():
    explanation = app.explicar_anomalia({'tipo': 'TRANSACCIONES REDONDAS SOSPECHOSAS'})

    assert 'no demuestra fraude' in explanation['accion_sugerida']


def test_accounting_sample_files_cover_balanced_unbalanced_and_data_quality_cases():
    samples_dir = Path(__file__).resolve().parents[1] / 'samples'
    balanced = app._read_dataframe_from_file(samples_dir / 'movimientos_contables_balanceados.csv')
    unbalanced = app._read_dataframe_from_file(samples_dir / 'movimientos_contables_desbalanceados.csv')
    data_quality = app._read_dataframe_from_file(samples_dir / 'movimientos_contables_alertas.csv')

    assert app.detectar_asientos_desbalanceados(balanced) == 0
    assert app.detectar_asientos_desbalanceados(unbalanced) == 1

    alert_types = {anomaly['tipo'] for anomaly in app.analizar_archivo(data_quality)['anomalias']}
    assert 'Valores nulos' in alert_types
    assert 'DUPLICADOS EXACTOS - POSIBLE ERROR' in alert_types
    assert 'VALORES EXTREMOS EN MONTOS' in alert_types


def test_heatmap_marks_cells_matching_detected_alerts_in_preview():
    samples_dir = Path(__file__).resolve().parents[1] / 'samples'
    df = app._read_dataframe_from_file(samples_dir / 'movimientos_contables_alertas.csv')
    analysis = app.analizar_archivo(df)
    heatmap = app.generar_mapa_calor(df, analysis['anomalias'])

    missing_cell = df['Debito'].isna().to_numpy().nonzero()[0][0]
    assert heatmap[missing_cell]['Debito'] == 3
    assert any(level > 0 for row in heatmap for level in row.values())


def test_heatmap_marks_each_row_correctly_instead_of_using_column_index():
    df = pd.DataFrame({
        'Importe': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1000],
        'Descripcion': list('ABCDEFGHIJK'),
    })
    analysis = app.analizar_archivo(df)
    heatmap = app.generar_mapa_calor(df, analysis['anomalias'])

    assert any(alert['familia'] == 'valores_atipicos' for alert in analysis['anomalias'])
    assert heatmap[10]['Importe'] > 0
    assert heatmap[0]['Importe'] == 0


def test_heatmap_marks_debit_and_credit_columns_when_their_totals_do_not_match():
    df = pd.DataFrame({
        'Debe': [100, 25],
        'Haber': [50, 60],
    })
    analysis = app.analizar_archivo(df)
    heatmap = app.generar_mapa_calor(df, analysis['anomalias'])

    assert all(row['Debe'] == 2 and row['Haber'] == 2 for row in heatmap)


def test_sequence_gaps_only_apply_to_columns_that_look_like_voucher_numbers():
    ordinary_numbers = pd.DataFrame({
        'Saldo': [100, 250, 400, 650, 900],
        'Cantidad': [1, 3, 5, 7, 9],
    })
    voucher_numbers = pd.DataFrame({
        'Numero comprobante': [101, 102, 104, 105, 108],
        'Importe': [17, 29, 31, 43, 59],
    })

    ordinary_alerts = app.analizar_archivo(ordinary_numbers)['anomalias']
    voucher_alerts = app.analizar_archivo(voucher_numbers)['anomalias']

    assert not any(alert['familia'] == 'numeracion_comprobantes' for alert in ordinary_alerts)
    sequence_alert = next(alert for alert in voucher_alerts if alert['familia'] == 'numeracion_comprobantes')
    assert 'por ejemplo' in sequence_alert['accion_sugerida'].lower()
    assert sequence_alert['columnas_afectadas'][0]['columna'] == 'Numero comprobante'


def test_repeated_amount_alerts_are_grouped_by_family_and_column():
    df = pd.DataFrame({
        'Debito': [1000 + i * 100 for i in range(10)],
        'Credito': [500 + i * 100 for i in range(10)],
        'Importe': [1500 + i * 200 for i in range(10)],
        'Saldo': [1400 + i * 150 for i in range(10)],
    })

    result = app.analizar_archivo(df)
    round_alerts = [alert for alert in result['anomalias'] if alert['familia'] == 'importes_redondos']

    assert len(round_alerts) == 1
    assert {item['columna'] for item in round_alerts[0]['columnas_afectadas']} >= {'Debito', 'Credito', 'Importe'}


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


def test_structured_risk_analysis_explains_inherent_and_residual_risk():
    form_data = {
        'empresa': 'Empresa prueba',
        'sector': 'Financiero',
        'tamano': 'Mediana',
        'dependencia_sistemas': 'alta',
        'tiempo_recuperacion': 'no_puede_operar',
        'proveedores': 'varios',
        'tipo_datos': 'financiera',
        'acceso_remoto': 'debil',
        'servicios_nube': 'desconocidos',
        'historial_incidentes': 'multiples',
        'mfa': 'algunos',
        'gestion_cuentas': 'parcial',
        'actualizaciones': 'ocasionales',
        'proteccion_equipos': 'algunos',
        'copias_respaldo': 'ocasionales',
        'prueba_respaldo': 'problemas',
        'aislamiento_respaldo': 'parcial',
        'respuesta_incidentes': 'informal',
        'capacitacion': 'ocasional',
        'observaciones': 'Se usan equipos compartidos y no hay MFA generalizado.'
    }

    result = app.analizar_riesgo(form_data)

    assert result['tipo_riesgo'] == 'Financiero'
    assert result['puntuacion_inherente'] > result['puntuacion_final']
    assert result['nivel_riesgo'] in {'Alto', 'Crítico'}
    assert result['hallazgos']
    assert 'ml_confidence' not in result


def test_unknown_controls_are_not_counted_as_effective_and_reduce_completeness():
    data = {
        'empresa': 'Empresa prueba',
        'sector': 'Servicios',
        'tamano': 'Pequeña',
        'dependencia_sistemas': 'parcial',
        'tiempo_recuperacion': '1_3_dias',
        'proveedores': 'algunos',
        'tipo_datos': 'contacto',
        'acceso_remoto': 'no_se',
        'servicios_nube': 'no_se',
        'historial_incidentes': 'no_se',
        'mfa': 'no_se',
        'gestion_cuentas': 'no_se',
        'actualizaciones': 'no_se',
        'proteccion_equipos': 'no_se',
        'copias_respaldo': 'no_se',
        'prueba_respaldo': 'no_se',
        'aislamiento_respaldo': 'no_se',
        'respuesta_incidentes': 'no_se',
        'capacitacion': 'no_se',
    }

    result = app.analizar_riesgo(data)

    assert result['efectividad_controles'] == 0
    assert result['completitud'] < 100
    assert len(result['recomendaciones']) >= 1


def test_structured_risk_analysis_rejects_missing_or_invalid_answers():
    with pytest.raises(ValueError):
        app.analizar_riesgo({'empresa': 'Empresa prueba'})


def test_guest_diagnosis_route_renders_structured_result():
    form_data = {
        'empresa': 'Empresa de prueba',
        'sector': 'Servicios',
        'tamano': 'Pequeña',
        'dependencia_sistemas': 'parcial',
        'tiempo_recuperacion': '1_3_dias',
        'proveedores': 'algunos',
        'tipo_datos': 'contacto',
        'acceso_remoto': 'controlado',
        'servicios_nube': 'controlados',
        'historial_incidentes': 'uno',
        'mfa': 'algunos',
        'gestion_cuentas': 'parcial',
        'actualizaciones': 'ocasionales',
        'proteccion_equipos': 'todos',
        'copias_respaldo': 'definidas',
        'prueba_respaldo': 'exitosa',
        'aislamiento_respaldo': 'si',
        'respuesta_incidentes': 'informal',
        'capacitacion': 'ocasional',
        'observaciones': '',
    }

    with app.app.test_client() as client:
        response = client.post('/diagnostico', data=form_data)

    assert response.status_code == 200
    assert b'Evaluaci' in response.data
    assert b'Riesgo inherente' in response.data
    assert b'An\xc3\xa1lisis con IA' not in response.data
